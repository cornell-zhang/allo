# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from collections.abc import Iterable, Sequence
from typing import Any, Union

from .bindings import ir, transform as tran_d, utils
from .bindings.utils import (
    FrontendProxy,
    INVALID,
    OperationProxy,
    ProxyState,
    STALE,
    VALID,
    ValueProxy,
)

SingleProxy = Union[FrontendProxy, str]
ProxyList = Iterable[SingleProxy]
Proxies = Union[SingleProxy, ProxyList]


def _is_loop_like(proxy: FrontendProxy) -> bool:
    return proxy.kind_str in {
        "affine.for",
        "scf.for",
        "affine.parallel",
        "scf.parallel",
    }


def _state_name(state: ProxyState) -> str:
    return state.name if hasattr(state, "name") else str(state)


class Schedule:  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """Experimental schedule backed by live operation/value proxies.

    The scheduler keeps Python-side proxy indexes as the source of truth for
    payload hierarchy, identifier resolution, and handle lifecycle:
    - `VALID`: proxy can still be materialized and used by later transforms.
    - `INVALID`: proxy was directly consumed or removed by a transform.
    - `STALE`: proxy survived only for diagnostics after a refresh or ancestor
      invalidation and must be re-queried before reuse.
    """

    # partition kind enum values
    Complete = tran_d.Complete
    Block = tran_d.Block
    Cyclic = tran_d.Cyclic

    payload_root: ir.Value
    payload_tree: OperationProxy
    transform_mod: ir.ModuleOp
    transform_seq: tran_d.NamedSequenceOp

    def __init__(self, module: ir.ModuleOp, context: ir.Context):
        self.payload = module
        self.context = context
        self.context.load_transform_dialects()
        self.builder = ir.AlloOpBuilder(context)
        self.builder.set_unknown_loc()

        # public trackers
        self.by_path: dict[str, FrontendProxy] = {}
        self.by_identifier: dict[str, list[FrontendProxy]] = {}
        self.by_parent: dict[str, list[OperationProxy]] = {}
        self.epoch = 0
        self.dirty = False
        self.active_proxy: FrontendProxy | None = None

        # private trackers
        self._valid_handle_cache: dict[int, ir.Value] = {}
        self._parent_path: dict[int, str | None] = {}
        self._known_proxy_keys: set[int] = set()
        self._live_proxy_keys: set[int] = set()
        self._proxy_effects: dict[int, dict[str, Any]] = {}
        self._effect_log: list[dict[str, Any]] = []
        self._effect_counter = 0
        self._id_remap: dict[str, str] = {}

        self._rebuild_tree_from_payload()
        self._init_transform()

    #################
    # Internal State
    #################
    def _proxy_key(self, proxy: FrontendProxy) -> int:
        return id(proxy)

    def _mark_dirty(self):
        self.dirty = True

    def _next_transform_id(self, op_name: str) -> str:
        self._effect_counter += 1
        return f"{op_name}:{self._effect_counter}"

    def _record_effect(self, effect_id: str, op_name: str, **data: Any):
        entry = {"id": effect_id, "op": op_name, "epoch": self.epoch}
        entry.update(data)
        self._effect_log.append(entry)

    def _record_proxy_effect(
        self,
        proxy: FrontendProxy,
        *,
        effect_id: str,
        reason: str,
        live: bool,
    ):
        key = self._proxy_key(proxy)
        if live:
            self._proxy_effects.pop(key, None)
            return
        self._proxy_effects[key] = {
            "effect_id": effect_id,
            "reason": reason,
        }

    def _append_identifier_mapping(self, identifier: str, proxy: FrontendProxy):
        bucket = self.by_identifier.setdefault(identifier, [])
        if any(existing is proxy for existing in bucket):
            raise ValueError(
                f"Duplicate proxy registration for identifier '{identifier}'"
            )
        bucket.append(proxy)

    def _remove_identifier_mapping(self, identifier: str, proxy: FrontendProxy):
        bucket = self.by_identifier.get(identifier)
        if bucket is None:
            return
        for idx, existing in enumerate(bucket):
            if existing is proxy:
                bucket.pop(idx)
                if not bucket:
                    self.by_identifier.pop(identifier, None)
                return

    def _register_live_proxy(
        self,
        proxy: FrontendProxy,
        *,
        parent_path: str | None = None,
    ):
        key = self._proxy_key(proxy)
        existing = self.by_path.get(proxy.hierarchy_name)
        if existing is not None and existing is not proxy:
            raise ValueError(f"Duplicate hierarchy_name '{proxy.hierarchy_name}'")

        self.by_path[proxy.hierarchy_name] = proxy
        self._append_identifier_mapping(proxy.identifier, proxy)
        self._known_proxy_keys.add(key)
        self._live_proxy_keys.add(key)
        proxy.state = VALID
        self._record_proxy_effect(proxy, effect_id="", reason="", live=True)

        if isinstance(proxy, OperationProxy):
            self._parent_path[key] = parent_path
            self.by_parent.setdefault(proxy.hierarchy_name, [])
            if parent_path is not None:
                self.by_parent.setdefault(parent_path, []).append(proxy)
        else:
            self._parent_path.pop(key, None)

    def _clear_indexes(self):
        self.by_path.clear()
        self.by_identifier.clear()
        self.by_parent.clear()
        self._parent_path.clear()
        self._live_proxy_keys.clear()
        self.active_proxy = None

    def _index_value_proxy(self, proxy: ValueProxy):
        self._register_live_proxy(proxy)

    def _index_operation_proxy(
        self,
        proxy: OperationProxy,
        *,
        parent_path: str | None,
    ):
        self._register_live_proxy(proxy, parent_path=parent_path)
        for value in proxy.values:
            self._index_value_proxy(value)
        for child in proxy.children:
            self._index_operation_proxy(child, parent_path=proxy.hierarchy_name)

    def _rebuild_tree_from_payload(self):
        self._clear_indexes()
        self.payload_tree = utils.build_proxy(self.payload)
        utils.complete_hierarchy_name(self.payload_tree)
        self._index_operation_proxy(self.payload_tree, parent_path=None)
        self._select_fallback_active()

    def _init_transform(self):
        self.builder.set_unknown_loc()
        sched_mod = ir.ModuleOp(self.builder)
        sched_mod.set_attr(
            "transform.with_named_sequence", ir.UnitAttr.get(self.context)
        )
        self.transform_mod = sched_mod

        self.builder.set_insertion_point_to_start(sched_mod.get_body())
        op_ty = tran_d.OperationType.get(self.context, "builtin.module")
        seq = tran_d.NamedSequenceOp(self.builder, "__transform_main", op_ty, [])
        self.transform_seq = seq

        self.builder.set_insertion_point_to_end(seq.get_entry_block())
        tran_d.YieldOp(self.builder, [])

        self.builder.set_insertion_point_to_start(seq.get_entry_block())
        attr = ir.DictionaryAttr.get(
            self.context,
            {utils.IDENTIFIER_ATTR_NAME: self.builder.get_string_attr("allo_payload")},
        )
        self.payload_root = tran_d.MatchOp(
            self.builder, seq.get_arg_at(0), op_ty, ["builtin.module"], attr
        ).get_result_at(0)
        self._valid_handle_cache.clear()
        self._valid_handle_cache[self._proxy_key(self.payload_tree)] = self.payload_root
        self.dirty = False

    #################
    # Proxy Helpers
    #################
    def _format_proxy_status(self, proxy: FrontendProxy, action: str) -> str:
        msg = (
            f"Target proxy '{proxy.identifier}', identified by "
            f"'{proxy.hierarchy_name}', is not usable for {action} "
            f"(state={_state_name(proxy.state)})"
        )
        effect = self._proxy_effects.get(self._proxy_key(proxy))
        if effect is not None and effect["effect_id"]:
            msg += f"; last affected by {effect['effect_id']}" f" ({effect['reason']})"
        return msg

    def _ensure_proxy_usable(self, proxy: FrontendProxy, action: str):
        key = self._proxy_key(proxy)
        if proxy.state != VALID:
            raise ValueError(self._format_proxy_status(proxy, action))
        if key not in self._known_proxy_keys:
            raise ValueError(
                f"Target proxy '{proxy.identifier}', identified by "
                f"'{proxy.hierarchy_name}', does not belong to this schedule"
            )
        if key not in self._live_proxy_keys:
            effect = self._proxy_effects.get(key)
            if effect is not None and effect["effect_id"]:
                raise ValueError(self._format_proxy_status(proxy, action))
            raise ValueError(
                f"Target proxy '{proxy.identifier}', identified by "
                f"'{proxy.hierarchy_name}', is detached from the live schedule state"
            )

    def _normalize_targets(self, targets: Proxies, desc: str) -> list[SingleProxy]:
        if isinstance(targets, (FrontendProxy, str)):
            targets = [targets]
        result = list(targets)
        if len(result) == 0:
            raise ValueError(f"{desc} requires at least one target")
        return result

    def _resolve_proxies(
        self,
        targets: Proxies | None,
        desc: str,
        *,
        require_loop: bool = False,
        require_value: bool = False,
        require_op: bool = True,
    ) -> list[FrontendProxy]:
        assert (
            require_op != require_value
        ), "require_op and require_value cannot both be true or both be false"
        if targets is None:
            if self.active_proxy is None:
                raise ValueError(f"No active target for {desc}, and no target provided")
            resolved = [self.active_proxy]
        else:
            resolved = []
            for target in self._normalize_targets(targets, desc):
                if isinstance(target, FrontendProxy):
                    resolved.append(target)
                    continue
                if not isinstance(target, str):
                    raise TypeError(
                        f"Invalid target type {type(target)} for {desc}, "
                        "expected FrontendProxy or str"
                    )
                matches = self.by_identifier.get(target)
                if not matches:
                    raise ValueError(
                        f"Target identifier '{target}' does not resolve to any proxy for {desc}"
                    )
                if require_op:
                    for proxy in matches:
                        if not isinstance(proxy, OperationProxy):
                            raise ValueError(
                                f"Target proxy '{proxy.identifier}', identified by "
                                f"'{proxy.hierarchy_name}', is not an operation proxy for {desc}"
                            )
                        if require_loop and not _is_loop_like(proxy):
                            raise ValueError(
                                f"Target proxy '{proxy.identifier}', identified by "
                                f"'{proxy.hierarchy_name}', is not loop-like for {desc}"
                            )
                        resolved.append(proxy)
                if require_value:
                    for proxy in matches:
                        if isinstance(proxy, ValueProxy):
                            resolved.append(proxy)

        for proxy in resolved:
            self._ensure_proxy_usable(proxy, desc)

        keys = [self._proxy_key(proxy) for proxy in resolved]
        assert len(keys) == len(set(keys)), (
            f"Internal invariant violated: _resolve_proxies produced duplicate "
            f"proxies for {desc}"
        )
        return resolved

    def _proxy_depth(self, proxy: OperationProxy) -> int:
        depth = 0
        parent_path = self._parent_path.get(self._proxy_key(proxy))
        while parent_path is not None:
            depth += 1
            parent = self.by_path.get(parent_path)
            if not isinstance(parent, OperationProxy):
                break
            parent_path = self._parent_path.get(self._proxy_key(parent))
        return depth

    def _is_strict_parent_chain(self, proxies: list[OperationProxy]) -> bool:
        if len(proxies) <= 1:
            return True
        return all(
            self._parent_path.get(self._proxy_key(proxies[idx]))
            == proxies[idx - 1].hierarchy_name
            for idx in range(1, len(proxies))
        )

    def _is_perfect_loop_band(self, proxies: list[OperationProxy]) -> bool:
        if not self._is_strict_parent_chain(proxies):
            return False
        return all(
            self.by_parent.get(parent.hierarchy_name, []) == [child]
            for parent, child in zip(proxies, proxies[1:])
        )

    def _select_fallback_active(self):
        for proxy in self._get_descendants(self.payload_tree, include_root=True):
            if proxy.state == VALID and _is_loop_like(proxy):
                self.active_proxy = proxy
                return
        self.active_proxy = (
            self.payload_tree if self.payload_tree.state == VALID else None
        )

    def _set_active_proxy(self, proxy: FrontendProxy | None):
        self.active_proxy = proxy

    def _create_proxy(
        self,
        identifier: str,
        hier_name: str,
        kind_str: str,
    ) -> OperationProxy:
        proxy = utils.create_operation_proxy(identifier, kind_str, hier_name)
        self._known_proxy_keys.add(self._proxy_key(proxy))
        return proxy

    def _create_value_proxy(
        self,
        owner: OperationProxy,
        identifier: str,
        hier_name: str,
        *,
        kind_str: str = "buffer",
        number: int = 0,
    ) -> ValueProxy:
        value = owner.add_value(identifier, kind_str, hier_name, number)
        self._known_proxy_keys.add(self._proxy_key(value))
        return value

    def _provide_proxies(
        self, proxies: list[FrontendProxy], handles: list[ir.Value | None]
    ):
        if len(proxies) != len(handles):
            raise ValueError(
                f"_provide_proxies expects equal lengths, got "
                f"{len(proxies)} proxies and {len(handles)} handles"
            )
        for proxy, handle in zip(proxies, handles):
            key = self._proxy_key(proxy)
            if key in self._live_proxy_keys:
                raise ValueError(
                    f"Proxy '{proxy.hierarchy_name}' is already live, cannot provide twice"
                )
            self._register_live_proxy(proxy)
            if handle is not None:
                self._valid_handle_cache[key] = handle

    def _update_owned_value_paths(self, owner: OperationProxy, old_hier_name: str):
        if old_hier_name == owner.hierarchy_name:
            return
        for value in owner.values:
            old_value_path = value.hierarchy_name
            if not old_value_path.startswith(old_hier_name + ":"):
                continue
            suffix = old_value_path[len(old_hier_name) :]
            default_identifier = value.identifier == old_value_path
            value.hierarchy_name = owner.hierarchy_name + suffix
            self.by_path.pop(old_value_path, None)
            self.by_path[value.hierarchy_name] = value
            if default_identifier:
                self._remove_identifier_mapping(old_value_path, value)
                value.identifier = value.hierarchy_name
                self._append_identifier_mapping(value.identifier, value)

    def _inplace_update_proxies(
        self,
        proxies: Sequence[FrontendProxy],
        handles: Sequence[ir.Value],
        identifiers: list[str] | None = None,
        hier_names: list[str] | None = None,
        kind_strs: list[str] | None = None,
    ):
        if len(proxies) != len(handles):
            raise ValueError(
                f"_inplace_update_proxies expects equal lengths, got "
                f"{len(proxies)} proxies and {len(handles)} handles"
            )
        if identifiers is not None and len(identifiers) != len(proxies):
            raise ValueError("identifier count does not match proxy count")
        if hier_names is not None and len(hier_names) != len(proxies):
            raise ValueError("hierarchy_name count does not match proxy count")
        if kind_strs is not None and len(kind_strs) != len(proxies):
            raise ValueError("kind_str count does not match proxy count")

        for idx, (proxy, handle) in enumerate(zip(proxies, handles)):
            key = self._proxy_key(proxy)
            if key not in self._live_proxy_keys:
                raise ValueError(
                    f"Cannot inplace-update detached proxy '{proxy.hierarchy_name}'"
                )

            old_identifier = proxy.identifier
            old_hier_name = proxy.hierarchy_name

            if identifiers is not None and identifiers[idx] != old_identifier:
                self._remove_identifier_mapping(old_identifier, proxy)
                proxy.identifier = identifiers[idx]
                self._append_identifier_mapping(proxy.identifier, proxy)

            if kind_strs is not None:
                proxy.kind_str = kind_strs[idx]

            if hier_names is not None and hier_names[idx] != old_hier_name:
                proxy.hierarchy_name = hier_names[idx]
                existing = self.by_path.get(proxy.hierarchy_name)
                if existing is not None and existing is not proxy:
                    raise ValueError(
                        f"Duplicate hierarchy_name '{proxy.hierarchy_name}' during inplace update"
                    )
                self.by_path.pop(old_hier_name, None)
                self.by_path[proxy.hierarchy_name] = proxy

                if isinstance(proxy, OperationProxy):
                    children = self.by_parent.pop(old_hier_name, [])
                    self.by_parent[proxy.hierarchy_name] = children
                    for child in children:
                        self._parent_path[self._proxy_key(child)] = proxy.hierarchy_name
                    self._update_owned_value_paths(proxy, old_hier_name)

            proxy.state = VALID
            self._valid_handle_cache[key] = handle
            self._record_proxy_effect(proxy, effect_id="", reason="", live=True)

    def _insert_proxies(
        self,
        proxies: list[OperationProxy],
        parent: OperationProxy,
        pos: int,
    ):
        parent_key = self._proxy_key(parent)
        if parent_key not in self._live_proxy_keys:
            raise ValueError(
                f"Cannot insert under detached parent '{parent.hierarchy_name}'"
            )

        bucket = self.by_parent.setdefault(parent.hierarchy_name, [])
        if pos < 0 or pos > len(bucket):
            raise ValueError(
                f"Insert position {pos} out of range for '{parent.hierarchy_name}'"
            )

        offset = pos
        for proxy in proxies:
            key = self._proxy_key(proxy)
            if key not in self._live_proxy_keys:
                raise ValueError(
                    f"Cannot insert detached proxy '{proxy.hierarchy_name}' before provide"
                )
            if self._parent_path.get(key) is not None:
                raise ValueError(f"Proxy '{proxy.hierarchy_name}' already has a parent")
            bucket.insert(offset, proxy)
            offset += 1
            self._parent_path[key] = parent.hierarchy_name
            self.by_parent.setdefault(proxy.hierarchy_name, [])
            proxy.parent = parent
            parent.add_child(
                pos, proxy.identifier, proxy.kind_str, proxy.hierarchy_name
            )

    def _reparent_proxy(
        self,
        proxy: OperationProxy,
        parent: OperationProxy | None,
        *,
        pos: int | None = None,
    ):
        key = self._proxy_key(proxy)
        if key not in self._live_proxy_keys:
            raise ValueError(f"Cannot reparent detached proxy '{proxy.hierarchy_name}'")

        new_parent_path = None if parent is None else parent.hierarchy_name
        old_parent_path = self._parent_path.get(key)
        if old_parent_path == new_parent_path and pos is None:
            proxy.parent = parent
            return

        if old_parent_path is not None:
            siblings = self.by_parent.get(old_parent_path)
            assert siblings is not None, (
                f"Internal invariant violated: missing sibling bucket for "
                f"'{old_parent_path}'"
            )
            removed = False
            for idx, sibling in enumerate(siblings):
                if sibling is proxy:
                    siblings.pop(idx)
                    removed = True
                    break
            assert removed, (
                f"Internal invariant violated: proxy '{proxy.hierarchy_name}' "
                "missing from its parent bucket"
            )

        self._parent_path[key] = new_parent_path
        if new_parent_path is not None:
            bucket = self.by_parent.setdefault(new_parent_path, [])
            insert_at = len(bucket) if pos is None else pos
            if insert_at < 0 or insert_at > len(bucket):
                raise ValueError(
                    f"Reparent position {insert_at} out of range for '{new_parent_path}'"
                )
            bucket.insert(insert_at, proxy)
        proxy.parent = parent

    def _splice_proxies(
        self,
        src: OperationProxy,
        dst: OperationProxy,
        start: int = 0,
    ):
        if src is dst:
            raise ValueError("_splice_proxies source and destination must differ")
        src_bucket = self.by_parent.setdefault(src.hierarchy_name, [])
        if start < 0 or start > len(src_bucket):
            raise ValueError(
                f"Splice start {start} out of range for '{src.hierarchy_name}'"
            )
        moved = src_bucket[start:]
        del src_bucket[start:]
        dst_bucket = self.by_parent.setdefault(dst.hierarchy_name, [])
        dst_bucket.extend(moved)
        for child in moved:
            self._parent_path[self._proxy_key(child)] = dst.hierarchy_name
            child.parent = dst

    def _outermost_affine_loop(self, proxy: OperationProxy) -> OperationProxy:
        root = proxy
        parent_path = self._parent_path.get(self._proxy_key(root))
        while parent_path is not None:
            parent = self.by_path.get(parent_path)
            if (
                not isinstance(parent, OperationProxy)
                or parent.kind_str != "affine.for"
            ):
                break
            root = parent
            parent_path = self._parent_path.get(self._proxy_key(root))
        return root

    def _invalidate_proxies(
        self,
        proxies: list[FrontendProxy],
        *,
        effect_id: str,
        reason: str,
        state: ProxyState = INVALID,
    ):
        for proxy in proxies:
            proxy.state = state
            self._valid_handle_cache.pop(self._proxy_key(proxy), None)
            self._record_proxy_effect(
                proxy, effect_id=effect_id, reason=reason, live=False
            )

    def _remove_proxies(
        self,
        proxies: Sequence[FrontendProxy],
        *,
        effect_id: str,
        reason: str,
    ):
        seen: set[int] = set()
        for proxy in proxies:
            key = self._proxy_key(proxy)
            if key in seen:
                continue
            seen.add(key)

            if isinstance(proxy, OperationProxy):
                owned_values = [
                    value
                    for value in proxy.values
                    if self._proxy_key(value) in self._live_proxy_keys
                ]
                if owned_values:
                    self._remove_proxies(
                        owned_values,
                        effect_id=effect_id,
                        reason=f"{reason}; owner removed",
                    )

                children = self.by_parent.get(proxy.hierarchy_name, [])
                if children:
                    raise ValueError(
                        f"Cannot remove proxy '{proxy.hierarchy_name}' while it still has children"
                    )

                parent_path = self._parent_path.pop(key, None)
                if parent_path is not None:
                    siblings = self.by_parent.get(parent_path, [])
                    for idx, sibling in enumerate(siblings):
                        if sibling is proxy:
                            siblings.pop(idx)
                            break
                self.by_parent.pop(proxy.hierarchy_name, None)
                proxy.parent = None
            else:
                self._parent_path.pop(key, None)

            self._valid_handle_cache.pop(key, None)
            self.by_path.pop(proxy.hierarchy_name, None)
            self._remove_identifier_mapping(proxy.identifier, proxy)
            self._live_proxy_keys.discard(key)
            proxy.state = INVALID
            self._record_proxy_effect(
                proxy, effect_id=effect_id, reason=reason, live=False
            )

        if (
            self.active_proxy is not None
            and self._proxy_key(self.active_proxy) not in self._live_proxy_keys
        ):
            self._select_fallback_active()

    def _get_descendants(
        self,
        proxy: OperationProxy,
        include_root: bool = False,
    ) -> list[OperationProxy]:
        out: list[OperationProxy] = []
        stack: list[OperationProxy] = []
        if include_root:
            stack.append(proxy)
        else:
            stack.extend(reversed(self.by_parent.get(proxy.hierarchy_name, [])))

        while stack:
            node = stack.pop()
            out.append(node)
            children = self.by_parent.get(node.hierarchy_name, [])
            stack.extend(reversed(children))
        return out

    def _descendant_invalidate(
        self,
        proxy: OperationProxy,
        *,
        include_root: bool = True,
    ):
        for op_proxy in self._get_descendants(proxy, include_root=include_root):
            self._valid_handle_cache.pop(self._proxy_key(op_proxy), None)
            for value in op_proxy.values:
                self._valid_handle_cache.pop(self._proxy_key(value), None)

    def _descendant_remove(self, proxy: OperationProxy, *, effect_id: str, reason: str):
        descendants = self._get_descendants(proxy, include_root=True)
        self._remove_proxies(
            list(reversed(descendants)),
            effect_id=effect_id,
            reason=reason,
        )

    def _mark_live_proxies_stale(self, *, effect_id: str, reason: str):
        current_live = list(self.by_path.values())
        self._valid_handle_cache.clear()
        for proxy in current_live:
            proxy.state = STALE
            self._record_proxy_effect(
                proxy, effect_id=effect_id, reason=reason, live=False
            )

    def _remap_id(self, old: str, new: str):
        """
        Remap an hierarchy name to a new identifier when refreshing proxies
        after a transform that changes identifiers.
        """
        self._id_remap[old] = new

    #################
    # Materialization
    #################
    def _materialize_proxies(
        self,
        proxies: FrontendProxy | Sequence[FrontendProxy],
    ) -> list[ir.Value]:
        if isinstance(proxies, FrontendProxy):
            proxies = [proxies]

        out: list[ir.Value] = []
        any_op_ty = tran_d.AnyOpType.get(self.context)
        for proxy in proxies:
            cache = self._valid_handle_cache.get(self._proxy_key(proxy))
            if cache is not None:
                out.append(cache)
                continue

            self._refresh_builder_loc_from_callsite()
            if proxy is self.payload_tree:
                handle = self.payload_root
            elif isinstance(proxy, OperationProxy):
                attrs = ir.DictionaryAttr.get(
                    self.context,
                    {
                        utils.IDENTIFIER_ATTR_NAME: self.builder.get_string_attr(
                            proxy.hierarchy_name
                        )
                    },
                )
                handle = tran_d.MatchOp(
                    self.builder, self.payload_root, any_op_ty, [proxy.kind_str], attrs
                ).get_result_at(0)
            elif isinstance(proxy, ValueProxy):
                owner_attrs = ir.DictionaryAttr.get(
                    self.context,
                    {
                        utils.IDENTIFIER_ATTR_NAME: self.builder.get_string_attr(
                            proxy.owner.hierarchy_name
                        )
                    },
                )
                owner_match = tran_d.MatchOp(
                    self.builder,
                    self.payload_root,
                    any_op_ty,
                    [proxy.owner.kind_str],
                    owner_attrs,
                ).get_result_at(0)
                source_kind = 0
                if ":arg" in proxy.hierarchy_name:
                    source_kind = 1
                elif ":res" in proxy.hierarchy_name:
                    source_kind = 2
                handle = tran_d.MatchValueOp(
                    self.builder, owner_match, proxy.number, source_kind
                ).get_result_at(0)
            else:
                raise TypeError(f"Unsupported proxy type: {type(proxy)}")

            self._valid_handle_cache[self._proxy_key(proxy)] = handle
            out.append(handle)

        return out

    #################
    # Builder Location
    #################
    def _refresh_builder_loc_from_callsite(self):
        frame = inspect.currentframe()
        try:
            if frame is not None:
                frame = frame.f_back
            internal_file = __file__
            while frame is not None and frame.f_code.co_filename == internal_file:
                frame = frame.f_back
            if frame is None:
                self.builder.set_unknown_loc()
                return
            self.builder.set_loc(
                ir.Location(
                    frame.f_code.co_filename,
                    frame.f_lineno,
                    1,
                    self.context,
                )
            )
        finally:
            del frame

    ###############
    # Constructors
    ###############
    @classmethod
    def from_module(cls, module: ir.ModuleOp) -> "Schedule":
        return cls(module, module.get_context())

    @classmethod
    def from_string(cls, s: str) -> "Schedule":
        context = ir.Context()
        context.load_dialects()
        module = utils.parse_from_string(context, s)
        return cls(module, context)

    @classmethod
    def from_file(cls, path: str) -> "Schedule":
        context = ir.Context()
        context.load_dialects()
        module = utils.parse_from_file(context, path)
        return cls(module, context)

    #################
    # Debug Helpers
    #################
    def _format_tree_node(
        self,
        proxy: OperationProxy,
        *,
        prefix: str,
        is_last: bool,
        include_values: bool,
        out: list[str],
    ):
        marker = ""
        child_prefix = prefix
        if prefix or proxy is not self.payload_tree:
            marker = "└─ " if is_last else "├─ "
            child_prefix = prefix + ("   " if is_last else "│  ")

        out.append(
            f"{prefix}{marker}{proxy.identifier} [{_state_name(proxy.state)}] "
            f"alias={proxy.hierarchy_name} kind={proxy.kind_str}"
        )
        if include_values:
            for idx, value in enumerate(proxy.values):
                value_last = idx == len(proxy.values) - 1 and not self.by_parent.get(
                    proxy.hierarchy_name
                )
                value_marker = "└─ " if value_last else "├─ "
                out.append(
                    f"{child_prefix}{value_marker}{value.identifier} "
                    f"[{_state_name(value.state)}] alias={value.hierarchy_name} "
                    f"kind={value.kind_str}"
                )

        children = self.by_parent.get(proxy.hierarchy_name, [])
        for idx, child in enumerate(children):
            self._format_tree_node(
                child,
                prefix=child_prefix,
                is_last=idx == len(children) - 1,
                include_values=include_values,
                out=out,
            )

    def format_tree(self, include_values: bool = True) -> str:
        out: list[str] = []
        self._format_tree_node(
            self.payload_tree,
            prefix="",
            is_last=True,
            include_values=include_values,
            out=out,
        )
        return "\n".join(out)

    def dump_tree(self, include_values: bool = True) -> str:
        tree = self.format_tree(include_values=include_values)
        print(tree)
        return tree

    def _format_map(self) -> str:
        lines = ["=== by_path ==="]
        for path, proxy in self.by_path.items():
            lines.append(f"{path}: {proxy.identifier} [{_state_name(proxy.state)}]")
        lines.append("=== by_identifier ===")
        for identifier, proxies in self.by_identifier.items():
            lines.append(f"{identifier}: {[proxy.hierarchy_name for proxy in proxies]}")
        lines.append("=== by_parent ===")
        for parent, children in self.by_parent.items():
            lines.append(f"{parent}: {[child.hierarchy_name for child in children]}")
        lines.append("=== valid_handle_cache ===")
        for key in self._valid_handle_cache:
            path = "<unknown>"
            for proxy in self.by_path.values():
                if self._proxy_key(proxy) == key:
                    path = proxy.hierarchy_name
                    break
            lines.append(path)
        return "\n".join(lines)

    def dump_map(self) -> str:
        text = self._format_map()
        print(text)
        return text

    def dump_effect_log(self, last_n: int | None = None) -> str:
        entries = self._effect_log
        if last_n is not None:
            if last_n <= 0:
                raise ValueError(f"last_n must be positive, got {last_n}")
            entries = entries[-last_n:]
        if not entries:
            return "<empty effect log>"
        return "\n".join(f"[{idx}] {entry}" for idx, entry in enumerate(entries))

    def dump_transform_script(self) -> str:
        return str(self.transform_mod)

    def dump_state(
        self,
        *,
        include_tree: bool = True,
        include_maps: bool = True,
        include_effects: bool = True,
        include_script: bool = True,
    ) -> str:
        lines = [
            "=== Schedule State ===",
            f"epoch={self.epoch}",
            f"dirty={self.dirty}",
            f"active={self.active_proxy.identifier if self.active_proxy is not None else '<none>'}",
            f"live_proxies={len(self._live_proxy_keys)}",
            f"cached_handles={len(self._valid_handle_cache)}",
            f"effects={len(self._effect_log)}",
        ]
        if include_tree:
            lines.append("--- tree ---")
            lines.append(self.format_tree())
        if include_maps:
            lines.append("--- maps ---")
            lines.append(self._format_map())
        if include_effects:
            lines.append("--- effect_log ---")
            lines.append(self.dump_effect_log())
        if include_script:
            lines.append("--- transform_script ---")
            lines.append(self.dump_transform_script())
        return "\n".join(lines)

    def debug_dump(self, **kwargs):
        text = self.dump_state(**kwargs)
        print(text)
        return self

    #################
    # Front-end APIs
    #################
    def select(self, target: SingleProxy):
        """Select the active proxy for subsequent implicit-target transforms."""
        proxies = self._resolve_proxies(target, "select")
        self._set_active_proxy(proxies[0])
        return self

    def query(
        self,
        *,
        identifier: str | None = None,
        op_kind: str | None = None,
        under: SingleProxy | None = None,
        state: ProxyState | None = VALID,
    ) -> list[OperationProxy]:
        """Query operation proxies with optional identifier, scope, and state filters.

        Parameters:
        - `identifier`: exact proxy identifier to match.
        - `op_kind`: exact operation kind to match.
        - `under`: optional anchor proxy/identifier. Results include the anchor.
        - `state`: lifecycle state filter. Use `None` to disable state filtering.
        """
        if under is not None:
            anchors = self._resolve_proxies(under, "query")
            if len(anchors) != 1:
                raise ValueError(
                    "Query 'under' must resolve to exactly one operation proxy"
                )
            anchor = anchors[0]
            if not isinstance(anchor, OperationProxy):
                raise ValueError("Query 'under' must be an operation proxy")
            candidates = self._get_descendants(anchor, include_root=True)
        else:
            candidates = self._get_descendants(self.payload_tree, include_root=True)

        out: list[OperationProxy] = []
        for proxy in candidates:
            if state is not None and proxy.state != state:
                continue
            if op_kind is not None and proxy.kind_str != op_kind:
                continue
            if identifier is not None and proxy.identifier != identifier:
                continue
            out.append(proxy)
        return out

    def refresh(self):
        """Apply the buffered transform script to payload IR and rebuild proxies.

        Validate:
        - transform module must verify before application.
        - payload module must verify after application.

        Invalidate/Provide:
        - marks all currently live proxies `STALE`
        - clears materialized handle caches
        - rebuilds proxy tree and indexes from refreshed payload IR
        """
        if not self.dirty:
            return self

        self.canonicalize(self.payload_tree)
        if not self.transform_mod.verify():
            raise RuntimeError("Transform module verification failed")

        failed, err_msg = tran_d.apply_transforms(
            self.payload.get_operation(),
            self.transform_seq.get_operation(),
            self.transform_mod,
        )
        if failed:
            raise RuntimeError(f"Failed to apply transforms: {err_msg}")
        if not self.payload.verify():
            print(self.payload)
            raise RuntimeError(
                "Payload module verification failed after applying transforms"
            )

        refresh_id = self._next_transform_id("refresh")
        self._mark_live_proxies_stale(
            effect_id=refresh_id, reason="superseded by refresh boundary"
        )
        self.epoch += 1
        utils.finalize_transform(self.payload_tree, self._id_remap)
        self._rebuild_tree_from_payload()
        self._init_transform()
        self._record_effect(refresh_id, "refresh", overwrite_identifiers=True)
        return self

    #########################
    # Generic Transformations
    #########################
    def cse(self, targets: Proxies | None = None):
        """Apply CSE to one or more target proxies.

        Validate:
        - every target proxy is `VALID`

        Invalidate/Provide:
        - no proxy is consumed
        - no new proxy is provided
        """
        resolved = self._resolve_proxies(targets, "cse")
        handles = self._materialize_proxies(resolved)
        transform_id = self._next_transform_id("cse")
        self._refresh_builder_loc_from_callsite()
        for handle in handles:
            tran_d.ApplyCSEOp(self.builder, handle)
        self._mark_dirty()
        if resolved:
            self._set_active_proxy(resolved[-1])
        self._record_effect(
            transform_id,
            "cse",
            targets=[proxy.hierarchy_name for proxy in resolved],
        )
        return self

    def dce(self, targets: Proxies | None = None):
        """Apply DCE to one or more target proxies.

        Validate:
        - every target proxy is `VALID`

        Invalidate/Provide:
        - no proxy is consumed
        - no new proxy is provided
        """
        resolved = self._resolve_proxies(targets, "dce")
        handles = self._materialize_proxies(resolved)
        transform_id = self._next_transform_id("dce")
        self._refresh_builder_loc_from_callsite()
        for handle in handles:
            tran_d.ApplyDCEOp(self.builder, handle)
        self._mark_dirty()
        if resolved:
            self._set_active_proxy(resolved[-1])
        self._record_effect(
            transform_id,
            "dce",
            targets=[proxy.hierarchy_name for proxy in resolved],
        )
        return self

    def licm(self, targets: Proxies | None = None):
        resolved = self._resolve_proxies(targets, "licm", require_loop=True)
        handles = self._materialize_proxies(resolved)
        transform_id = self._next_transform_id("licm")
        self._refresh_builder_loc_from_callsite()
        for handle in handles:
            tran_d.ApplyLICMOp(self.builder, handle)
        self._mark_dirty()
        if resolved:
            self._set_active_proxy(resolved[-1])
        self._record_effect(
            transform_id,
            "licm",
            targets=[proxy.hierarchy_name for proxy in resolved],
        )
        return self

    _supported_patterns = {"canonicalize": tran_d.ApplyCanonicalizationOp}

    def apply_patterns(
        self,
        patterns: str | list[str],
        targets: Proxies | None = None,
    ):
        """Apply one or more transform pattern ops to target proxies.

        Validate:
        - every target proxy is `VALID`
        - every requested pattern is supported

        Invalidate/Provide:
        - no proxy is consumed
        - no new proxy is provided
        """
        pattern_names = [patterns] if isinstance(patterns, str) else list(patterns)
        if len(pattern_names) == 0:
            raise ValueError("apply_patterns requires at least one pattern")

        pattern_ops = []
        for pattern in pattern_names:
            op_cls = self._supported_patterns.get(pattern)
            if op_cls is None:
                raise ValueError(f"Unsupported pattern '{pattern}' in apply_patterns")
            pattern_ops.append(op_cls)

        resolved = self._resolve_proxies(targets, "apply_patterns")
        handles = self._materialize_proxies(resolved)
        transform_id = self._next_transform_id("apply_patterns")
        self._refresh_builder_loc_from_callsite()
        for handle in handles:
            entry = tran_d.ApplyPatternsOp(self.builder, handle)
            ip = self.builder.save_insertion_point()
            self.builder.set_insertion_point_to_end(entry.get_body())
            for op_cls in pattern_ops:
                op_cls(self.builder)
            self.builder.restore_insertion_point(ip)
        self._mark_dirty()
        if resolved:
            self._set_active_proxy(resolved[-1])
        self._record_effect(
            transform_id,
            "apply_patterns",
            patterns=pattern_names,
            targets=[proxy.hierarchy_name for proxy in resolved],
        )
        return self

    def canonicalize(self, targets: Proxies | None = None):
        """Apply canonicalization patterns to target proxies."""
        return self.apply_patterns("canonicalize", targets)

    #######################
    # Loop transformations
    #######################
    def polyhedral(self, targets: Proxies | None = None):
        """Raise loop-like targets to affine form in place.

        Validate:
        - every target proxy is `VALID` and loop-like

        Invalidate/Provide:
        - does not consume proxies
        - rebinds selected loop handles to raised affine results
        """
        resolved = self._resolve_proxies(targets, "polyhedral", require_loop=True)
        handles = self._materialize_proxies(resolved)
        outputs: list[ir.Value] = []
        kind_updates: list[str] = []
        transform_id = self._next_transform_id("polyhedral")

        self._refresh_builder_loc_from_callsite()
        for proxy, handle in zip(resolved, handles):
            outputs.append(
                tran_d.RaiseToAffineOp(self.builder, handle).get_result_at(0)
            )
            if proxy.kind_str.startswith("scf."):
                kind_updates.append(proxy.kind_str.replace("scf.", "affine.", 1))
            else:
                kind_updates.append(proxy.kind_str)

        self._inplace_update_proxies(
            resolved,
            outputs,
            kind_strs=kind_updates,
        )
        self._mark_dirty()
        if resolved:
            self._set_active_proxy(resolved[-1])
        self._record_effect(
            transform_id,
            "polyhedral",
            targets=[proxy.hierarchy_name for proxy in resolved],
        )
        return self

    def unroll(
        self,
        targets: Proxies | None = None,
        factor: int = 0,
        tag_only: bool = True,
    ):
        """Unroll loop targets or attach unroll metadata.

        Parameters:
        - `targets`: loop proxies or identifiers.
        - `factor`: non-negative unroll factor.
        - `tag_only`: whether to tag only or consume the loop handle.

        Validate:
        - every target proxy is `VALID` and loop-like
        - `factor >= 0`

        Invalidate/Provide:
        - `tag_only=True`: no proxy is consumed
        - `tag_only=False`: target proxy is consumed and descendants are stale
        """
        if factor < 0:
            raise ValueError(f"Unroll factor must be non-negative, got {factor}")

        resolved = self._resolve_proxies(targets, "unroll", require_loop=True)
        handles = self._materialize_proxies(resolved)
        transform_id = self._next_transform_id("unroll")

        self._refresh_builder_loc_from_callsite()
        for proxy, handle in zip(resolved, handles):
            if tag_only:
                tran_d.TagUnrollOp(self.builder, handle, factor)
            else:
                tran_d.LoopUnrollOp(self.builder, handle, factor)
                self._invalidate_proxies(
                    [proxy],
                    effect_id=transform_id,
                    reason="consumed by unroll",
                )
                self._descendant_invalidate(proxy, include_root=False)

        self._mark_dirty()
        if tag_only and resolved:
            self._set_active_proxy(resolved[-1])
        else:
            self._select_fallback_active()
        self._record_effect(
            transform_id,
            "unroll",
            targets=[proxy.hierarchy_name for proxy in resolved],
            factor=factor,
            tag_only=tag_only,
        )
        return self

    def flatten(self, targets: Proxies | None = None):
        """Flatten selected loop proxies into one loop.

        Validate:
        - at least one target
        - every target proxy is `VALID` and loop-like

        Invalidate/Provide:
        - keeps the outermost proxy as `<target>::flat`
        - removes intermediate selected proxies
        - descendants are reattached under the flattened loop
        """
        resolved = self._resolve_proxies(targets, "flatten", require_loop=True)
        if len(resolved) == 0:
            raise ValueError("flatten requires at least one loop target")
        sorted_resolved = sorted(resolved, key=self._proxy_depth)
        handles = self._materialize_proxies(resolved)
        transform_id = self._next_transform_id("flatten")

        self._refresh_builder_loc_from_callsite()
        merged = tran_d.MergeHandlesOp(
            self.builder, handles, deduplicate=True
        ).get_result_at(0)
        flattened = tran_d.LoopFlattenOp(self.builder, merged).get_result_at(0)

        outermost = sorted_resolved[0]
        innermost = sorted_resolved[-1]
        outer_id = outermost.identifier
        outer_path = outermost.hierarchy_name

        self._descendant_invalidate(outermost, include_root=False)
        self._inplace_update_proxies(
            [outermost],
            [flattened],
            identifiers=[outer_id + "::flat"],
            hier_names=[outer_path + "::flat"],
        )

        self._remap_id(f"{outer_path}::flat", f"{outer_id}::flat")

        if innermost is not outermost:
            self._splice_proxies(innermost, outermost, 0)

        to_remove = list(reversed(sorted_resolved[1:]))
        if to_remove:
            self._remove_proxies(
                to_remove,
                effect_id=transform_id,
                reason="removed by flatten",
            )

        self._mark_dirty()
        self._set_active_proxy(outermost)
        self._record_effect(
            transform_id,
            "flatten",
            targets=[proxy.hierarchy_name for proxy in sorted_resolved],
        )
        return self

    def tile(self, targets: Proxies | None = None, factors: int | list[int] = 1):
        """Tile one or more loops and provide tile/point proxies.

        Parameters:
        - `targets`: ordered loop proxies or identifiers.
        - `factors`: one positive factor per target loop.

        Validate:
        - every target proxy is `VALID` and loop-like
        - `len(factors) == len(targets)`
        - every factor is positive

        Invalidate/Provide:
        - each selected loop becomes `<target>::tile`
        - each selected loop also provides `<target>::point`
        - descendants move under the innermost `::point`
        """
        if isinstance(factors, int):
            factors = [factors]
        factors = list(factors)
        if len(factors) == 0:
            raise ValueError("tile requires at least one factor")
        for factor in factors:
            if factor <= 0:
                raise ValueError(f"tile factors must be positive, got {factor}")

        resolved = self._resolve_proxies(targets, "tile", require_loop=True)
        if len(resolved) != len(factors):
            raise ValueError(
                "tile requires the number of factors to match the number of loop targets"
            )

        handles = self._materialize_proxies(resolved)
        sorted_resolved = sorted(resolved, key=self._proxy_depth)
        transform_id = self._next_transform_id("tile")

        base_ids = {
            self._proxy_key(proxy): proxy.identifier for proxy in sorted_resolved
        }
        base_paths = {
            self._proxy_key(proxy): proxy.hierarchy_name for proxy in sorted_resolved
        }

        self._refresh_builder_loc_from_callsite()
        merged = tran_d.MergeHandlesOp(
            self.builder, handles, deduplicate=True
        ).get_result_at(0)
        tiled = tran_d.LoopTileOp(self.builder, merged, factors)
        tile_group = tiled.get_result_at(0)
        point_group = tiled.get_result_at(1)
        tile_split = tran_d.SplitHandleOp(
            self.builder, tile_group, len(sorted_resolved)
        )
        point_split = tran_d.SplitHandleOp(
            self.builder, point_group, len(sorted_resolved)
        )

        for proxy in sorted_resolved:
            self._descendant_invalidate(proxy, include_root=False)

        tile_handles = [
            tile_split.get_result_at(i) for i in range(len(sorted_resolved))
        ]
        self._inplace_update_proxies(
            sorted_resolved,
            tile_handles,
            identifiers=[
                base_ids[self._proxy_key(proxy)] + "::tile" for proxy in sorted_resolved
            ],
            hier_names=[
                base_paths[self._proxy_key(proxy)] + "::tile"
                for proxy in sorted_resolved
            ],
        )
        # remap ids
        for proxy in sorted_resolved:
            self._remap_id(
                f"{base_paths[self._proxy_key(proxy)]}::tile",
                f"{base_ids[self._proxy_key(proxy)]}::tile",
            )
            self._remap_id(
                f"{base_paths[self._proxy_key(proxy)]}::point",
                f"{base_ids[self._proxy_key(proxy)]}::point",
            )

        deepest_tile = sorted_resolved[-1]
        current_parent = deepest_tile
        point_proxies: list[OperationProxy] = []
        for idx, source in enumerate(sorted_resolved):
            point_proxy = self._create_proxy(
                base_ids[self._proxy_key(source)] + "::point",
                base_paths[self._proxy_key(source)] + "::point",
                source.kind_str,
            )
            self._provide_proxies([point_proxy], [point_split.get_result_at(idx)])
            self._insert_proxies([point_proxy], current_parent, 0)
            point_proxies.append(point_proxy)
            current_parent = point_proxy

        if point_proxies:
            self._splice_proxies(deepest_tile, point_proxies[-1], 1)

        self._mark_dirty()
        if point_proxies:
            self._set_active_proxy(point_proxies[-1])
        else:
            self._set_active_proxy(deepest_tile)
        self._record_effect(
            transform_id,
            "tile",
            targets=[proxy.hierarchy_name for proxy in sorted_resolved],
            factors=factors,
        )
        return self

    def pipeline(self, targets: Proxies | None = None, ii: int = 1):
        """Attach pipeline annotation to loop targets.

        Parameters:
        - `targets`: loop proxies or identifiers. If omitted, use the active proxy.
        - `ii`: positive initiation interval.

        Validate:
        - every target proxy is `VALID` and loop-like
        - `ii > 0`

        Invalidate/Provide:
        - no proxy is consumed
        - no new proxy is provided
        """
        if ii <= 0:
            raise ValueError(f"Pipeline II must be positive, got {ii}")

        resolved = self._resolve_proxies(targets, "pipeline", require_loop=True)
        handles = self._materialize_proxies(resolved)
        transform_id = self._next_transform_id("pipeline")

        self._refresh_builder_loc_from_callsite()
        for handle in handles:
            tran_d.TagPipelineOp(self.builder, handle, ii)

        self._mark_dirty()
        if resolved:
            self._set_active_proxy(resolved[-1])
        self._record_effect(
            transform_id,
            "pipeline",
            targets=[proxy.hierarchy_name for proxy in resolved],
            ii=ii,
        )
        return self

    def split(self, targets: Proxies | None = None, factor: int = 1):
        """Split loop targets into outer/inner pairs.

        Parameters:
        - `targets`: loop proxies or identifiers.
        - `factor`: positive split factor.

        Validate:
        - every target proxy is `VALID` and loop-like
        - `factor > 0`

        Invalidate/Provide:
        - selected loop becomes `<target>::outer`
        - provides a detached `<target>::inner` child
        - descendants move under the new inner loop
        """
        if factor <= 0:
            raise ValueError(f"Split factor must be positive, got {factor}")

        resolved = self._resolve_proxies(targets, "split", require_loop=True)
        resolved = sorted(resolved, key=self._proxy_depth, reverse=True)
        handles = self._materialize_proxies(resolved)
        transform_id = self._next_transform_id("split")

        created_inner: list[OperationProxy] = []
        self._refresh_builder_loc_from_callsite()
        for proxy, handle in zip(resolved, handles):
            split_op = tran_d.LoopSplitOp(self.builder, handle, factor)
            outer_handle = split_op.get_result_at(0)
            inner_handle = split_op.get_result_at(1)

            self._descendant_invalidate(proxy, include_root=False)

            base_identifier = proxy.identifier
            base_path = proxy.hierarchy_name

            inner_proxy = self._create_proxy(
                base_identifier + "::inner",
                base_path + "::inner",
                proxy.kind_str,
            )
            self._provide_proxies([inner_proxy], [inner_handle])
            self._insert_proxies([inner_proxy], proxy, 0)
            self._splice_proxies(proxy, inner_proxy, 1)

            self._inplace_update_proxies(
                [proxy],
                [outer_handle],
                identifiers=[base_identifier + "::outer"],
                hier_names=[base_path + "::outer"],
            )
            self._remap_id(f"{base_path}::inner", f"{base_identifier}::inner")
            self._remap_id(f"{base_path}::outer", f"{base_identifier}::outer")
            created_inner.append(inner_proxy)

        self._mark_dirty()
        if created_inner:
            self._set_active_proxy(created_inner[-1])
        self._record_effect(
            transform_id,
            "split",
            targets=[proxy.hierarchy_name for proxy in resolved],
            factor=factor,
        )
        return self

    def outline(self, target: SingleProxy | None = None, *, func_name: str):
        """Outline one target op into a kernel function and call.

        Parameters:
        - `target`: operation proxy or identifier. If omitted, use the active proxy.
        - `func_name`: non-empty kernel function symbol.

        Validate:
        - target resolves to exactly one `VALID` operation proxy
        - target is not the payload root module
        - `func_name` is non-empty

        Invalidate/Provide:
        - consumes the outlined source proxy and descendants
        - provides a new kernel `func.func` proxy
        - provides a replacement `func.call` proxy at the original position
        """
        if len(func_name) == 0:
            raise ValueError("outline requires a non-empty func_name")

        source = self._resolve_proxies(target, "outline")
        if len(source) != 1:
            raise ValueError("outline requires a single target proxy")
        source = source[0]
        if source is self.payload_tree:
            raise ValueError("outline cannot target the payload root module")

        handle = self._materialize_proxies([source])[0]
        transform_id = self._next_transform_id("outline")

        self._refresh_builder_loc_from_callsite()
        outlined = tran_d.OutlineOp(self.builder, handle, func_name)

        kernel_proxy = self._create_proxy(
            func_name,
            source.hierarchy_name,
            "func.func",
        )
        call_proxy = self._create_proxy(
            func_name + "::call",
            source.hierarchy_name + "::call",
            "func.call",
        )
        self._remap_id(f"{source.hierarchy_name}::call", f"{func_name}::call")

        assert isinstance(source, OperationProxy)
        call_parent_path = self._parent_path.get(self._proxy_key(source))
        assert (
            call_parent_path is not None
        ), "Internal invariant violated: outline source has no live parent path"
        call_parent = self.by_path.get(call_parent_path)
        assert isinstance(
            call_parent, OperationProxy
        ), "Internal invariant violated: outline source parent is not an operation proxy"
        source_siblings = self.by_parent.get(call_parent_path, [])
        source_pos: int | None = None
        for idx, sibling in enumerate(source_siblings):
            if sibling is source:
                source_pos = idx
                break
        assert (
            source_pos is not None
        ), "Internal invariant violated: outline source missing from parent bucket"
        self._descendant_remove(
            source,
            effect_id=transform_id,
            reason="removed by outline",
        )
        self._provide_proxies(
            [kernel_proxy, call_proxy],
            [outlined.get_result_at(0), outlined.get_result_at(1)],
        )
        self._insert_proxies(
            [kernel_proxy],
            self.payload_tree,
            len(self.by_parent.get(self.payload_tree.hierarchy_name, [])),
        )
        self._insert_proxies([call_proxy], call_parent, source_pos)

        self._mark_dirty()
        self._set_active_proxy(call_proxy)
        self._record_effect(
            transform_id,
            "outline",
            target=source.hierarchy_name,
            func_name=func_name,
            kernel=kernel_proxy.hierarchy_name,
            call=call_proxy.hierarchy_name,
        )
        return self

    def reorder(self, targets: Proxies, order: list[int]):
        """Reorder a perfect affine loop band according to a permutation.

        Parameters:
        - `targets`: ordered affine loop proxies participating in reorder.
        - `order`: permutation over `targets`.

        Validate:
        - at least two loop targets
        - every target proxy is `VALID` and resolves to `affine.for`
        - targets form a strict parent chain in one perfect affine band
        - `order` is a permutation of `[0, ..., n - 1]`

        Invalidate/Provide:
        - no proxy is consumed
        - selected loop handles are rebound in place
        - Python-side parent/child indexes are updated pre-refresh
        """
        resolved = self._resolve_proxies(targets, "reorder", require_loop=True)
        if len(resolved) < 2:
            raise ValueError("reorder requires at least two loop targets")
        if len(order) != len(resolved):
            raise ValueError(
                f"reorder expects order length {len(resolved)}, got {len(order)}"
            )
        if sorted(order) != list(range(len(resolved))):
            raise ValueError(f"reorder expects a permutation, got {order}")

        affine_loops: list[OperationProxy] = []
        for proxy in resolved:
            assert isinstance(
                proxy, OperationProxy
            ), "Internal invariant violated: loop resolve produced a value proxy"
            if proxy.kind_str == "scf.for":
                raise ValueError(
                    f"reorder only supports affine.for, got '{proxy.hierarchy_name}' "
                    "as scf.for; run polyhedral first"
                )
            if proxy.kind_str != "affine.for":
                raise ValueError(
                    f"reorder only supports affine.for loops, got '{proxy.kind_str}' "
                    f"for '{proxy.hierarchy_name}'"
                )
            affine_loops.append(proxy)

        if not self._is_strict_parent_chain(affine_loops):
            raise ValueError("reorder requires targets to form a strict parent chain")
        if not self._is_perfect_loop_band(affine_loops):
            raise ValueError(
                "reorder requires targets to be in the same perfect affine loop band"
            )

        handles = self._materialize_proxies(affine_loops)
        transform_id = self._next_transform_id("reorder")

        old_outer = affine_loops[0]
        old_outer_parent_path = self._parent_path.get(self._proxy_key(old_outer))
        external_parent: OperationProxy | None = None
        old_outer_pos: int | None = None
        if old_outer_parent_path is not None:
            external_parent_proxy = self.by_path.get(old_outer_parent_path)
            assert isinstance(external_parent_proxy, OperationProxy), (
                "Internal invariant violated: operation parent path resolves to "
                "non-operation proxy"
            )
            external_parent = external_parent_proxy
            siblings = self.by_parent.get(old_outer_parent_path, [])
            for idx, sibling in enumerate(siblings):
                if sibling is old_outer:
                    old_outer_pos = idx
                    break
            assert (
                old_outer_pos is not None
            ), "Internal invariant violated: outer loop missing from parent bucket"

        old_innermost = affine_loops[-1]
        selected_keys = {self._proxy_key(proxy) for proxy in affine_loops}
        non_selected_children = [
            child
            for child in list(self.by_parent.get(old_innermost.hierarchy_name, []))
            if self._proxy_key(child) not in selected_keys
        ]

        self._refresh_builder_loc_from_callsite()
        merged = tran_d.MergeHandlesOp(
            self.builder, handles, deduplicate=False
        ).get_result_at(0)
        tran_d.LoopReorderOp(self.builder, merged, order)
        split_back = tran_d.SplitHandleOp(self.builder, merged, len(affine_loops))

        self._inplace_update_proxies(
            affine_loops,
            [split_back.get_result_at(idx) for idx in range(len(affine_loops))],
        )

        new_chain = [affine_loops[idx] for idx in order]
        self._reparent_proxy(new_chain[0], external_parent, pos=old_outer_pos)
        for idx in range(1, len(new_chain)):
            self._reparent_proxy(new_chain[idx], new_chain[idx - 1], pos=0)
        for child in non_selected_children:
            self._reparent_proxy(child, new_chain[-1])

        self._mark_dirty()
        self._set_active_proxy(affine_loops[-1])
        self._record_effect(
            transform_id,
            "reorder",
            targets=[proxy.hierarchy_name for proxy in affine_loops],
            order=list(order),
            hierarchy_updated=True,
        )
        return self

    def compute_at(self, target: SingleProxy, axis: SingleProxy):
        """Move a producer subtree to execute at an affine axis.

        Validate:
        - `target` resolves to exactly one `VALID` operation proxy
        - `axis` resolves to exactly one `VALID` `affine.for`
        - target and axis are different proxies

        Invalidate/Provide:
        - consumes the target proxy and its owned values
        - marks target descendants stale
        - does not rewrite Python-side topology before refresh
        """
        resolved_target = self._resolve_proxies(target, "compute_at target")
        if len(resolved_target) != 1:
            raise ValueError("compute_at requires a single target proxy")
        target_proxy = resolved_target[0]
        assert isinstance(
            target_proxy, OperationProxy
        ), "Internal invariant violated: compute_at target is not an operation proxy"

        resolved_axis = self._resolve_proxies(
            axis,
            "compute_at axis",
            require_loop=True,
        )
        if len(resolved_axis) != 1:
            raise ValueError("compute_at requires a single axis loop")
        axis_proxy = resolved_axis[0]
        assert isinstance(
            axis_proxy, OperationProxy
        ), "Internal invariant violated: compute_at axis is not an operation proxy"

        if target_proxy is axis_proxy:
            raise ValueError("compute_at target and axis must differ")
        if axis_proxy.kind_str == "scf.for":
            raise ValueError(
                f"compute_at only supports affine.for axis, got '{axis_proxy.hierarchy_name}' "
                "as scf.for; run polyhedral first"
            )
        if axis_proxy.kind_str != "affine.for":
            raise ValueError(
                f"compute_at only supports affine.for axis, got '{axis_proxy.kind_str}' "
                f"for '{axis_proxy.hierarchy_name}'"
            )

        target_handle = self._materialize_proxies([target_proxy])[0]
        axis_handle = self._materialize_proxies([axis_proxy])[0]
        transform_id = self._next_transform_id("compute_at")

        self._refresh_builder_loc_from_callsite()
        tran_d.ComputeAtOp(self.builder, target_handle, axis_handle)

        descendants = self._get_descendants(target_proxy, include_root=False)
        target_values = [
            value
            for value in target_proxy.values
            if self._proxy_key(value) in self._live_proxy_keys
        ]
        descendant_values = [
            value
            for proxy in descendants
            for value in proxy.values
            if self._proxy_key(value) in self._live_proxy_keys
        ]

        self._invalidate_proxies(
            [target_proxy, *target_values],
            effect_id=transform_id,
            reason="consumed by compute_at",
            state=INVALID,
        )
        if descendants or descendant_values:
            self._invalidate_proxies(
                [*descendants, *descendant_values],
                effect_id=transform_id,
                reason="stale after compute_at",
                state=STALE,
            )

        self._mark_dirty()
        self._set_active_proxy(axis_proxy)
        self._record_effect(
            transform_id,
            "compute_at",
            target=target_proxy.hierarchy_name,
            axis=axis_proxy.hierarchy_name,
            requires_refresh_topology=True,
        )
        return self

    def reuse_at(
        self, target: str | ValueProxy, axis: SingleProxy, ring: bool = False
    ) -> ValueProxy:
        """Create a reuse buffer for one buffer value at an affine axis.

        Parameters:
        - `target`: buffer value proxy or identifier.
        - `axis`: affine loop proxy or identifier.
        - `ring`: whether to create a circular buffer for reuse (experimental)

        Validate:
        - `target` resolves to exactly one `VALID` buffer value proxy
        - `axis` resolves to exactly one `VALID` `affine.for`

        Invalidate/Provide:
        - keeps source buffer and axis proxies live
        - provides a new `memref.alloc` op proxy and its result value proxy
        - inserts the alloc before the outermost affine root loop
        """
        resolved_values = self._resolve_proxies(
            target, "reuse_at target", require_value=True, require_op=False
        )
        if len(resolved_values) != 1:
            raise ValueError("reuse_at expected a single target proxy")
        source_value_proxy = resolved_values[0]
        assert isinstance(source_value_proxy, ValueProxy)
        if source_value_proxy.kind_str != "buffer":
            raise ValueError(
                f"Value '{source_value_proxy.hierarchy_name}' is not a buffer and cannot be reused"
            )

        resolved_axis = self._resolve_proxies(
            axis,
            "reuse_at axis",
            require_loop=True,
        )
        if len(resolved_axis) != 1:
            raise ValueError("reuse_at requires a single axis loop")
        axis_proxy = resolved_axis[0]
        assert isinstance(axis_proxy, OperationProxy)

        if axis_proxy.kind_str == "scf.for":
            raise ValueError(
                f"reuse_at only supports affine.for axis, got '{axis_proxy.hierarchy_name}' "
                "as scf.for; run polyhedral first"
            )
        if axis_proxy.kind_str != "affine.for":
            raise ValueError(
                f"reuse_at only supports affine.for axis, got '{axis_proxy.kind_str}' "
                f"for '{axis_proxy.hierarchy_name}'"
            )

        handles = self._materialize_proxies([source_value_proxy, axis_proxy])
        target_handle = handles[0]
        axis_handle = handles[1]

        transform_id = self._next_transform_id("reuse_at")

        root_loop = self._outermost_affine_loop(axis_proxy)
        root_parent_path = self._parent_path.get(self._proxy_key(root_loop))
        if root_parent_path is None:
            raise ValueError(
                f"reuse_at cannot place a buffer for root loop '{root_loop.hierarchy_name}' "
                "without a parent scope"
            )
        root_parent = self.by_path.get(root_parent_path)
        assert isinstance(root_parent, OperationProxy)
        siblings = self.by_parent.get(root_parent_path, [])
        root_pos: int | None = None
        for idx, sibling in enumerate(siblings):
            if sibling is root_loop:
                root_pos = idx
                break
        assert root_pos is not None

        self._refresh_builder_loc_from_callsite()
        value_handle = tran_d.ReuseAtOp(
            self.builder, target_handle, axis_handle, ring
        ).get_result_at(0)

        assert source_value_proxy.owner is not None

        value_identifier = f"{source_value_proxy.identifier}::reuse"
        alloc_identifier = value_identifier
        alloc_hierarchy = f"{source_value_proxy.owner.hierarchy_name}::reuse"
        value_hierarchy = alloc_hierarchy + ":res0"

        alloc_proxy = self._create_proxy(
            alloc_identifier,
            alloc_hierarchy,
            "memref.alloc",
        )
        value_proxy = self._create_value_proxy(
            alloc_proxy,
            value_identifier,
            value_hierarchy,
            kind_str="buffer",
            number=0,
        )

        self._provide_proxies(
            [alloc_proxy, value_proxy],
            [None, value_handle],
        )
        self._insert_proxies([alloc_proxy], root_parent, root_pos)

        self._remap_id(alloc_hierarchy, value_identifier)

        self._mark_dirty()
        self._set_active_proxy(alloc_proxy)
        self._record_effect(
            transform_id,
            "reuse_at",
            target=source_value_proxy.hierarchy_name,
            axis=axis_proxy.hierarchy_name,
            alloc=alloc_proxy.hierarchy_name,
            result=value_proxy.hierarchy_name,
            placement_parent=root_parent.hierarchy_name,
            placement_before=root_loop.hierarchy_name,
        )
        return value_proxy

    def buffer_at(self, target: str | ValueProxy, axis: SingleProxy) -> ValueProxy:
        """Create a local buffer for one buffer value inside an affine axis.

        Parameters:
        - `target`: buffer value proxy or identifier.
        - `axis`: affine loop proxy or identifier.

        Validate:
        - `target` resolves to exactly one `VALID` buffer value proxy
        - `axis` resolves to exactly one `VALID` `affine.for`

        Invalidate/Provide:
        - keeps source buffer and axis proxies live
        - provides a new `memref.alloc` op proxy and its result value proxy
        - inserts the alloc at the start of the selected axis body
        """
        resolved_values = self._resolve_proxies(
            target, "buffer_at target", require_value=True, require_op=False
        )
        if len(resolved_values) != 1:
            raise ValueError("buffer_at expected a single target proxy")
        source_value_proxy = resolved_values[0]
        assert isinstance(source_value_proxy, ValueProxy)
        if source_value_proxy.kind_str != "buffer":
            raise ValueError(
                f"Value '{source_value_proxy.hierarchy_name}' is not a buffer and cannot be buffered"
            )

        resolved_axis = self._resolve_proxies(
            axis,
            "buffer_at axis",
            require_loop=True,
        )
        if len(resolved_axis) != 1:
            raise ValueError("buffer_at requires a single axis loop")
        axis_proxy = resolved_axis[0]
        assert isinstance(axis_proxy, OperationProxy)

        if axis_proxy.kind_str == "scf.for":
            raise ValueError(
                f"buffer_at only supports affine.for axis, got '{axis_proxy.hierarchy_name}' "
                "as scf.for; run polyhedral first"
            )
        if axis_proxy.kind_str != "affine.for":
            raise ValueError(
                f"buffer_at only supports affine.for axis, got '{axis_proxy.kind_str}' "
                f"for '{axis_proxy.hierarchy_name}'"
            )

        handles = self._materialize_proxies([source_value_proxy, axis_proxy])
        target_handle = handles[0]
        axis_handle = handles[1]

        transform_id = self._next_transform_id("buffer_at")

        self._refresh_builder_loc_from_callsite()
        value_handle = tran_d.BufferAtOp(
            self.builder, target_handle, axis_handle
        ).get_result_at(0)

        assert source_value_proxy.owner is not None

        value_identifier = f"{source_value_proxy.identifier}::local"
        alloc_identifier = value_identifier
        alloc_hierarchy = f"{source_value_proxy.owner.hierarchy_name}::local"
        value_hierarchy = alloc_hierarchy + ":res0"

        alloc_proxy = self._create_proxy(
            alloc_identifier,
            alloc_hierarchy,
            "memref.alloc",
        )
        value_proxy = self._create_value_proxy(
            alloc_proxy,
            value_identifier,
            value_hierarchy,
            kind_str="buffer",
            number=0,
        )

        self._provide_proxies(
            [alloc_proxy, value_proxy],
            [None, value_handle],
        )
        self._insert_proxies([alloc_proxy], axis_proxy, 0)

        self._remap_id(alloc_hierarchy, value_identifier)

        self._mark_dirty()
        self._set_active_proxy(alloc_proxy)
        self._record_effect(
            transform_id,
            "buffer_at",
            target=source_value_proxy.hierarchy_name,
            axis=axis_proxy.hierarchy_name,
            alloc=alloc_proxy.hierarchy_name,
            result=value_proxy.hierarchy_name,
            placement_parent=axis_proxy.hierarchy_name,
            placement_before=(
                self.by_parent[axis_proxy.hierarchy_name][1].hierarchy_name
                if len(self.by_parent[axis_proxy.hierarchy_name]) > 1
                else None
            ),
        )
        return value_proxy

    def partition(
        self,
        targets: Proxies | None,
        *,
        dim: int = 0,
        kind: tran_d.PartitionKind = tran_d.Complete,
        factor: int = 0,
    ):
        """Attach partition attributes to concrete buffer values.

        Parameters:
        - `targets`: one or more value proxies or identifiers.
        - `dim`: partition dimension, must be non-negative.
        - `kind`: partition kind.
        - `factor`: partition factor. Must be `0` for complete partition and
          positive otherwise.

        Validate:
        - every target resolves to a `VALID` buffer value proxy
        - `dim >= 0`
        - factor constraints match `kind`

        Invalidate/Provide:
        - no proxy is consumed
        - no new proxy is provided
        - payload IR is updated in place with `allo.part`
        """
        resolved = self._resolve_proxies(
            targets,
            "partition",
            require_value=True,
            require_op=False,
        )
        for p in resolved:
            if p.kind_str != "buffer":
                raise ValueError(
                    f"Value '{p.hierarchy_name}' is not a buffer and cannot be partitioned"
                )
        if dim < 0:
            raise ValueError(f"partition requires non-negative dim, got {dim}")
        if kind == tran_d.Complete:
            if factor != 0:
                raise ValueError("Complete partition cannot have non-zero factor")
        elif factor <= 0:
            raise ValueError(
                f"{kind.name} partition must have positive factor, got {factor}"
            )

        handles = self._materialize_proxies(resolved)
        transform_id = self._next_transform_id("partition")
        part = tran_d.PartitionAttr.get(self.context, [(dim, kind, factor)])

        self._refresh_builder_loc_from_callsite()
        for h in handles:
            tran_d.PartitionOp(self.builder, h, part)

        self._mark_dirty()
        self._record_effect(
            transform_id,
            "partition",
            targets=[proxy.hierarchy_name for proxy in resolved],
            dim=dim,
            kind=kind.name,
            factor=factor,
        )
        return self


__all__ = ["Schedule"]
