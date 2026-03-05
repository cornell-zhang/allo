# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect as pyinspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from .bindings import ir, transform as tran_d, allo as allo_d


class HandleState(str, Enum):
    VALID = "valid"
    CONSUMED = "consumed"
    STALE = "stale"


@dataclass
class HandleProxy:
    instance_id: str
    identifier: str
    path: str
    parent_instance_id: str | None
    op_kind: str
    state: HandleState = HandleState.VALID
    meta: dict[str, Any] = field(default_factory=dict)

    def is_looplike(self) -> bool:
        return self.op_kind in {"affine.for", "scf.for"}


@dataclass
class ValueProxy:
    value_identifier: str
    owner_identifier: str
    owner_instance_id: str | None
    owner_op_kind: str
    source_kind: str
    source_index: int
    type_str: str
    is_memref: bool
    root_kind: str
    root_owner_identifier: str
    root_arg_number: int


@dataclass(frozen=True)
class _HandleConsume:
    """Internal consume descriptor used by handle-lifecycle bookkeeping."""

    proxy: HandleProxy


@dataclass(frozen=True)
class _HandleProvide:
    """Internal provide descriptor used by handle-lifecycle bookkeeping."""

    source: HandleProxy
    identifier: str
    value: ir.Value
    tag: str
    path: str | None = None
    meta_updates: dict[str, Any] | None = None
    select: bool = False


class Schedule:
    """Experimental schedule: identifier-first proxy + transform linkage.

    The scheduler maintains a front-end proxy graph that mirrors payload IR
    hierarchy and tracks per-handle lifecycle:
    - `VALID`: can be used by later transforms.
    - `CONSUMED`: explicitly consumed by a transform.
    - `STALE`: descendants invalidated by a consumed ancestor.
    """

    def __init__(
        self,
        module: ir.ModuleOp,
        context: ir.Context | None = None,
    ):
        self.module = module
        self.context = module.context if context is None else context
        self.context.load_transform_dialects()

        self.by_instance: dict[str, HandleProxy] = {}
        self.by_identifier: dict[str, set[str]] = {}
        self.by_path: dict[str, set[str]] = {}
        self.by_parent: dict[str, set[str]] = {}
        self._order: list[str] = []
        self.by_value_identifier: dict[str, ValueProxy] = {}
        self.by_owner_instance: dict[str, set[str]] = {}
        self._value_order: list[str] = []

        self._effect_counter = 0
        self.epoch = 0
        self.effect_log: list[dict[str, Any]] = []
        self.active_instance_id: str | None = None

        self._dirty = False
        self._handle_cache: dict[str, ir.Value] = {}
        self._value_handle_cache: dict[str, ir.Value] = {}

        self._rebuild_from_module_tree()
        self._init_transform()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            return False
        self.refresh()
        ir.finalize_transform(self.module)
        return True

    #################
    # Constructors
    #################
    @classmethod
    def from_module(
        cls,
        module: ir.ModuleOp,
    ):
        return cls(module)

    @classmethod
    def from_string(
        cls,
        context: ir.Context,
        s: str,
    ):
        context.load_dialects()
        context.load_transform_dialects()
        module = ir.parse_from_string(context, s)
        return cls(module, context)

    @classmethod
    def from_file(
        cls,
        context: ir.Context,
        filename: str,
    ):
        context.load_dialects()
        context.load_transform_dialects()
        module = ir.parse_from_file(context, filename)
        return cls(module, context)

    @property
    def handles(self) -> list[HandleProxy]:
        return [self.by_instance[hid] for hid in self._order]

    @property
    def values(self) -> list[ValueProxy]:
        return [self.by_value_identifier[vid] for vid in self._value_order]

    @property
    def active(self) -> HandleProxy | None:
        if self.active_instance_id is None:
            return None
        return self.by_instance[self.active_instance_id]

    @property
    def dirty(self) -> bool:
        return self._dirty

    #################
    # Init / Rebuild
    #################
    def _init_transform(self):
        self.builder = ir.AlloOpBuilder(self.context)

        self.builder.set_insertion_point_to_end(self.module.body)
        sched_mod = ir.ModuleOp.create(self.builder)
        sched_mod.set_attr(
            "transform.with_named_sequence", ir.UnitAttr.get(self.context)
        )
        self.sched_mod = sched_mod

        self.builder.set_insertion_point_to_start(sched_mod.body)
        op_ty = tran_d.OperationType.get(self.context, "builtin.module")
        seq = tran_d.NamedSequenceOp.create(self.builder, "__transform_main", op_ty, [])
        self.entry = seq
        self.entry_block = seq.entry_block
        self.root_handle = seq.get_arg_at(0)

        self.builder.set_insertion_point_to_end(self.entry_block)
        tran_d.YieldOp.create(self.builder, [])

        self.builder.set_insertion_point_to_start(self.entry_block)
        self._dirty = False
        self._handle_cache.clear()
        self._value_handle_cache.clear()

    def _rebuild_from_module_tree(self, *, overwrite_identifiers: bool = False):
        stats = ir.complete_op_identifiers(self.module, overwrite=overwrite_identifiers)
        self.effect_log.append(
            {
                "op": "complete_identifiers",
                "assigned": int(stats["assigned"]),
                "rewritten": int(stats["rewritten"]),
                "visited": int(stats["visited"]),
                "overwrite_identifiers": overwrite_identifiers,
            }
        )
        self.root = ir.parse_proxy_tree(self.module)
        self._clear_indexes()
        self._ingest_node(self.root, None)
        for handle in self.handles:
            if handle.state == HandleState.VALID and handle.is_looplike():
                self.active_instance_id = handle.instance_id
                break

    def _clear_indexes(self):
        self.by_instance.clear()
        self.by_identifier.clear()
        self.by_path.clear()
        self.by_parent.clear()
        self._order = []
        self.by_value_identifier.clear()
        self.by_owner_instance.clear()
        self._value_order = []
        self.active_instance_id = None

    def _register_proxy(self, proxy: HandleProxy):
        if proxy.instance_id in self.by_instance:
            raise ValueError(f"Duplicate proxy instance_id: {proxy.instance_id}")
        self.by_instance[proxy.instance_id] = proxy
        self._order.append(proxy.instance_id)
        self.by_identifier.setdefault(proxy.identifier, set()).add(proxy.instance_id)
        self.by_path.setdefault(proxy.path, set()).add(proxy.instance_id)
        if proxy.parent_instance_id is not None:
            self.by_parent.setdefault(proxy.parent_instance_id, set()).add(
                proxy.instance_id
            )

    def _register_value_proxy(self, value_proxy: ValueProxy):
        if value_proxy.value_identifier in self.by_value_identifier:
            raise ValueError(
                f"Duplicate value identifier: {value_proxy.value_identifier}"
            )
        self.by_value_identifier[value_proxy.value_identifier] = value_proxy
        self._value_order.append(value_proxy.value_identifier)
        if value_proxy.owner_instance_id is not None:
            self.by_owner_instance.setdefault(value_proxy.owner_instance_id, set()).add(
                value_proxy.value_identifier
            )

    def _ingest_node(self, node: ir.ProxyNode, parent_instance_id: str | None):
        path = str(node.hierarchy_name)
        identifier = str(node.op_identifier)

        current_parent = parent_instance_id
        if identifier != "":
            proxy = HandleProxy(
                instance_id=identifier,
                identifier=identifier,
                path=path if path != "" else identifier,
                parent_instance_id=parent_instance_id,
                op_kind=str(node.op_kind),
                meta={"origin": "ir", "epoch": self.epoch},
            )
            self._register_proxy(proxy)
            current_parent = proxy.instance_id

        for value in node.values:
            value_id = str(value.value_identifier)
            if value_id == "":
                continue
            owner_identifier = str(value.owner_op_identifier)
            owner_instance_id: str | None = None
            if owner_identifier in self.by_instance:
                owner_instance_id = owner_identifier
            elif identifier != "" and owner_identifier == identifier:
                owner_instance_id = identifier
            value_proxy = ValueProxy(
                value_identifier=value_id,
                owner_identifier=owner_identifier,
                owner_instance_id=owner_instance_id,
                owner_op_kind=str(value.owner_op_kind),
                source_kind=str(value.source_kind),
                source_index=int(value.source_index),
                type_str=str(value.type_str),
                is_memref=bool(value.is_memref),
                root_kind=str(value.root_kind),
                root_owner_identifier=str(value.root_owner_identifier),
                root_arg_number=int(value.root_arg_number),
            )
            self._register_value_proxy(value_proxy)

        for child in node.children:
            self._ingest_node(child, current_parent)

    #################
    # Helpers
    #################
    def _mark_dirty(self):
        self._dirty = True

    def _refresh_builder_loc_from_callsite(self):
        frame = pyinspect.currentframe()
        try:
            if frame is not None:
                frame = frame.f_back
            internal_file = __file__
            while frame is not None and frame.f_code.co_filename == internal_file:
                frame = frame.f_back
            if frame is None:
                self.builder.set_unknown_loc()
                return
            self.builder.loc = ir.Location(
                frame.f_code.co_filename, frame.f_lineno, 1, self.context
            )
        finally:
            del frame

    def _normalize_targets(
        self,
        targets: str | HandleProxy | Iterable[str | HandleProxy],
        action: str,
    ) -> list[str | HandleProxy]:
        if isinstance(targets, (str, HandleProxy)):
            return [targets]
        result = list(targets)
        if len(result) == 0:
            raise ValueError(f"Empty target list for `{action}`.")
        return result

    def _resolve_identifier_instance(self, identifier: str, action: str) -> str:
        ids = self.by_identifier.get(identifier)
        if ids is None or len(ids) == 0:
            raise ValueError(f"Identifier '{identifier}' not found for `{action}`.")

        valid_ids = [
            hid
            for hid in sorted(ids)
            if self.by_instance[hid].state == HandleState.VALID
        ]
        if len(valid_ids) == 1:
            return valid_ids[0]
        if len(valid_ids) > 1:
            candidates = ", ".join(valid_ids)
            raise ValueError(
                f"Identifier '{identifier}' is ambiguous for `{action}`. Candidates: {candidates}."
            )

        if len(ids) == 1:
            only = next(iter(ids))
            state = self.by_instance[only].state.value
            raise ValueError(
                f"Identifier '{identifier}' refers to a {state} handle for `{action}`."
            )

        candidates = ", ".join(sorted(ids))
        raise ValueError(
            f"Identifier '{identifier}' has no valid handles for `{action}`. Candidates: {candidates}."
        )

    def _resolve_proxy(
        self,
        target: str | HandleProxy | None,
        action: str,
        *,
        require_loop: bool = False,
    ) -> HandleProxy:
        if target is None:
            if self.active_instance_id is None:
                raise ValueError(f"No active handle for `{action}`.")
            resolved_id = self.active_instance_id
        elif isinstance(target, HandleProxy):
            resolved_id = target.instance_id
        elif isinstance(target, str):
            resolved_id = (
                target
                if target in self.by_instance
                else self._resolve_identifier_instance(target, action)
            )
        else:
            raise TypeError(f"Unsupported target type for `{action}`: {type(target)}")

        if resolved_id not in self.by_instance:
            raise ValueError(f"Handle '{resolved_id}' not found for `{action}`.")
        proxy = self.by_instance[resolved_id]

        if proxy.state != HandleState.VALID:
            raise ValueError(
                f"Handle '{proxy.identifier}' is {proxy.state.value}; cannot `{action}`."
            )
        if require_loop and not proxy.is_looplike():
            raise ValueError(
                f"Handle '{proxy.identifier}' ({proxy.op_kind}) is not loop-like; cannot `{action}`."
            )
        return proxy

    def _materialize_payload_handle(self, proxy: HandleProxy) -> ir.Value:
        """Materialize (or reuse) a transform handle that matches this proxy."""
        cached = self._handle_cache.get(proxy.instance_id)
        if cached is not None:
            return cached

        attrs = {
            ir.OP_IDENTIFIER_ATTR_NAME: self.builder.get_str_attr(proxy.identifier),
        }
        dict_attr = self.builder.get_dict_attr(attrs)
        op_names = [proxy.op_kind] if proxy.op_kind != "" else []
        res_ty = tran_d.AnyOpType.get(self.context)
        self._refresh_builder_loc_from_callsite()
        value = tran_d.MatchOp.create(
            self.builder, self.root_handle, res_ty, op_names, dict_attr
        )
        self._mark_dirty()
        self._handle_cache[proxy.instance_id] = value
        return value

    def _resolve_value_proxy(self, target: str | ValueProxy, action: str) -> ValueProxy:
        if isinstance(target, ValueProxy):
            value_id = target.value_identifier
        elif isinstance(target, str):
            value_id = target
        else:
            raise TypeError(
                f"Unsupported value target type for `{action}`: {type(target)}"
            )

        value_proxy = self.by_value_identifier.get(value_id)
        if value_proxy is None:
            raise ValueError(f"Value identifier '{value_id}' not found for `{action}`.")
        if value_proxy.owner_instance_id is None:
            raise ValueError(
                f"Value '{value_proxy.value_identifier}' has no owner handle for `{action}`."
            )
        owner = self.by_instance.get(value_proxy.owner_instance_id)
        if owner is None:
            raise ValueError(
                f"Owner handle '{value_proxy.owner_instance_id}' not found for `{action}`."
            )
        if owner.state != HandleState.VALID:
            raise ValueError(
                f"Owner handle '{owner.identifier}' is {owner.state.value}; "
                f"cannot `{action}` on value '{value_proxy.value_identifier}'."
            )
        return value_proxy

    def _materialize_value_handle(self, value_proxy: ValueProxy) -> ir.Value:
        cached = self._value_handle_cache.get(value_proxy.value_identifier)
        if cached is not None:
            return cached

        owner_id = value_proxy.owner_instance_id
        if owner_id is None:
            raise ValueError(
                f"Value '{value_proxy.value_identifier}' has no owner handle for matching."
            )
        owner_proxy = self.by_instance[owner_id]
        owner_payload = self._materialize_payload_handle(owner_proxy)
        source_kind = 0
        if value_proxy.source_kind == "arg":
            source_kind = 1
        elif value_proxy.source_kind == "res":
            source_kind = 2
        self._refresh_builder_loc_from_callsite()
        value_handle = allo_d.MatchValueOp.create(
            self.builder, owner_payload, value_proxy.source_index, source_kind
        )
        self._mark_dirty()
        self._value_handle_cache[value_proxy.value_identifier] = value_handle
        return value_handle

    def _invalidate_value_cache_for_owner(self, owner_instance_id: str):
        for value_id in self.by_owner_instance.get(owner_instance_id, ()):
            self._value_handle_cache.pop(value_id, None)

    def _descendants(self, root_id: str) -> list[str]:
        stack = list(self.by_parent.get(root_id, ()))
        out: list[str] = []
        while len(stack) > 0:
            node_id = stack.pop()
            out.append(node_id)
            stack.extend(self.by_parent.get(node_id, ()))
        return out

    def _reparent_handle(self, child_id: str, new_parent_id: str | None):
        """Update front-end hierarchy edge: parent(child) := new_parent_id."""
        if child_id not in self.by_instance:
            raise ValueError(f"Cannot reparent unknown handle: {child_id}")
        child = self.by_instance[child_id]
        old_parent_id = child.parent_instance_id
        if old_parent_id == new_parent_id:
            return

        if old_parent_id is not None:
            siblings = self.by_parent.get(old_parent_id)
            if siblings is not None:
                siblings.discard(child_id)
                if len(siblings) == 0:
                    self.by_parent.pop(old_parent_id, None)

        child.parent_instance_id = new_parent_id
        if new_parent_id is not None:
            self.by_parent.setdefault(new_parent_id, set()).add(child_id)

    def _is_strict_parent_chain(self, proxies: list[HandleProxy]) -> bool:
        """Return True if proxies are strictly nested in the given order."""
        if len(proxies) <= 1:
            return True
        for i in range(1, len(proxies)):
            if proxies[i].parent_instance_id != proxies[i - 1].instance_id:
                return False
        return True

    def _apply_reorder_hierarchy_effect(
        self, proxies: list[HandleProxy], order: list[int]
    ) -> bool:
        """Update semantic parent links after loop reorder.

        The handle identifiers stay unchanged. We only remap parent-child
        relations so stale propagation follows reordered loop semantics.
        """
        if len(proxies) <= 1:
            return True
        if not self._is_strict_parent_chain(proxies):
            return False

        selected_ids = [proxy.instance_id for proxy in proxies]
        selected_set = set(selected_ids)
        old_outer = proxies[0]
        old_innermost = proxies[-1]
        external_parent = old_outer.parent_instance_id

        # Non-selected body ops owned by the old innermost loop should move to
        # the new innermost loop after reorder.
        old_innermost_children = list(self.by_parent.get(old_innermost.instance_id, ()))
        order_index = {hid: idx for idx, hid in enumerate(self._order)}
        old_innermost_children.sort(key=lambda hid: order_index.get(hid, 0))
        non_selected_children = [
            hid for hid in old_innermost_children if hid not in selected_set
        ]

        new_chain = [proxies[idx] for idx in order]
        self._reparent_handle(new_chain[0].instance_id, external_parent)
        for i in range(1, len(new_chain)):
            self._reparent_handle(
                new_chain[i].instance_id, new_chain[i - 1].instance_id
            )

        new_innermost_id = new_chain[-1].instance_id
        for child_id in non_selected_children:
            self._reparent_handle(child_id, new_innermost_id)
        return True

    def _consume_with_descendants(
        self, source: HandleProxy, *, stale_descendants: bool = True
    ):
        source.state = HandleState.CONSUMED
        self._handle_cache.pop(source.instance_id, None)
        self._invalidate_value_cache_for_owner(source.instance_id)
        if self.active_instance_id == source.instance_id:
            self.active_instance_id = None
        if not stale_descendants:
            return
        for desc_id in self._descendants(source.instance_id):
            desc = self.by_instance[desc_id]
            self._handle_cache.pop(desc.instance_id, None)
            self._invalidate_value_cache_for_owner(desc.instance_id)
            if desc.state == HandleState.VALID:
                desc.state = HandleState.STALE

    def _next_effect_instance_id(self, base_id: str, tag: str) -> str:
        self._effect_counter += 1
        return f"{base_id}::{tag}{self._effect_counter}"

    def _transform_result_id(self, base: str, suffix: str) -> str:
        """Build identifier for transform-derived results."""
        return f"{base}::{suffix}"

    def _spawn_proxy(
        self,
        source: HandleProxy,
        *,
        identifier: str,
        tag: str,
        path: str | None = None,
        meta_updates: dict[str, Any] | None = None,
    ) -> HandleProxy:
        new_meta = dict(source.meta)
        new_meta["epoch"] = self.epoch
        if meta_updates is not None:
            new_meta.update(meta_updates)

        proxy = HandleProxy(
            instance_id=self._next_effect_instance_id(source.instance_id, tag),
            identifier=identifier,
            path=identifier if path is None else path,
            parent_instance_id=source.parent_instance_id,
            op_kind=source.op_kind,
            meta=new_meta,
        )
        self._register_proxy(proxy)
        return proxy

    def _apply_handle_effects(
        self,
        *,
        consumes: list[_HandleConsume] | None = None,
        provides: list[_HandleProvide] | None = None,
        stale_descendants: bool = True,
    ) -> list[HandleProxy]:
        """Apply front-end lifecycle updates for one transform step.

        This is the single source of truth for handle validity transitions:
        consumed handles become `CONSUMED`; valid descendants become `STALE`;
        provided handles are registered as new `VALID` proxies.
        """
        consumes = [] if consumes is None else consumes
        provides = [] if provides is None else provides

        consumed_seen: set[str] = set()
        for consume in consumes:
            source = consume.proxy
            if source.instance_id in consumed_seen:
                continue
            consumed_seen.add(source.instance_id)
            self._consume_with_descendants(source, stale_descendants=stale_descendants)

        produced: list[HandleProxy] = []
        selected: HandleProxy | None = None
        for provide in provides:
            proxy = self._spawn_proxy(
                provide.source,
                identifier=provide.identifier,
                tag=provide.tag,
                path=provide.path,
                meta_updates=provide.meta_updates,
            )
            self._handle_cache[proxy.instance_id] = provide.value
            produced.append(proxy)
            if provide.select:
                selected = proxy

        if selected is None and len(produced) > 0:
            selected = produced[-1]
        if selected is not None:
            self.active_instance_id = selected.instance_id
        return produced

    #################
    # Developer APIs
    #################
    def dev_resolve_targets(
        self,
        targets: str | HandleProxy | None | Iterable[str | HandleProxy | None],
        *,
        action: str,
        require_loop: bool = False,
        deduplicate: bool = True,
    ) -> list[HandleProxy]:
        """Resolve user-facing targets for extension transforms.

        :targets: One target or a target list (identifier or `HandleProxy`).
        :action: Action name used in validation errors.
        :require_loop: Require each resolved handle to be loop-like.
        :deduplicate: If True, deduplicate by instance id while preserving order.
        """
        if targets is None:
            target_list: list[str | HandleProxy | None] = [None]
        else:
            target_list = self._normalize_targets(targets, action)
        resolved = [
            self._resolve_proxy(target, action, require_loop=require_loop)
            for target in target_list
        ]
        if not deduplicate:
            return resolved
        out: list[HandleProxy] = []
        seen: set[str] = set()
        for proxy in resolved:
            if proxy.instance_id in seen:
                continue
            seen.add(proxy.instance_id)
            out.append(proxy)
        return out

    def dev_materialize_payloads(
        self, proxies: Iterable[HandleProxy]
    ) -> list[ir.Value]:
        """Materialize transform handles for resolved proxies."""
        return [self._materialize_payload_handle(proxy) for proxy in proxies]

    def dev_rebind_payloads(
        self,
        proxies: Iterable[HandleProxy],
        payloads: Iterable[ir.Value],
        *,
        select: int | None = None,
        meta_updates: dict[str, Any] | None = None,
    ) -> list[HandleProxy]:
        """Rebind payload handles for non-consuming transforms.

        Use this when a transform updates payload mapping for existing handles
        (e.g. merge/split based orchestration) but should keep their lifecycle
        state as `VALID` instead of consume/provide replacement.
        """
        proxy_list = list(proxies)
        payload_list = list(payloads)
        if len(proxy_list) != len(payload_list):
            raise ValueError(
                f"`dev_rebind_payloads` expects equal lengths, got "
                f"{len(proxy_list)} proxies and {len(payload_list)} payloads."
            )

        seen: set[str] = set()
        for proxy in proxy_list:
            if proxy.instance_id in seen:
                raise ValueError(
                    f"`dev_rebind_payloads` expects unique proxies, got duplicate "
                    f"'{proxy.identifier}'."
                )
            seen.add(proxy.instance_id)
            if proxy.state != HandleState.VALID:
                raise ValueError(
                    f"Handle '{proxy.identifier}' is {proxy.state.value}; "
                    "cannot rebind payload."
                )

        for proxy, payload in zip(proxy_list, payload_list):
            self._handle_cache[proxy.instance_id] = payload
            if meta_updates is not None:
                proxy.meta.update(meta_updates)

        if select is not None:
            if select < 0 or select >= len(proxy_list):
                raise ValueError(
                    f"`select` out of range for `dev_rebind_payloads`: {select}"
                )
            self.active_instance_id = proxy_list[select].instance_id
        return proxy_list

    def dev_resolve_values(
        self,
        targets: str | ValueProxy | Iterable[str | ValueProxy],
        *,
        action: str,
        deduplicate: bool = True,
    ) -> list[ValueProxy]:
        """Resolve value targets for extension transforms."""
        target_list = (
            [targets] if isinstance(targets, (str, ValueProxy)) else list(targets)
        )
        if len(target_list) == 0:
            raise ValueError(f"Empty value target list for `{action}`.")
        resolved = [self._resolve_value_proxy(target, action) for target in target_list]
        if not deduplicate:
            return resolved
        out: list[ValueProxy] = []
        seen: set[str] = set()
        for value_proxy in resolved:
            if value_proxy.value_identifier in seen:
                continue
            seen.add(value_proxy.value_identifier)
            out.append(value_proxy)
        return out

    def dev_materialize_values(self, values: Iterable[ValueProxy]) -> list[ir.Value]:
        """Materialize transform value handles for resolved value proxies."""
        return [self._materialize_value_handle(value_proxy) for value_proxy in values]

    def dev_make_consume(self, proxy: HandleProxy) -> _HandleConsume:
        """Create a consume descriptor for `dev_finalize_transform`."""
        return _HandleConsume(proxy=proxy)

    def dev_make_provide(
        self,
        source: HandleProxy,
        *,
        identifier: str,
        value: ir.Value,
        tag: str,
        path: str | None = None,
        meta_updates: dict[str, Any] | None = None,
        select: bool = False,
    ) -> _HandleProvide:
        """Create a provide descriptor for `dev_finalize_transform`."""
        return _HandleProvide(
            source=source,
            identifier=identifier,
            value=value,
            tag=tag,
            path=path,
            meta_updates=meta_updates,
            select=select,
        )

    def dev_finalize_transform(
        self,
        *,
        transform: str,
        consumes: list[_HandleConsume] | None = None,
        provides: list[_HandleProvide] | None = None,
        mark_dirty: bool = False,
        stale_descendants: bool = True,
        log: dict[str, Any] | None = None,
    ) -> list[HandleProxy]:
        """Finalize one extension transform in a structured way.

        Intended flow for adding a new transform:
        1) `dev_resolve_targets` + `dev_materialize_payloads`
        2) emit transform ops with builder
        3) `dev_finalize_transform` to apply consume/provide effects and logging

        Parameters:
        - `stale_descendants`: if `True`, consuming a handle marks valid
          descendants as `STALE`; if `False`, descendants remain usable.
        """
        if mark_dirty:
            self._mark_dirty()
        produced = self._apply_handle_effects(
            consumes=consumes,
            provides=provides,
            stale_descendants=stale_descendants,
        )
        entry = {"op": transform}
        if log is not None:
            entry.update(log)
        self.effect_log.append(entry)
        return produced

    #################
    # Front-end APIs
    #################
    _pattern_map = {"canonicalize": tran_d.ApplyCanonicalizationOp}

    def select(self, target: str | HandleProxy):
        """Select the active handle for subsequent implicit-target transforms."""
        proxy = self._resolve_proxy(target, "select")
        self.active_instance_id = proxy.instance_id
        return self

    def query(
        self,
        *,
        op_kind: str | None = None,
        under: str | HandleProxy | None = None,
        state: HandleState | None = HandleState.VALID,
    ) -> list[HandleProxy]:
        """Query proxies with optional filters on kind, scope, and lifecycle state."""
        anchor = self._resolve_proxy(under, "query") if under is not None else None
        scope_ids: set[str] | None = None
        if anchor is not None:
            scope_ids = {anchor.instance_id}
            scope_ids.update(self._descendants(anchor.instance_id))
        result: list[HandleProxy] = []
        for handle in self.handles:
            if state is not None and handle.state != state:
                continue
            if op_kind is not None and handle.op_kind != op_kind:
                continue
            if scope_ids is not None and handle.instance_id not in scope_ids:
                continue
            result.append(handle)
        return result

    def cse(self, target: str | HandleProxy | None = None):
        """Apply CSE to target handle.

        Validate:
        - target handle is `VALID`

        Invalidate/Provide:
        - no handle is consumed
        - no new handle is provided
        """
        proxy = self._resolve_proxy(target, "cse")
        payload = self._materialize_payload_handle(proxy)
        self._refresh_builder_loc_from_callsite()
        tran_d.ApplyCSEOp.create(self.builder, payload)
        self._mark_dirty()
        self.dev_finalize_transform(transform="cse", log={"target": proxy.identifier})
        return self

    def dce(self, target: str | HandleProxy | None = None):
        """Apply DCE to target handle.

        Validate:
        - target handle is `VALID`

        Invalidate/Provide:
        - no handle is consumed
        - no new handle is provided
        """
        proxy = self._resolve_proxy(target, "dce")
        payload = self._materialize_payload_handle(proxy)
        self._refresh_builder_loc_from_callsite()
        tran_d.ApplyDCEOp.create(self.builder, payload)
        self._mark_dirty()
        self.dev_finalize_transform(transform="dce", log={"target": proxy.identifier})
        return self

    def apply_patterns(
        self,
        patterns: str | list[str],
        target: str | HandleProxy | None = None,
    ):
        """Apply transform pattern ops to target handle.

        Validate:
        - target handle is `VALID`
        - every pattern is supported

        Invalidate/Provide:
        - no handle is consumed
        - no new handle is provided
        """
        pattern_names = [patterns] if isinstance(patterns, str) else list(patterns)
        if len(pattern_names) == 0:
            raise ValueError("`apply_patterns` requires at least one pattern.")
        pattern_ops: list[Any] = []
        for pattern in pattern_names:
            op = self._pattern_map.get(pattern)
            if op is None:
                raise ValueError(f"Unsupported pattern: {pattern}")
            pattern_ops.append(op)

        proxy = self._resolve_proxy(target, "apply_patterns")
        payload = self._materialize_payload_handle(proxy)
        self._refresh_builder_loc_from_callsite()
        entry = tran_d.ApplyPatternsOp.create(self.builder, payload).body
        self._mark_dirty()

        ip = self.builder.save_insertion_point()
        self.builder.set_insertion_point_to_start(entry)
        for op in pattern_ops:
            self._refresh_builder_loc_from_callsite()
            op.create(self.builder)
        self.builder.restore_insertion_point(ip)
        self.dev_finalize_transform(
            transform="apply_patterns",
            log={"target": proxy.identifier, "patterns": pattern_names},
        )
        return self

    def canonicalize(self, target: str | HandleProxy | None = None):
        """Apply canonicalization patterns to target handle."""
        return self.apply_patterns("canonicalize", target)

    def licm(self, target: str | HandleProxy | None = None):
        """Apply LICM to target handle.

        Validate:
        - target handle is `VALID`

        Invalidate/Provide:
        - no handle is consumed
        - no new handle is provided
        """
        proxy = self._resolve_proxy(target, "licm")
        payload = self._materialize_payload_handle(proxy)
        self._refresh_builder_loc_from_callsite()
        tran_d.ApplyLICMOp.create(self.builder, payload)
        self._mark_dirty()
        self.dev_finalize_transform(transform="licm", log={"target": proxy.identifier})
        return self

    def to_affine(self, target: str | HandleProxy | None = None):
        """Raise loop handle to affine form.

        Validate:
        - target handle is `VALID` and loop-like

        Invalidate/Provide:
        - does not consume operation handles
        - does not invalidate descendants
        - rebinds target handle payload to raised loop result
        """
        source = self.dev_resolve_targets(
            target, action="to_affine", require_loop=True, deduplicate=True
        )[0]
        payload = self.dev_materialize_payloads([source])[0]
        self._refresh_builder_loc_from_callsite()
        raise_op = allo_d.RaiseToAffineOp.create(self.builder, payload)
        self._mark_dirty()
        self.dev_rebind_payloads(
            [source],
            [raise_op.get_result_at(0)],
            select=0,
            meta_updates={"raised_to_affine": True},
        )
        self.dev_finalize_transform(
            transform="to_affine",
            log={"target": source.identifier},
        )
        return self

    def unroll(
        self,
        target: str | HandleProxy | None = None,
        factor: int = 1,
        tag_only: bool = True,
    ):
        """Unroll loop handle.

        Validate:
        - target handle is `VALID` and loop-like
        - `factor > 0`

        Invalidate/Provide:
        - `tag_only=True`: no consume/provide
        - `tag_only=False`: consume target handle
        """
        if factor <= 0:
            raise ValueError(f"`unroll` requires positive factor, got {factor}.")
        source = self.dev_resolve_targets(
            target, action="unroll", require_loop=True, deduplicate=True
        )[0]
        payload = self.dev_materialize_payloads([source])[0]
        self._refresh_builder_loc_from_callsite()
        if tag_only:
            allo_d.TagUnrollOp.create(self.builder, payload, factor)
            self._mark_dirty()
            source.meta["unroll_factor"] = factor
            self.dev_finalize_transform(
                transform="unroll",
                log={
                    "target": source.identifier,
                    "factor": factor,
                    "tag_only": True,
                },
            )
            return self
        tran_d.LoopUnrollOp.create(self.builder, payload, factor)
        self._mark_dirty()
        self.dev_finalize_transform(
            transform="unroll",
            consumes=[self.dev_make_consume(source)],
            log={"target": source.identifier, "factor": factor, "tag_only": False},
        )
        return self

    def outline(
        self,
        target: str | HandleProxy | None = None,
        *,
        func_name: str,
    ):
        """Outline target operation/loop into a kernel call pair.

        Validate:
        - target handle is `VALID`
        - `func_name` is non-empty

        Invalidate/Provide:
        - consume target handle
        - provide `<source>` and `<source>::call`
        - set active handle to `<source>::call`
        """
        if len(func_name) == 0:
            raise ValueError("`outline` requires non-empty `func_name`.")
        source = self.dev_resolve_targets(
            target, action="outline", require_loop=False, deduplicate=True
        )[0]
        payload = self.dev_materialize_payloads([source])[0]
        self._refresh_builder_loc_from_callsite()
        outline_op = allo_d.OutlineOp.create(self.builder, payload, func_name)
        self._mark_dirty()
        self.dev_finalize_transform(
            transform="outline",
            consumes=[self.dev_make_consume(source)],
            provides=[
                self.dev_make_provide(
                    source=source,
                    identifier=source.identifier,
                    value=outline_op.get_result_at(0),
                    tag="outline_kernel_",
                    path=source.path,
                ),
                self.dev_make_provide(
                    source=source,
                    identifier=self._transform_result_id(source.identifier, "call"),
                    value=outline_op.get_result_at(1),
                    tag="outline_call_",
                    select=True,
                ),
            ],
            log={"target": source.identifier, "func_name": func_name},
        )
        return self

    def flatten(self, targets: str | HandleProxy | Iterable[str | HandleProxy]):
        """Flatten selected loop handles into one loop.

        Validate:
        - at least one target
        - each target handle is `VALID` and loop-like

        Invalidate/Provide:
        - consume all selected target handles
        - provide `<first_target>::flat`
        """
        resolved = self.dev_resolve_targets(
            targets, action="flatten", require_loop=True, deduplicate=True
        )
        payloads = self.dev_materialize_payloads(resolved)
        self._refresh_builder_loc_from_callsite()
        merged = tran_d.MergeHandlesOp.create(
            self.builder, payloads, deduplicate=True
        ).get_result_at(0)
        flatten_op = allo_d.LoopFlattenOp.create(self.builder, merged)
        self._mark_dirty()
        first = resolved[0]
        self.dev_finalize_transform(
            transform="flatten",
            consumes=[self.dev_make_consume(source) for source in resolved],
            provides=[
                self.dev_make_provide(
                    source=first,
                    identifier=self._transform_result_id(first.identifier, "flat"),
                    value=flatten_op.get_result_at(0),
                    tag="flatten_",
                    select=True,
                )
            ],
            log={"targets": [proxy.identifier for proxy in resolved]},
        )
        return self

    def compute_at(self, target: str | HandleProxy, axis: str | HandleProxy):
        """Move target to execute at axis loop.

        Validate:
        - target handle is `VALID`
        - axis handle is `VALID` and loop-like

        Invalidate/Provide:
        - consume target handle
        - axis stays valid (read-only)
        """
        target_proxy = self.dev_resolve_targets(
            target, action="compute_at target", require_loop=False, deduplicate=True
        )[0]
        axis_proxy = self.dev_resolve_targets(
            axis, action="compute_at axis", require_loop=True, deduplicate=True
        )[0]
        target_payload, axis_payload = self.dev_materialize_payloads(
            [target_proxy, axis_proxy]
        )
        self._refresh_builder_loc_from_callsite()
        allo_d.ComputeAtOp.create(self.builder, target_payload, axis_payload)
        self._mark_dirty()
        self.dev_finalize_transform(
            transform="compute_at",
            consumes=[self.dev_make_consume(target_proxy)],
            log={
                "target": target_proxy.identifier,
                "axis": axis_proxy.identifier,
            },
        )
        return self

    def reuse_at(
        self,
        producer: str | HandleProxy,
        consumer_loop: str | HandleProxy,
    ):
        """Insert reuse buffer at consumer loop level.

        Validate:
        - producer handle is `VALID`
        - consumer_loop handle is `VALID` and loop-like

        Invalidate/Provide:
        - consume producer
        - consume consumer_loop
        - provide replacement handle for consumer_loop identifier
        """
        producer_proxy = self.dev_resolve_targets(
            producer, action="reuse_at producer", require_loop=False, deduplicate=True
        )[0]
        consumer_proxy = self.dev_resolve_targets(
            consumer_loop,
            action="reuse_at consumer_loop",
            require_loop=True,
            deduplicate=True,
        )[0]
        producer_payload, consumer_payload = self.dev_materialize_payloads(
            [producer_proxy, consumer_proxy]
        )
        self._refresh_builder_loc_from_callsite()
        reuse_op = allo_d.ReuseAtOp.create(
            self.builder, producer_payload, consumer_payload
        )
        self._mark_dirty()
        self.dev_finalize_transform(
            transform="reuse_at",
            consumes=[
                self.dev_make_consume(producer_proxy),
                self.dev_make_consume(consumer_proxy),
            ],
            provides=[
                self.dev_make_provide(
                    source=consumer_proxy,
                    identifier=consumer_proxy.identifier,
                    value=reuse_op.get_result_at(0),
                    tag="reuse_at_",
                    path=consumer_proxy.path,
                    select=True,
                )
            ],
            log={
                "producer": producer_proxy.identifier,
                "consumer_loop": consumer_proxy.identifier,
            },
        )
        return self

    def pipeline(self, target: str | HandleProxy | None = None, ii: int = 1):
        """Attach pipeline annotation to a loop handle.

        Parameters:
        - `target`: loop identifier/proxy. If `None`, uses current active handle.
        - `ii`: positive initiation interval.

        Validate:
        - `ii > 0`
        - target handle is `VALID` and loop-like (`affine.for`/`scf.for`)

        Invalidate/Provide:
        - no handle is consumed
        - no new handle is provided
        - target remains `VALID`
        """
        if ii <= 0:
            raise ValueError(f"`pipeline` requires positive ii, got {ii}.")
        proxy = self._resolve_proxy(target, "pipeline", require_loop=True)
        payload = self._materialize_payload_handle(proxy)
        self._refresh_builder_loc_from_callsite()
        allo_d.TagPipelineOp.create(self.builder, payload, ii)
        self._mark_dirty()

        proxy.meta["pipeline_ii"] = ii
        self.active_instance_id = proxy.instance_id
        self.dev_finalize_transform(
            transform="pipeline",
            log={"target": proxy.identifier, "ii": ii},
        )
        return self

    def split(self, target: str | HandleProxy | None = None, factor: int = 1):
        """Split one loop into outer/inner loops.

        Parameters:
        - `target`: loop identifier/proxy. If `None`, uses current active handle.
        - `factor`: positive split factor.

        Validate:
        - `factor > 0`
        - target handle is `VALID` and loop-like

        Invalidate/Provide:
        - consume target handle
        - invalidate (mark `STALE`) all valid descendants of target
        - provide `<target>::outer` and `<target>::inner`
        - set active handle to `<target>::inner`
        """
        if factor <= 0:
            raise ValueError(f"`split` requires positive factor, got {factor}.")
        source = self.dev_resolve_targets(
            target,
            action="split",
            require_loop=True,
            deduplicate=True,
        )[0]
        payload = self.dev_materialize_payloads([source])[0]

        self._refresh_builder_loc_from_callsite()
        split_op = allo_d.LoopSplitOp.create(self.builder, payload, factor)
        self._mark_dirty()

        self.dev_finalize_transform(
            transform="split",
            consumes=[self.dev_make_consume(source)],
            provides=[
                self.dev_make_provide(
                    source=source,
                    identifier=self._transform_result_id(source.identifier, "outer"),
                    value=split_op.get_result_at(0),
                    tag="split_outer_",
                    meta_updates={"split_factor": factor},
                ),
                self.dev_make_provide(
                    source=source,
                    identifier=self._transform_result_id(source.identifier, "inner"),
                    value=split_op.get_result_at(1),
                    tag="split_inner_",
                    meta_updates={"split_factor": factor},
                    select=True,
                ),
            ],
            log={"target": source.identifier, "factor": factor},
        )
        return self

    def tile(
        self,
        targets: str | HandleProxy | Iterable[str | HandleProxy],
        factors: int | list[int] = 1,
    ):
        """Tile one or multiple loops in a nest.

        Parameters:
        - `targets`: one loop handle or a list of loop handles.
        - `factors`: a positive factor list (or single int promoted to list).

        Validate:
        - at least one factor
        - each target handle is `VALID` and loop-like
        - duplicate targets are deduplicated by instance id

        Invalidate/Provide:
        - consume all selected target handles
        - invalidate descendants of each consumed target
        - provide `<target>::tile` and `<target>::point` for each target
        - set active handle to last provided `::point`
        """
        if isinstance(factors, int):
            factors = [factors]
        if len(factors) == 0:
            raise ValueError("`tile` requires at least one factor.")

        deduped = self.dev_resolve_targets(
            targets,
            action="tile",
            require_loop=True,
            deduplicate=True,
        )

        payloads = self.dev_materialize_payloads(deduped)
        self._refresh_builder_loc_from_callsite()
        merged = tran_d.MergeHandlesOp.create(
            self.builder, payloads, deduplicate=True
        ).get_result_at(0)
        self._mark_dirty()
        tiled = allo_d.LoopTileOp.create(self.builder, merged, factors)
        tile_group = tiled.get_result_at(0)
        point_group = tiled.get_result_at(1)
        split_tile = tran_d.SplitHandleOp.create(self.builder, tile_group, len(deduped))
        split_point = tran_d.SplitHandleOp.create(
            self.builder, point_group, len(deduped)
        )

        provides: list[_HandleProvide] = []
        for i, source in enumerate(deduped):
            provides.append(
                self.dev_make_provide(
                    source=source,
                    identifier=self._transform_result_id(source.identifier, "tile"),
                    value=split_tile.get_result_at(i),
                    tag="tile_",
                    meta_updates={"tile_factors": list(factors)},
                )
            )
            provides.append(
                self.dev_make_provide(
                    source=source,
                    identifier=self._transform_result_id(source.identifier, "point"),
                    value=split_point.get_result_at(i),
                    tag="point_",
                    meta_updates={"tile_factors": list(factors)},
                    select=(i + 1 == len(deduped)),
                )
            )

        self.dev_finalize_transform(
            transform="tile",
            consumes=[self.dev_make_consume(source) for source in deduped],
            provides=provides,
            log={
                "targets": [p.identifier for p in deduped],
                "factors": list(factors),
            },
        )
        return self

    def reorder(self, targets: Iterable[str | HandleProxy], order: list[int]):
        """Reorder selected loops according to a permutation.

        Parameters:
        - `targets`: ordered loop handles participating in reorder.
        - `order`: permutation over `targets` indices.

        Validate:
        - each target handle is `VALID` and loop-like
        - `len(order) == len(targets)`
        - `order` is a permutation of `[0, ..., n-1]`
        - targets are unique (no duplicated handle instances)

        Invalidate/Provide:
        - does not consume operation handles
        - does not invalidate descendants
        - refreshes payload handles via `merge_handles + split_handle`
        - keeps selected identifiers usable for follow-up transforms
        """
        resolved = self.dev_resolve_targets(
            list(targets),
            action="reorder",
            require_loop=True,
            deduplicate=False,
        )
        if len(order) != len(resolved):
            raise ValueError(
                f"`reorder` expects order length {len(resolved)}, got {len(order)}."
            )
        if sorted(order) != list(range(len(resolved))):
            raise ValueError(f"`reorder` expects a permutation, got {order}.")
        seen: set[str] = set()
        for source in resolved:
            if source.instance_id in seen:
                raise ValueError(
                    f"`reorder` expects unique targets, got duplicate '{source.identifier}'."
                )
            seen.add(source.instance_id)

        payloads = self.dev_materialize_payloads(resolved)
        self._refresh_builder_loc_from_callsite()
        merged = tran_d.MergeHandlesOp.create(
            self.builder, payloads, deduplicate=False
        ).get_result_at(0)
        allo_d.LoopReorderOp.create(self.builder, merged, order)
        split_back = tran_d.SplitHandleOp.create(self.builder, merged, len(resolved))
        self._mark_dirty()

        split_payloads = [split_back.get_result_at(i) for i in range(len(resolved))]
        self.dev_rebind_payloads(
            resolved,
            split_payloads,
            select=0 if len(resolved) > 0 else None,
            meta_updates={"reordered": True, "reorder_order": list(order)},
        )
        hierarchy_updated = self._apply_reorder_hierarchy_effect(resolved, order)

        self.dev_finalize_transform(
            transform="reorder",
            log={
                "targets": [p.identifier for p in resolved],
                "order": list(order),
                "hierarchy_updated": hierarchy_updated,
            },
        )
        return self

    def partition(
        self,
        target: str | ValueProxy,
        *,
        dim: int = 0,
        kind: allo_d.PartitionKind = allo_d.Complete,
        factor: int = 0,
    ):
        """Attach partition attribute to a concrete memref value.

        Parameters:
        - `target`: value identifier (`<op_id>:argN`/`<op_id>:resN`) or `ValueProxy`.
        - `dim`: partitioned axis, must be non-negative.
        - `kind`: partition kind (`Complete`, `Block`, `Cyclic`).
        - `factor`: partition factor. Must be 0 for `Complete`, >0 otherwise.

        Validate:
        - target resolves to a `VALID` owner and a memref value
        - `dim >= 0`
        - factor constraints by kind

        Invalidate/Provide:
        - does not consume/provide operation handles
        - payload IR is modified in place by attaching `allo.part`
        """
        value_proxy = self._resolve_value_proxy(target, "partition")
        if not value_proxy.is_memref:
            raise ValueError(
                f"Value '{value_proxy.value_identifier}' is not memref-typed; cannot `partition`."
            )
        if dim < 0:
            raise ValueError(f"`partition` requires non-negative dim, got {dim}.")
        if kind == allo_d.Complete and factor != 0:
            raise ValueError("Complete partition cannot have non-zero factor.")
        if kind != allo_d.Complete and factor <= 0:
            raise ValueError(
                f"{kind} partition must have positive factor, got {factor}."
            )

        value_handle = self.dev_materialize_values([value_proxy])[0]
        part = allo_d.PartitionAttr.get(self.context, [(dim, kind.value, factor)])
        self._refresh_builder_loc_from_callsite()
        allo_d.PartitionOp.create(self.builder, value_handle, part)
        self._mark_dirty()
        self.dev_finalize_transform(
            transform="partition",
            log={
                "target": value_proxy.value_identifier,
                "dim": dim,
                "kind": str(kind),
                "factor": factor,
                "root_kind": value_proxy.root_kind,
                "root_owner_identifier": value_proxy.root_owner_identifier,
                "root_arg_number": value_proxy.root_arg_number,
            },
        )
        return self

    def refresh(self):
        """Apply the buffered transform script to payload IR and rebuild proxies."""
        if not self._dirty:
            return self
        self.canonicalize("__allo_module__")
        if not self.sched_mod.verify():
            raise ValueError("Schedule module verification failed.")
        failed, err_msg = tran_d.apply_transforms(
            self.module.operation, self.entry.operation, self.sched_mod
        )
        if failed:
            raise RuntimeError(f"Schedule application failed:\n{err_msg}")
        if not self.module.verify():
            raise RuntimeError("Module verification failed after applying schedule.")

        self._dirty = False
        self._handle_cache.clear()
        self._value_handle_cache.clear()
        self._refresh_from_module_impl(
            reset_transform_script=True, overwrite_identifiers=True
        )
        return self

    def _refresh_from_module_impl(
        self,
        *,
        reset_transform_script: bool,
        overwrite_identifiers: bool,
    ):
        self.epoch += 1
        self._rebuild_from_module_tree(overwrite_identifiers=overwrite_identifiers)
        self.effect_log.append(
            {
                "op": "refresh",
                "epoch": self.epoch,
                "overwrite_identifiers": overwrite_identifiers,
            }
        )
        self._handle_cache.clear()
        self._value_handle_cache.clear()
        if reset_transform_script:
            self._init_transform()
        return self

    def refresh_from_module(
        self,
        module: ir.ModuleOp | None = None,
        *,
        reset_transform_script: bool = False,
    ):
        """Rebuild proxy/index state from current payload module."""
        if module is not None:
            self.module = module
        if self.module is None:
            raise ValueError("`refresh_from_module` requires an attached module.")
        return self._refresh_from_module_impl(
            reset_transform_script=reset_transform_script,
            overwrite_identifiers=False,
        )

    #################
    # Debug dumps
    #################
    def dump_handles(self, include_meta: bool = False) -> str:
        """Dump all handles with lifecycle state."""
        lines: list[str] = []
        for handle in self.handles:
            line = (
                f"{handle.identifier} [{handle.state.value}] inst={handle.instance_id} "
                f"kind={handle.op_kind} path={handle.path}"
            )
            if include_meta:
                line += f" meta={handle.meta}"
            lines.append(line)
        return "\n".join(lines)

    def dump_effect_log(self, last_n: int | None = None) -> str:
        """Dump recorded schedule effects."""
        if len(self.effect_log) == 0:
            return "<empty effect log>"
        start = 0
        if last_n is not None:
            if last_n <= 0:
                raise ValueError(f"`last_n` must be positive, got {last_n}.")
            start = max(0, len(self.effect_log) - last_n)
        lines: list[str] = []
        for idx in range(start, len(self.effect_log)):
            lines.append(f"[{idx}] {self.effect_log[idx]}")
        return "\n".join(lines)

    def dump_indexes(self) -> str:
        """Dump internal indexes (`by_identifier`, `by_path`, `by_parent`)."""
        lines: list[str] = ["by_identifier:"]
        for ident in sorted(self.by_identifier):
            ids = ", ".join(sorted(self.by_identifier[ident]))
            lines.append(f"  {ident}: [{ids}]")

        lines.append("by_path:")
        for path in sorted(self.by_path):
            ids = ", ".join(sorted(self.by_path[path]))
            lines.append(f"  {path}: [{ids}]")

        lines.append("by_parent:")
        for parent in sorted(self.by_parent):
            children = ", ".join(sorted(self.by_parent[parent]))
            lines.append(f"  {parent}: [{children}]")

        lines.append("by_value_identifier:")
        for value_id in sorted(self.by_value_identifier):
            value_proxy = self.by_value_identifier[value_id]
            lines.append(f"  {value_id}: owner={value_proxy.owner_instance_id}")

        lines.append("by_owner_instance:")
        for owner in sorted(self.by_owner_instance):
            values = ", ".join(sorted(self.by_owner_instance[owner]))
            lines.append(f"  {owner}: [{values}]")
        return "\n".join(lines)

    def dump_values(self) -> str:
        """Dump all structured value proxies."""
        lines: list[str] = []
        for value_proxy in self.values:
            lines.append(
                f"{value_proxy.value_identifier} owner={value_proxy.owner_identifier} "
                f"kind={value_proxy.source_kind}{value_proxy.source_index} "
                f"type={value_proxy.type_str} memref={value_proxy.is_memref} "
                f"root={value_proxy.root_kind} root_owner={value_proxy.root_owner_identifier} "
                f"root_arg={value_proxy.root_arg_number}"
            )
        return "\n".join(lines)

    def dump_state(
        self,
        *,
        include_handles: bool = True,
        include_values: bool = True,
        include_indexes: bool = True,
        include_effect_log: bool = True,
        include_tree: bool = False,
        include_meta: bool = False,
        last_n_effects: int | None = None,
    ) -> str:
        """Dump full scheduler internal state for debugging."""
        lines: list[str] = ["=== Sched State ==="]
        active = "<none>" if self.active is None else self.active.identifier
        lines.append(
            f"epoch={self.epoch}, active={active}, dirty={self._dirty}, "
            f"cached_handles={len(self._handle_cache)}, num_handles={len(self.by_instance)}, "
            f"cached_values={len(self._value_handle_cache)}, num_values={len(self.by_value_identifier)}, "
            f"num_effects={len(self.effect_log)}"
        )
        lines.append(f"effect_counter={self._effect_counter}")

        if include_handles:
            lines.append("--- handles ---")
            lines.append(self.dump_handles(include_meta=include_meta))
        if include_values:
            lines.append("--- values ---")
            lines.append(self.dump_values())
        if include_indexes:
            lines.append("--- indexes ---")
            lines.append(self.dump_indexes())
        if include_effect_log:
            lines.append("--- effect_log ---")
            lines.append(self.dump_effect_log(last_n=last_n_effects))
        if include_tree:
            lines.append("--- proxy_tree ---")
            lines.append(str(self.root))
        return "\n".join(lines)

    def debug_dump(self, **kwargs):
        """Print `dump_state` directly and return `self`."""
        print(self.dump_state(**kwargs))
        return self


__all__ = ["HandleState", "HandleProxy", "ValueProxy", "Schedule"]
