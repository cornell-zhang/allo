#include "ir.h"

using namespace mlir;

void init_transform(nb::module_ &m) {
  m.def("apply_transforms",
        [](Operation &payload, Operation &transformRoot,
           ModuleOp &transformModule) -> std::pair<bool, std::string> {
          // capture the error message from the diagnostic handler
          std::string errMsg;
          llvm::raw_string_ostream os(errMsg);
          llvm::SourceMgr sourceMgr;
          mlir::SourceMgrDiagnosticHandler handler(
              sourceMgr, transformModule->getContext(), os);

          transform::TransformOptions options;
          options.enableEnforceSingleToplevelTransformOp();
          auto ret = transform::applyTransformNamedSequence(
              &payload, &transformRoot, transformModule, options);
          os.flush();
          return {failed(ret), errMsg};
        });

  nb::class_<transform::OperationType, Type>(m, "OperationType")
      .def_static(
          "get",
          [](MLIRContext &context, const std::string &opName) {
            return transform::OperationType::get(
                &context, StringAttr::get(&context, opName));
          },
          nb::arg("builder"), nb::arg("op_name"));

  nb::class_<transform::ParamType, Type>(m, "ParamType")
      .def_static(
          "get",
          [](MLIRContext &context, Type &type) {
            return transform::ParamType::get(&context, type);
          },
          nb::arg("builder"), nb::arg("type"));

  nb::class_<transform::AnyOpType, Type>(m, "AnyOpType")
      .def_static("get", [](MLIRContext &context) {
        return transform::AnyOpType::get(&context);
      });

  nb::class_<transform::AnyParamType, Type>(m, "AnyParamType")
      .def_static("get", [](MLIRContext &context) {
        return transform::AnyParamType::get(&context);
      });

  nb::class_<transform::NamedSequenceOp, OpState>(m, "NamedSequenceOp")
      .def_static("create",
                  [](AlloOpBuilder &builder, const std::string &name,
                     Type &rootType, const std::vector<Type> &resTypes) {
                    // use a dummy body builder since the actual body will be
                    // built in Python
                    auto bodyBuilder = [](OpBuilder &, Location,
                                          BlockArgument) { return; };
                    return transform::NamedSequenceOp::create(
                        builder, builder.get_loc(), name, rootType, resTypes,
                        bodyBuilder);
                  })
      .def_prop_ro(
          "entry_block",
          [](transform::NamedSequenceOp &self) {
            return &self->getRegion(0).front();
          },
          nb::rv_policy::reference)
      .def("get_arg_at",
           [](transform::NamedSequenceOp &self, unsigned idx) -> BlockArgument {
             return self.getArgument(idx);
           });

  nb::class_<transform::YieldOp, OpState>(m, "YieldOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, const std::vector<Value> &operands) {
            return transform::YieldOp::create(builder, builder.get_loc(),
                                              operands);
          },
          nb::arg("builder"), nb::arg("operands"));

  // common transformations
  nb::class_<transform::ApplyCommonSubexpressionEliminationOp, OpState>(
      m, "ApplyCSEOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &target) {
            return transform::ApplyCommonSubexpressionEliminationOp::create(
                builder, builder.get_loc(), target);
          },
          nb::arg("builder"), nb::arg("target"));

  nb::class_<transform::ApplyDeadCodeEliminationOp, OpState>(m, "ApplyDCEOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &target) {
            return transform::ApplyDeadCodeEliminationOp::create(
                builder, builder.get_loc(), target);
          },
          nb::arg("builder"), nb::arg("target"));

  nb::class_<transform::ApplyCanonicalizationPatternsOp, OpState>(
      m, "ApplyCanonicalizationOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder) {
            return transform::ApplyCanonicalizationPatternsOp::create(
                builder, builder.get_loc());
          },
          nb::arg("builder"));

  nb::class_<transform::ApplyLoopInvariantCodeMotionOp, OpState>(m,
                                                                 "ApplyLICMOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &target) {
            return transform::ApplyLoopInvariantCodeMotionOp::create(
                builder, builder.get_loc(), target);
          },
          nb::arg("builder"), nb::arg("target"));

  nb::class_<transform::ApplyPatternsOp, OpState>(m, "ApplyPatternsOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &target) {
            return transform::ApplyPatternsOp::create(
                builder, builder.get_loc(), target);
          },
          nb::arg("builder"), nb::arg("target"))
      .def_prop_ro(
          "body",
          [](transform::ApplyPatternsOp &self) { return self.getBody(); },
          nb::rv_policy::reference);

  nb::class_<transform::ApplyRegisteredPassOp, OpState>(m,
                                                        "ApplyRegisteredPassOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &target, const std::string &passName,
             DictionaryAttr passOptions,
             const std::vector<Value> &dynamicArgs) {
            auto anyOpType = transform::AnyOpType::get(builder.getContext());
            return transform::ApplyRegisteredPassOp::create(
                builder, builder.get_loc(), anyOpType, target, passName,
                passOptions, dynamicArgs);
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("pass_name"),
          nb::arg("pass_options"), nb::arg("dynamic_args"));

  // operation matching
  nb::class_<transform::MatchOp, OpState>(m, "MatchOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &target, Type &resType,
             const std::vector<std::string> &opNames,
             std::optional<DictionaryAttr> opAttrs) -> Value {
            auto match = transform::MatchOp::create(builder, builder.get_loc(),
                                                    resType, target);
            if (!opNames.empty()) {
              llvm::SmallVector<llvm::StringRef, 2> opNamesRef;
              for (const auto &name : opNames) {
                opNamesRef.push_back(name);
              }
              auto opNamesAttr = builder.getStrArrayAttr(opNamesRef);
              match->setAttr(match.getOpsAttrName(), opNamesAttr);
            }
            if (opAttrs.has_value()) {
              match->setAttr(match.getOpAttrsAttrName(), *opAttrs);
            }
            return match;
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("res_type"),
          nb::arg("op_names"), nb::arg("op_attrs") = std::nullopt);

  nb::class_<transform::MergeHandlesOp, OpState>(m, "MergeHandlesOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, const std::vector<Value> &handles,
             bool deduplicate = true) {
            return transform::MergeHandlesOp::create(builder, builder.get_loc(),
                                                     handles, deduplicate);
          },
          nb::arg("builder"), nb::arg("handles"),
          nb::arg("deduplicate") = true);

  nb::class_<transform::SplitHandleOp, OpState>(m, "SplitHandleOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &handle, unsigned numSplits) {
            return transform::SplitHandleOp::create(builder, builder.get_loc(),
                                                    handle, numSplits);
          },
          nb::arg("builder"), nb::arg("handle"), nb::arg("num_splits"));

  nb::class_<transform::LoopUnrollOp, OpState>(m, "LoopUnrollOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &target, int factor) {
            return transform::LoopUnrollOp::create(
                builder, builder.get_loc(), target,
                static_cast<uint64_t>(factor));
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("factor"));
}

void init_allo_transforms(nb::module_ &m) {
  nb::class_<transform::RenameOp, OpState>(m, "RenameOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &target, const std::string &name) {
            return transform::RenameOp::create(builder, builder.get_loc(),
                                               target, name);
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("name"));

  nb::class_<transform::RaiseToAffineOp, OpState>(m, "RaiseToAffineOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &target) {
            return transform::RaiseToAffineOp::create(
                builder, builder.get_loc(), target);
          },
          nb::arg("builder"), nb::arg("target"));

  nb::class_<transform::OutlineOp, OpState>(m, "OutlineOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &target,
             const std::string &kernelName) {
            return transform::OutlineOp::create(builder, builder.get_loc(),
                                                target, kernelName);
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("kernel_name"));

  nb::class_<transform::TagPipelineOp, OpState>(m, "TagPipelineOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &target, int ii) {
            return transform::TagPipelineOp::create(
                builder, builder.get_loc(), target, static_cast<uint64_t>(ii));
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("ii"));

  nb::class_<transform::TagUnrollOp, OpState>(m, "TagUnrollOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &target, int factor) {
            return transform::TagUnrollOp::create(
                builder, builder.get_loc(), target,
                static_cast<uint64_t>(factor));
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("factor"));

  nb::class_<transform::LoopReorderOp, OpState>(m, "LoopReorderOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &target,
             const std::vector<int32_t> &order) {
            return transform::LoopReorderOp::create(builder, builder.get_loc(),
                                                    target, order);
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("order"));

  nb::class_<transform::LoopSplitOp, OpState>(m, "LoopSplitOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &target, int factor) {
            return transform::LoopSplitOp::create(builder, builder.get_loc(),
                                                  target, factor);
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("factor"));

  nb::class_<transform::LoopTileOp, OpState>(m, "LoopTileOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &target,
             const std::vector<int64_t> &factors) {
            return transform::LoopTileOp::create(builder, builder.get_loc(),
                                                 target, factors);
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("factors"));

  nb::class_<transform::LoopFlattenOp, OpState>(m, "LoopFlattenOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &target) {
            return transform::LoopFlattenOp::create(builder, builder.get_loc(),
                                                    target);
          },
          nb::arg("builder"), nb::arg("target"));

  nb::class_<transform::ReuseAtOp, OpState>(m, "ReuseAtOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &target, Value &axis) {
            return transform::ReuseAtOp::create(builder, builder.get_loc(),
                                                target, axis);
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("axis"));

  nb::class_<transform::ComputeAtOp, OpState>(m, "ComputeAtOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &producer, Value &consumer) {
            return transform::ComputeAtOp::create(builder, builder.get_loc(),
                                                  producer, consumer);
          },
          nb::arg("builder"), nb::arg("producer"), nb::arg("consumer_loop"));

  nb::class_<transform::MatchValueOp, OpState>(m, "MatchValueOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &target, int64_t index,
             int64_t sourceKind) -> Value {
            return transform::MatchValueOp::create(builder, builder.get_loc(),
                                                   target, index, sourceKind);
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("index"),
          nb::arg("source_kind") = 0);

  nb::class_<transform::PartitionOp, OpState>(m, "PartitionOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &target,
             allo::PartitionAttr &partition) {
            return transform::PartitionOp::create(builder, builder.get_loc(),
                                                  target, partition);
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("partition"));
}
