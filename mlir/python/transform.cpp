#include "ir.h"

using namespace mlir;

void bindTransform(nb::module_ &m) {
  m.def(
      "apply_transforms",
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
      },
      nb::arg("payload"), nb::arg("transform_root"),
      nb::arg("transform_module"));

  nb::class_<transform::OperationType, Type>(m, "OperationType")
      .def_static(
          "get",
          [](MLIRContext &context, std::string_view opName) {
            return transform::OperationType::get(
                &context, StringAttr::get(&context, opName));
          },
          nb::arg("context"), nb::arg("op_name"));

  nb::class_<transform::ParamType, Type>(m, "ParamType")
      .def_static(
          "get",
          [](MLIRContext &context, Type &type) {
            return transform::ParamType::get(&context, type);
          },
          nb::arg("context"), nb::arg("type"));

  nb::class_<transform::AnyOpType, Type>(m, "AnyOpType")
      .def_static(
          "get",
          [](MLIRContext &context) {
            return transform::AnyOpType::get(&context);
          },
          nb::arg("context"));

  nb::class_<transform::AnyParamType, Type>(m, "AnyParamType")
      .def_static(
          "get",
          [](MLIRContext &context) {
            return transform::AnyParamType::get(&context);
          },
          nb::arg("context"));

  auto annotateOp = bindOp<transform::AnnotateOp>(m, "AnnotateOp");
  bindConstructor(
      annotateOp,
      [](AlloOpBuilder &builder, Value &target, std::string_view name,
         Attribute value) {
        auto anyParam = transform::AnyParamType::get(builder.getContext());
        auto param = transform::ParamConstantOp::create(
            builder, builder.getLocation(), anyParam, value);
        return transform::AnnotateOp::create(builder, builder.getLocation(),
                                             target, name, param);
      },
      nb::arg("target"), nb::arg("name"), nb::arg("value"));

  auto getDefiningOp = bindOp<transform::GetDefiningOp>(m, "GetDefiningOp");
  bindConstructor(
      getDefiningOp,
      [](AlloOpBuilder &builder, Value &target) {
        auto anyOpType = transform::AnyOpType::get(builder.getContext());
        return transform::GetDefiningOp::create(builder, builder.getLocation(),
                                                anyOpType, target);
      },
      nb::arg("target"));

  auto namedSequence = bindOp<transform::NamedSequenceOp>(m, "NamedSequenceOp");
  bindConstructor(
      namedSequence,
      [](AlloOpBuilder &builder, std::string_view name, Type &rootType,
         const std::vector<Type> &resTypes) {
        auto bodyBuilder = [](OpBuilder &, Location, BlockArgument) { return; };
        return transform::NamedSequenceOp::create(
            builder, builder.getLocation(), name, rootType, resTypes,
            bodyBuilder);
      },
      nb::arg("name"), nb::arg("root_type"), nb::arg("res_types"))
      .def(
          "get_entry_block",
          [](transform::NamedSequenceOp &self) {
            return &self->getRegion(0).front();
          },
          nb::rv_policy::reference)
      .def(
          "get_arg_at",
          [](transform::NamedSequenceOp &self, unsigned idx) -> BlockArgument {
            return self.getArgument(idx);
          },
          nb::arg("idx"));

  auto yieldOp = bindOp<transform::YieldOp>(m, "YieldOp");
  bindConstructor(
      yieldOp,
      [](AlloOpBuilder &builder, const std::vector<Value> &operands) {
        return transform::YieldOp::create(builder, builder.getLocation(),
                                          operands);
      },
      nb::arg("operands"));

  // common transformations
  (void)bindUnaryValueOp<transform::ApplyCommonSubexpressionEliminationOp>(
      m, "ApplyCSEOp", "target");
  (void)bindUnaryValueOp<transform::ApplyDeadCodeEliminationOp>(m, "ApplyDCEOp",
                                                                "target");

  auto applyCanonicalization =
      bindOp<transform::ApplyCanonicalizationPatternsOp>(
          m, "ApplyCanonicalizationOp");
  bindConstructor(applyCanonicalization, [](AlloOpBuilder &builder) {
    return transform::ApplyCanonicalizationPatternsOp::create(
        builder, builder.getLocation());
  });

  (void)bindUnaryValueOp<transform::ApplyLoopInvariantCodeMotionOp>(
      m, "ApplyLICMOp", "target");

  auto applyPatterns = bindOp<transform::ApplyPatternsOp>(m, "ApplyPatternsOp");
  bindConstructor(
      applyPatterns,
      [](AlloOpBuilder &builder, Value &target) {
        return transform::ApplyPatternsOp::create(
            builder, builder.getLocation(), target);
      },
      nb::arg("target"))
      .def(
          "get_body",
          [](transform::ApplyPatternsOp &self) { return self.getBody(); },
          nb::rv_policy::reference);

  auto applyRegisteredPass =
      bindOp<transform::ApplyRegisteredPassOp>(m, "ApplyRegisteredPassOp");
  bindConstructor(
      applyRegisteredPass,
      [](AlloOpBuilder &builder, Value &target, std::string_view passName,
         DictionaryAttr passOptions, const std::vector<Value> &dynArgs) {
        auto anyOpType = transform::AnyOpType::get(builder.getContext());
        return transform::ApplyRegisteredPassOp::create(
            builder, builder.getLocation(), anyOpType, target, passName,
            passOptions, dynArgs);
      },
      nb::arg("target"), nb::arg("pass_name"), nb::arg("pass_options"),
      nb::arg("dynamic_args"));

  // operation matching
  auto matchOp = bindOp<transform::MatchOp>(m, "MatchOp");
  bindConstructor(
      matchOp,
      [](AlloOpBuilder &builder, Value &target, Type &resType,
         const std::vector<std::string> &opNames,
         std::optional<DictionaryAttr> opAttrs) {
        auto match = transform::MatchOp::create(builder, builder.getLocation(),
                                                resType, target);
        if (!opNames.empty()) {
          llvm::SmallVector<llvm::StringRef, 2> opNamesRef;
          for (const auto &name : opNames)
            opNamesRef.push_back(name);
          auto opNamesAttr = builder.getStrArrayAttr(opNamesRef);
          match->setAttr(match.getOpsAttrName(), opNamesAttr);
        }
        if (opAttrs.has_value())
          match->setAttr(match.getOpAttrsAttrName(), *opAttrs);
        return match;
      },
      nb::arg("target"), nb::arg("res_type"), nb::arg("op_names"),
      nb::arg("op_attrs") = std::nullopt);

  auto mergeHandles = bindOp<transform::MergeHandlesOp>(m, "MergeHandlesOp");
  bindConstructor(
      mergeHandles,
      [](AlloOpBuilder &builder, const std::vector<Value> &handles,
         bool deduplicate) {
        return transform::MergeHandlesOp::create(builder, builder.getLocation(),
                                                 handles, deduplicate);
      },
      nb::arg("handles"), nb::arg("deduplicate") = true);

  auto splitHandle = bindOp<transform::SplitHandleOp>(m, "SplitHandleOp");
  bindConstructor(
      splitHandle,
      [](AlloOpBuilder &builder, Value &handle, unsigned numSplits) {
        return transform::SplitHandleOp::create(builder, builder.getLocation(),
                                                handle, numSplits);
      },
      nb::arg("handle"), nb::arg("num_splits"));

  auto loopUnroll = bindOp<transform::LoopUnrollOp>(m, "LoopUnrollOp");
  bindConstructor(
      loopUnroll,
      [](AlloOpBuilder &builder, Value &target, int factor) {
        return transform::LoopUnrollOp::create(builder, builder.getLocation(),
                                               target,
                                               static_cast<uint64_t>(factor));
      },
      nb::arg("target"), nb::arg("factor"));

  nb::enum_<allo::PartitionKindEnum>(m, "PartitionKind")
      .value("Complete", allo::PartitionKindEnum::CompletePartition)
      .value("Block", allo::PartitionKindEnum::BlockPartition)
      .value("Cyclic", allo::PartitionKindEnum::CyclicPartition)
      .export_values();

  nb::class_<allo::PartitionAttr, Attribute>(m, "PartitionAttr")
      .def_static(
          "get",
          [](MLIRContext &context, nb::list &subPartitions) {
            SmallVector<allo::PartitionAxisAttr> partitionAxes;
            for (nb::handle item : subPartitions) {
              auto triple = nb::cast<nb::tuple>(item);
              if (triple.size() != 3) {
                throw nb::value_error(
                    "Each sub-partition must be a tuple/list of size 3: (dim, "
                    "kind, factor).");
              }
              int64_t dim = nb::cast<int64_t>(triple[0]);
              auto kind = nb::cast<allo::PartitionKindEnum>(triple[1]);
              int64_t factor = nb::cast<int64_t>(triple[2]);
              partitionAxes.push_back(
                  allo::PartitionAxisAttr::get(&context, kind, factor, dim));
            }
            return allo::PartitionAttr::get(&context, partitionAxes);
          },
          nb::arg("context"), nb::arg("sub_partitions"));
  PyAttributeRegistry::registerAttr<allo::PartitionAttr>();

  auto renameOp = bindOp<transform::RenameOp>(m, "RenameOp");
  bindConstructor(
      renameOp,
      [](AlloOpBuilder &builder, Value &target, std::string_view name) {
        return transform::RenameOp::create(builder, builder.getLocation(),
                                           target, name);
      },
      nb::arg("target"), nb::arg("name"));

  (void)bindUnaryValueOp<transform::RaiseToAffineOp>(m, "RaiseToAffineOp",
                                                     "target");

  auto outlineOp = bindOp<transform::OutlineOp>(m, "OutlineOp");
  bindConstructor(
      outlineOp,
      [](AlloOpBuilder &builder, Value &target, std::string_view kernelName) {
        return transform::OutlineOp::create(builder, builder.getLocation(),
                                            target, kernelName);
      },
      nb::arg("target"), nb::arg("kernel_name"));

  auto tagPipeline = bindOp<transform::TagPipelineOp>(m, "TagPipelineOp");
  bindConstructor(
      tagPipeline,
      [](AlloOpBuilder &builder, Value &target, int ii) {
        return transform::TagPipelineOp::create(
            builder, builder.getLocation(), target, static_cast<uint64_t>(ii));
      },
      nb::arg("target"), nb::arg("ii"));

  auto tagUnroll = bindOp<transform::TagUnrollOp>(m, "TagUnrollOp");
  bindConstructor(
      tagUnroll,
      [](AlloOpBuilder &builder, Value &target, int factor) {
        return transform::TagUnrollOp::create(builder, builder.getLocation(),
                                              target,
                                              static_cast<uint64_t>(factor));
      },
      nb::arg("target"), nb::arg("factor"));

  auto loopReorder = bindOp<transform::LoopReorderOp>(m, "LoopReorderOp");
  bindConstructor(
      loopReorder,
      [](AlloOpBuilder &builder, Value &target,
         const std::vector<int32_t> &order) {
        return transform::LoopReorderOp::create(builder, builder.getLocation(),
                                                target, order);
      },
      nb::arg("target"), nb::arg("order"));

  auto loopSplit = bindOp<transform::LoopSplitOp>(m, "LoopSplitOp");
  bindConstructor(
      loopSplit,
      [](AlloOpBuilder &builder, Value &target, int factor) {
        return transform::LoopSplitOp::create(builder, builder.getLocation(),
                                              target, factor);
      },
      nb::arg("target"), nb::arg("factor"));

  auto loopTile = bindOp<transform::LoopTileOp>(m, "LoopTileOp");
  bindConstructor(
      loopTile,
      [](AlloOpBuilder &builder, Value &target,
         const std::vector<int64_t> &factors) {
        return transform::LoopTileOp::create(builder, builder.getLocation(),
                                             target, factors);
      },
      nb::arg("target"), nb::arg("factors"));

  (void)bindUnaryValueOp<transform::LoopFlattenOp>(m, "LoopFlattenOp",
                                                   "target");

  auto reuseAt = bindOp<transform::ReuseAtOp>(m, "ReuseAtOp");
  bindConstructor(
      reuseAt,
      [](AlloOpBuilder &builder, Value &target, Value &axis, bool ring) {
        return transform::ReuseAtOp::create(builder, builder.getLocation(),
                                            target, axis, ring);
      },
      nb::arg("target"), nb::arg("axis"), nb::arg("ring") = false);

  auto computeAt = bindOp<transform::ComputeAtOp>(m, "ComputeAtOp");
  bindConstructor(
      computeAt,
      [](AlloOpBuilder &builder, Value &producer, Value &consumer) {
        return transform::ComputeAtOp::create(builder, builder.getLocation(),
                                              producer, consumer);
      },
      nb::arg("producer"), nb::arg("consumer_loop"));

  auto bufferAt = bindOp<transform::BufferAtOp>(m, "BufferAtOp");
  bindConstructor(
      bufferAt,
      [](AlloOpBuilder &builder, Value &target, Value &axis) {
        return transform::BufferAtOp::create(builder, builder.getLocation(),
                                             target, axis);
      },
      nb::arg("target"), nb::arg("axis"));

  auto matchValue = bindOp<transform::MatchValueOp>(m, "MatchValueOp");
  bindConstructor(
      matchValue,
      [](AlloOpBuilder &builder, Value &target, int64_t index,
         int64_t sourceKind) {
        return transform::MatchValueOp::create(builder, builder.getLocation(),
                                               target, index, sourceKind);
      },
      nb::arg("target"), nb::arg("index"), nb::arg("source_kind") = 0);

  auto partitionOp = bindOp<transform::PartitionOp>(m, "PartitionOp");
  bindConstructor(
      partitionOp,
      [](AlloOpBuilder &builder, Value &target,
         allo::PartitionAttr &partition) {
        return transform::PartitionOp::create(builder, builder.getLocation(),
                                              target, partition);
      },
      nb::arg("target"), nb::arg("partition"));
}
