#include "ir.h"

using namespace mlir;
using namespace mlir::allo;

void init_allo_ir(nb::module_ &m) {
  nb::enum_<allo::PartitionKindEnum>(m, "PartitionKind")
      .value("Complete", allo::PartitionKindEnum::CompletePartition)
      .value("Block", allo::PartitionKindEnum::BlockPartition)
      .value("Cyclic", allo::PartitionKindEnum::CyclicPartition)
      .export_values();

  nb::class_<allo::PartitionAttr, Attribute>(m, "PartitionAttr")
      .def_static(
          "get",
          [](MLIRContext &context, nb::list subPartitions) {
            SmallVector<PartitionAxisAttr> partitionAxes;
            for (nb::handle item : subPartitions) {
              auto triple = nb::cast<nb::tuple>(item);
              if (triple.size() != 3) {
                throw nb::value_error(
                    "Each sub-partition must be a tuple/list of size 3: (dim, "
                    "kind, factor).");
              }
              int64_t dim = nb::cast<int64_t>(triple[0]);
              uint32_t kind = nb::cast<uint32_t>(triple[1]);
              int64_t factor = nb::cast<int64_t>(triple[2]);
              partitionAxes.push_back(PartitionAxisAttr::get(
                  &context, static_cast<PartitionKindEnum>(kind), factor, dim));
            }
            return allo::PartitionAttr::get(&context, partitionAxes);
          },
          nb::arg("context"), nb::arg("sub_partitions"));
  PyAttributeRegistry::registerAttr<allo::PartitionAttr>();
}
