add_mlir_library(MLIRAlloTransformOps
        AlloTransformOps.cpp

        DEPENDS
        MLIRAlloIncGen

        LINK_LIBS PUBLIC
        MLIRIR
)