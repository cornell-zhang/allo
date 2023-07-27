# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# ======================================== Check Xilinx SDX Environment Settings ================================================== #
ifndef XILINX_SDX
  $(error Environment variable XILINX_SDX is required and should point to SDx install area)
endif

# =============================================== Tools Used in Rosetta =========================================================== #

# sdaccel tools
OCL_CXX   = xcpp
XOCC      = xocc

# sdsoc tools
SDSXX     = sds++

# default sw compiler
SW_CXX = g++

# ============================================= SDAccel Platform and Target Settings ============================================== #

# Set Default OpenCL device and platform
USR_PLATFORM = n
OCL_DEVICE = xilinx:adm-pcie-7v3:1ddr:3.0
OCL_PLATFORM = $(AWS_PLATFORM)

# Check if the user specified opencl platform
ifneq ($(OCL_PLATFORM), one_of_default_platforms)
  USR_PLATFORM=y
endif

# Check OCL_TARGET value
OCL_TARGET  = sw_emu
ifeq ($(OCL_TARGET),sw_emu)
else ifeq ($(OCL_TARGET),hw_emu)
else ifeq ($(OCL_TARGET),hw)
else
  $(error "OCL_TARGET does not support the $(OCL_TARGET) value. Supported values are: sw_emu, hw_emu, hw")
endif

# Check opencl kernel file type
OCL_KERNEL_TYPE = ocl

ifeq ($(suffix $(OCL_KERNEL_SRC)),.cl)
  OCL_KERNEL_TYPE=ocl
else
  OCL_KERNEL_TYPE=c
endif

# OpenCL runtime Libraries
ifndef XILINX_XRT
  OPENCL_INC = /opt/xilinx/Xilinx_SDx_2018.2_0614_1954/SDx/2018.2/runtime/include/1_2/
  OPENCL_LIB = /opt/xilinx/Xilinx_SDx_2018.2_0614_1954/SDx/2018.2/runtime/lib/x86_64
else
  OPENCL_INC = $(XILINX_XRT)/include/
  OPENCL_LIB = $(XILINX_XRT)/lib/
endif

# opencl harness files
OCL_HARNESS_DIR     = .
OCL_HARNESS_SRC_CPP = $(OCL_HARNESS_DIR)/CLKernel.cpp $(OCL_HARNESS_DIR)/CLMemObj.cpp $(OCL_HARNESS_DIR)/CLWorld.cpp
OCL_HARNESS_SRC_H   = $(OCL_HARNESS_DIR)/CLKernel.h   $(OCL_HARNESS_DIR)/CLMemObj.h   $(OCL_HARNESS_DIR)/CLWorld.h

# host compilation flags
OCL_HOST_FLAGS = -DOCL -g -lxilinxopencl -I$(OPENCL_INC) $(HOST_INC) -L$(OPENCL_LIB) $(HOST_LIB) -I$(OCL_HARNESS_DIR) -I$(APPLICATION_DIR)

# xclbin compilation flags
XCLBIN_FLAGS = -s -t $(OCL_TARGET) -g 
XCLBIN_FLAGS += --profile_kernel data:all:all:all

# change OCL_HOST_FLAG
ifdef K_CONST
 OCL_HOST_FLAGS += -DK_CONST=$(K_CONST)
endif
ifdef NUM_ITER 
 OCL_HOST_FLAGS += -DNUM_ITER=$(NUM_ITER)
endif
ifdef FIXED_FLAG
 OCL_HOST_FLAGS += -DFIXED_TYPE
endif


ifneq ($(KERNEL_TYPE),ocl)
  XCLBIN_FLAGS += --kernel $(KERNEL_NAME)
endif

ifeq ($(USR_PLATFORM),n)
  XCLBIN_FLAGS += --xdevice $(OCL_DEVICE)
else
  XCLBIN_FLAGS += --platform $(OCL_PLATFORM)
endif


# change XCLBIN_FLAGS
ifdef K_CONST
 XCLBIN_FLAGS += -DK_CONST=$(K_CONST)
endif
ifdef NUM_ITER
  XCLBIN_FLAGS += -DNUM_ITER=$(NUM_ITER)
endif
ifdef FIXED_FLAG
  XCLBIN_FLAGS += -DFIXED_TYPE
endif


XCLBIN_FLAGS += $(OCL_KERNEL_ARGS)


# host exe
OCL_HOST_EXE        = $(KERNEL_NAME)_host.exe

# Kernel XCLBIN file
XCLBIN        = $(KERNEL_NAME).$(OCL_TARGET).xclbin
XO            = $(KERNEL_NAME).$(OCL_TARGET).xo

# =============================================== SDSoC Platform and Target Settings ============================================== #

# platform
SDSOC_PLATFORM = zc706

# executable
SDSOC_EXE = $(KERNEL_NAME).elf

# sds++ flags
SDSFLAGS = -sds-pf $(SDSOC_PLATFORM) -sds-hw $(KERNEL_NAME) $(SDSOC_KERNEL_SRC) -sds-end -clkid 3  \
           -poll-mode 1 -verbose
SDSCFLAGS += -DSDSOC -Wall -O3 -c
SDSCFLAGS += -MMD -MP -MF"$(@:%.o=%.d)"
SDSLFLAGS = -O3 

# objects
ALL_SDSOC_SRC = $(HOST_SRC_CPP) $(SDSOC_KERNEL_SRC)
OBJECTS := $(ALL_SDSOC_SRC:.cpp=.o)
DEPS := $(OBJECTS:.o=.d)

# =============================================== Pure Software Compilation Settings ============================================== #

# compiler flags
SW_FLAGS = -DSW -O3

# sw executable
SW_EXE = $(KERNEL_NAME)_sw.exe

# ========================================================= Rules ================================================================= #

# we will have 4 top-level rules: ocl, sdsoc, sw and clean
# default to sw

.PHONY: all ocl sdsoc sw host clean

all: sw

# host rule 
host: $(OCL_HOST_EXE)

# ocl rules
ocl: $(OCL_HOST_EXE) $(XCLBIN)

# ocl secondary rule: host executable
$(OCL_HOST_EXE): $(HOST_SRC_CPP) $(HOST_SRC_H) $(OCL_HARNESS_SRC_CPP) $(OCL_HARNESS_SRC_H) $(DATA)
	$(OCL_CXX) $(OCL_HOST_FLAGS) -o $@ $(HOST_SRC_CPP) $(OCL_HARNESS_SRC_CPP) 

# ocl secondary rule: xclbin 
$(XCLBIN): $(XO)
	$(XOCC) -l --xp param:compiler.useHlsGpp=1 $(XCLBIN_FLAGS) -o $@ $(XO)

# ocl secondary rule: xo
$(XO): $(OCL_KERNEL_SRC) $(OCL_KERNEL_H)
	$(XOCC) -c $(XCLBIN_FLAGS) -o $@ $(OCL_KERNEL_SRC)

# sdsoc rules
sdsoc: $(SDSOC_EXE)

$(SDSOC_EXE): $(OBJECTS)
	$(SDSXX) $(SDSFLAGS) $(SDSLFLAGS) ${OBJECTS} -o $@

-include $(DEPS)

%.o: %.cpp
	$(SDSXX) $(SDSFLAGS) $(SDSCFLAGS) $< -o $@


# software rules
sw: $(HOST_SRC_CPP) $(HOST_SRC_H) $(SW_KENREL_SRC) $(SW_KERNEL_H) $(DATA)
	$(SW_CXX) $(SW_FLAGS) -o $(SW_EXE) $(HOST_SRC_CPP) $(SW_KERNEL_SRC)

# cleanup
clean:
	@echo "Cleaning old files"
	rm -rf *.exe
	rm -rf *.elf
	rm -rf *.xclbin
	rm -rf *.bit
	rm -rf *.rpt
	rm -rf system_estimate.xtxt
	rm -rf _xocc*
	rm -rf _sds
	rm -rf sd_card
	rm -rf .Xil
	rm -rf ./src/host/*.d
	rm -rf ./src/sdsoc/*.o
	rm -rf ./src/sdsoc/*.d
	rm -rf ./src/host/*.o
	rm -rf *.dat
	rm -rf *.html
	rm -rf *.csv
	rm -rf *.json
