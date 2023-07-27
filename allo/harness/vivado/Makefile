#==========================================================================
# Makefile
#==========================================================================
# @brief: A makefile the compiles and synthesizes the program
#
# @desc: 1. "make" runs csim by default
#        2. "make csim" compiles & executes the fixed-point implementation
#        3. "make clean" cleans up the directory


# Extract Vivado HLS include path
VHLS_PATH := $(dir $(shell which vivado_hls))/..
VHLS_INC ?= ${VHLS_PATH}/include

CFLAGS = -g -I${VHLS_INC} 

all: csim

csim:  host.cpp kernel.cpp 
	@echo "Compiling & simulating on amdpool ..."
	g++ ${CFLAGS} $^ -o out -lrt -std=gnu++0x
	@/usr/bin/time --format="Simulation time: %U seconds" ./out

vivado_hls:
	@echo "Run Vivado HLS"
	vivado_hls -f run.tcl

vitis_hls:
	@echo "Run Vitis HLS"
	vitis_hls -f run.tcl

clean:
	rm -rf out *.txt *.dat *.prj *.log *.json
	rm -rf zedboard_project* xillydemo.bit

