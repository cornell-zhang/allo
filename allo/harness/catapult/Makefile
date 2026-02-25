#==========================================================================
# Makefile
#==========================================================================
# @brief: A makefile that compiles and synthesizes the program using Catapult HLS
#
# @desc: 1. "make" runs catapult synthesis by default
#        2. "make catapult" runs Catapult HLS synthesis
#        3. "make clean" cleans up the directory

all: catapult

catapult:
	@echo "Running Catapult HLS synthesis..."
	catapult -shell -f run.tcl

clean:
	rm -rf out *.txt *.dat *.prj *.log *.json
	rm -rf catapult.log solution* *work* *.rpt *.cfg 