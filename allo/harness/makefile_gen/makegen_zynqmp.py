# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Reference: https://github.com/Xilinx/Vitis_Accel_Examples/tree/master/common/utility/makefile_gen
# pylint: skip-file

import json
import os

# ini flags
config_file = 0


def mk_copyright(target):
    target.write(
        """#
# Copyright 2019-2021 Xilinx, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# makefile-generator v1.0.3
#
"""
    )
    return


def create_params(target, data):
    target.write(
        "############################## Setting up Project Variables ##############################\n"
    )
    target.write("TARGET := hw\n")
    target.write(
        "SYSROOT := $(EDGE_COMMON_SW)/sysroots/cortexa72-cortexa53-xilinx-linux\n"
    )
    target.write("SD_IMAGE_FILE := $(EDGE_COMMON_SW)/Image\n")
    target.write("VPP_LDFLAGS :=\n")
    target.write("include ./utils.mk\n")
    target.write("\n")
    target.write("TEMP_DIR := ./_x.$(TARGET).$(XSA)\n")
    target.write("BUILD_DIR := ./build_dir.$(TARGET).$(XSA)\n")
    target.write("\n")

    if "containers" in data:
        for con in data["containers"]:
            target.write("LINK_OUTPUT := $(BUILD_DIR)/")
            target.write(con["name"] + ".link.xclbin\n")
    target.write("\n")
    target.write("EMU_PS := QEMU\n")
    target.write("ifeq ($(TARGET), sw_emu)\n")
    target.write("EMU_PS := X86\n")
    target.write("endif\n\n")
    target.write("# SoC variables\n")
    target.write("RUN_APP_SCRIPT = ./run_app.sh\n")
    target.write("PACKAGE_OUT = ./package.$(TARGET)\n")
    target.write("\n")
    target.write("LAUNCH_EMULATOR = $(PACKAGE_OUT)/launch_$(TARGET).sh\n")
    target.write("RESULT_STRING = TEST PASSED\n")
    target.write("\nVPP_PFLAGS := ")
    if "launch" in data:
        target.write("\n")
        target.write("CMD_ARGS =")
        cmd_args = data["launch"][0]["cmd_args"].split(" ")
        for cmdargs in cmd_args[0:]:
            target.write(" ")
            cmdargs = cmdargs.replace("BUILD", "$(BUILD_DIR)")
            cmdargs = cmdargs.replace("REPO_DIR", "$(XF_PROJ_ROOT)")
            cmdargs = cmdargs.replace("PROJECT", ".")
            target.write(cmdargs)
    target.write("\n")
    target.write("SD_CARD := $(PACKAGE_OUT)\n")
    target.write("\n")
    if "config_make" in data:
        target.write("include ")
        target.write(data["config_make"])
        target.write("\n\n")
    target.write("ifeq ($(EMU_PS), X86)\n")
    target.write(
        "CXXFLAGS += -I$(XILINX_XRT)/include -I$(XILINX_VIVADO)/include -Wall -O0 -g -std=c++1y\n"
    )
    if "host" in data:
        if "linker" in data["host"]:
            if "libraries" in data["host"]["linker"]:
                if (
                    "xrt_coreutil" in data["host"]["linker"]["libraries"]
                    and "uuid" in data["host"]["linker"]["libraries"]
                ):
                    target.write("CXXFLAGS += $(GXX_EXTRA_FLAGS)\n")
    target.write("LDFLAGS += -L$(XILINX_XRT)/lib -pthread -lOpenCL\n")
    target.write("else\n")
    target.write(
        "CXXFLAGS += -I$(SYSROOT)/usr/include/xrt -I$(XILINX_VIVADO)/include -Wall -O0 -g -std=c++1y\n"
    )
    target.write("LDFLAGS += -L$(SYSROOT)/usr/lib -pthread -lxilinxopencl\n")
    target.write("endif\n\n")
    target.write("ifeq ($(TARGET),$(filter $(TARGET),sw_emu))\n")
    target.write("VPP_PFLAGS+= --package.emu_ps qemu\n")
    target.write("endif\n\n")
    target.write("#Check for EMU_PS\n")
    target.write("ifeq ($(TARGET), $(filter $(TARGET),hw_emu hw))\n")
    target.write("ifeq ($(EMU_PS), X86)\n")
    target.write(
        "$(error For hw_emu and hw, the design has to run on QEMU. Thus, please give EMU_PS=QEMU for these targets.)\n"
    )
    target.write("endif\n")
    target.write("endif\n")

    blocklist = [board for board in data.get("platform_blocklist", [])]
    forbid_others = False
    target.write(
        "\n########################## Checking if PLATFORM in allowlist #######################\n"
    )
    if blocklist:
        target.write("PLATFORM_BLOCKLIST += ")
        for board in blocklist:
            if board != "others":
                target.write(board)
                target.write(" ")
            else:
                forbid_others = True
        target.write("\n")
    allowlist = [board for board in data.get("platform_allowlist", [])]
    if allowlist:
        target.write("PLATFORM_ALLOWLIST += ")
        for board in allowlist:
            target.write(board)
            target.write(" ")
        target.write("\n\n")
    return


def add_host_flags(target, data):
    target.write(
        "############################## Setting up Host Variables ##############################\n"
    )
    target.write("#Include Required Host Source Files\n")

    if "host" in data:
        if "compiler" in data["host"]:
            if "includepaths" in data["host"]["compiler"]:
                for path in data["host"]["compiler"]["includepaths"]:
                    path = path.replace("BUILD", "$(BUILD_DIR)")
                    path = path.replace("PROJECT", ".")
                    path = path.replace("REPO_DIR", "$(XF_PROJ_ROOT)")
                    target.write("CXXFLAGS += -I" + path + "\n")

    target.write("HOST_SRCS += ")
    source_flag = 0
    if "sources" in data["host"]["compiler"]:
        for src in data["host"]["compiler"]["sources"]:
            src = src.replace("PROJECT", ".")
            src = src.replace("REPO_DIR", "$(XF_PROJ_ROOT)")
            target.write(src + " ")
            source_flag += 1
    if not source_flag:
        target.write("src/host.cpp\n")
    target.write("\n")
    target.write("# Host compiler global settings\n")
    target.write("CXXFLAGS += ")
    target.write("-fmessage-length=0")

    if "compiler" in data["host"]:
        if "options" in data["host"]["compiler"]:
            option = data["host"]["compiler"]["options"].split(" ")
            for opt in option[0:]:
                target.write(" ")
                target.write(opt)
    target.write("\n")
    target.write("LDFLAGS += ")
    target.write("-lrt -lstdc++ ")
    if "linker" in data["host"]:
        if "libraries" in data["host"]["linker"]:
            target.write("\nLDFLAGS +=")
            for library in data["host"]["linker"]["libraries"]:
                target.write(" -l")
                target.write(library)
    if "linker" in data["host"]:
        if "options" in data["host"]["linker"]:
            target.write("\nLDFLAGS +=")
            option = data["host"]["linker"]["options"].split(" ")
            for opt in option[0:]:
                target.write(" ")
                target.write(opt)
    target.write("\n")
    target.write("ifneq ($(EMU_PS), X86)\n")
    target.write("LDFLAGS += --sysroot=$(SYSROOT)\n")
    target.write("endif\n")

    return


def add_kernel_flags(target, data):
    target.write(
        "############################## Setting up Kernel Variables ##############################\n"
    )
    target.write("# Kernel compiler global settings\n")

    if "v++" in data:
        target.write("VPP_FLAGS += \n")
    target.write("VPP_FLAGS += ")
    target.write("--save-temps \n")
    if "containers" in data:
        for con in data["containers"]:
            for acc in con["accelerators"]:
                if "max_memory_ports" in acc:
                    target.write("VPP_FLAGS += ")
                    target.write(" --max_memory_ports ")
                    target.write(acc["name"])
                    target.write("\n")

    if "containers" in data:
        for con in data["containers"]:
            for acc in con["accelerators"]:
                if "clflags" in acc:
                    target.write("VPP_FLAGS_" + acc["name"] + " += ")
                    flags = acc["clflags"].split(" ")
                    for flg in flags[0:]:
                        target.write(" ")
                        flg = flg.replace("PROJECT", ".")
                        target.write(flg)
                    target.write("\n")

    if "compiler" in data["host"]:
        if "symbols" in data["host"]["compiler"]:
            target.write("\nCXXFLAGS +=")
            for sym in data["host"]["compiler"]["symbols"]:
                target.write(" ")
                target.write("-D")
                target.write(sym)
        target.write("\n")

    if "containers" in data:
        for con in data["containers"]:
            if "ldclflags" in con:
                target.write("\n")
                target.write("# Kernel linker flags\n")
                target.write("VPP_LDFLAGS_" + con["name"] + " +=")
                ldclflags = con["ldclflags"].split(" ")
                for flg in ldclflags[0:]:
                    target.write(" ")
                    flg = flg.replace("PROJECT", ".")
                    target.write(flg)
            target.write("\n")

    if "platform_properties" in data:
        target.write("# Kernel linker flags")
        for key, value in data["platform_properties"].items():
            if "v++" in value:
                if "linker" in value["v++"]:
                    if "ldclflags" in value["v++"]["linker"]:
                        target.write("\nifeq ($(findstring ")
                        target.write(key)
                        target.write(", $(PLATFORM)), ")
                        target.write(key)
                        target.write(")\n")
                        target.write("VPP_LDFLAGS +=")
                        ldclflags = value["v++"]["linker"]["ldclflags"][0].split(" ")
                        for flg in ldclflags[0:]:
                            target.write(" ")
                            flg = flg.replace("PROJECT", ".")
                            target.write(flg)
                        target.write("\nendif")
        target.write("\n")

    target.write("EXECUTABLE = ./")
    if "host_exe" in data["host"]:
        target.write(data["host"]["host_exe"])
    else:
        target.write("host")
    target.write("\n")

    target.write("EMCONFIG_DIR = $(TEMP_DIR)\n")
    target.write("\n")

    if "v++" in data:
        if "compiler" in data["v++"]:
            if "includepaths" in data["v++"]["compiler"]:
                for path in data["v++"]["compiler"]["includepaths"]:
                    path = path.replace("BUILD", "$(BUILD_DIR)")
                    path = path.replace("PROJECT", ".")
                    path = path.replace("REPO_DIR", "$(XF_PROJ_ROOT)")
                    target.write("VPP_FLAGS += -I" + path + "\n")
                target.write("\n")

    if "v++" in data:
        if "compiler" in data["v++"]:
            if "clflags" in data["v++"]["compiler"]:
                clflags = data["v++"]["compiler"]["clflags"]
                target.write("VPP_FLAGS +=")
                for path in clflags:
                    path = path.replace("BUILD", "$(BUILD_DIR)")
                    path = path.replace("PROJECT", ".")
                    path = path.replace("REPO_DIR", "$(XF_PROJ_ROOT)")
                    target.write(" " + path)
                target.write("\n\n")

    if "v++" in data:
        if "compiler" in data["v++"]:
            if "symbols" in data["v++"]["compiler"]:
                target.write("VPP_FLAGS += ")
                for symb in data["v++"]["compiler"]["symbols"]:
                    target.write("-D" + symb + " ")
                target.write("\n\n")

    return


def building_kernel(target, data):
    if "containers" in data:
        target.write(
            "############################## Setting Rules for Binary Containers (Building Kernels) ##############################\n"
        )
        for con in data["containers"]:
            if "accelerators" in con:
                for acc in con["accelerators"]:
                    type_systemC = 0
                    if "kernel_type" in acc:
                        if acc["kernel_type"] == "SystemC":
                            type_systemC = 1
                    if not type_systemC:
                        target.write("$(TEMP_DIR)/")
                    target.write(acc["name"])
                    target.write(".xo: ")
                    location = acc["location"]
                    location = location.replace("PROJECT", ".")
                    target.write(location)
                    target.write("\n")
                    target.write("\tmkdir -p $(TEMP_DIR)\n")
                    if type_systemC:
                        target.write("\tcreate_sc_xo ")
                        target.write(location)
                        target.write("\n")
                    else:
                        target.write("\tv++ -c ")
                        target.write(
                            "$(VPP_FLAGS) -t $(TARGET) --platform $(PLATFORM) "
                        )
                        if "clflags" in acc:
                            target.write("$(VPP_FLAGS_" + acc["name"] + ") ")
                        target.write("-k ")
                        target.write(acc["name"])
                        target.write(" --temp_dir ")
                        target.write("$(TEMP_DIR) ")
                        target.write(" -I'$(<D)'")
                        target.write(" -o'$@' '$<'\n")
        target.write("\n")
        target.write("$(LINK_OUTPUT):")
        for con in data["containers"]:
            if "accelerators" in con:
                for acc in con["accelerators"]:
                    if "kernel_type" in acc and acc["kernel_type"] == "SystemC":
                        target.write(" ")
                    else:
                        target.write(" $(TEMP_DIR)/")
                    target.write(acc["name"])
                    target.write(".xo")
            target.write("\n")
            target.write("\tmkdir -p $(BUILD_DIR)\n")
            target.write(
                "\tv++ -l $(VPP_FLAGS) $(VPP_LDFLAGS) -t $(TARGET) --platform $(PLATFORM) --temp_dir $(TEMP_DIR)"
            )
            if "ldclflags" in con:
                target.write(" $(VPP_LDFLAGS_" + con["name"] + ")")
            target.write(" -o'$(LINK_OUTPUT)' $(+)\n")
        target.write("\n")
        target.write("ifeq ($(EMU_PS), X86)\n")
        target.write(
            "\tv++ -p $(LINK_OUTPUT) $(VPP_FLAGS) -t $(TARGET) --platform $(PLATFORM) "
        )
        target.write("-o $(BUILD_DIR)/" + con["name"] + ".xclbin\n")
    target.write("endif\n")
    return


def building_kernel_rtl(target, data):
    target.write("# Building kernel\n")
    target.write("$(LINK_OUTPUT):")
    if "containers" in data:
        for con in data["containers"]:
            if "accelerators" in con:
                for acc in con["accelerators"]:
                    if "kernel_type" in acc and acc["kernel_type"] == "SystemC":
                        target.write(" ")
                    else:
                        target.write(" $(TEMP_DIR)/")
                    target.write(acc["name"])
                    target.write(".xo")
            target.write("\n")
            target.write("\tmkdir -p $(BUILD_DIR)\n")
            target.write(
                "\tv++ -l $(VPP_FLAGS) $(VPP_LDFLAGS) -t $(TARGET) --platform $(PLATFORM) --temp_dir $(TEMP_DIR)"
            )
            if "ldclflags" in con:
                target.write(" $(VPP_LDFLAGS_" + con["name"] + ")")
            target.write(" -o'$(LINK_OUTPUT)' $(+)\n")
    if "testinfo" in data:
        if "targets" in data["testinfo"]:
            if "vitis_sw_emu" in data["testinfo"]["targets"]:
                if "containers" in data:
                    for con in data["containers"]:
                        target.write("ifeq ($(EMU_PS), X86)\n")
                        target.write(
                            "\tv++ -p $(LINK_OUTPUT) -t $(TARGET) --platform $(PLATFORM) --package.out_dir $(PACKAGE_OUT) -o "
                        )
                        target.write("$(BUILD_DIR)/" + con["name"] + ".xclbin\n")
                        target.write("endif\n")
    return


def building_host(target, data):
    target.write(
        "############################## Setting Rules for Host (Building Host Executable) ##############################\n"
    )

    target.write("$(EXECUTABLE): $(HOST_SRCS)\n")
    target.write("ifeq ($(EMU_PS), X86)\n")
    target.write("\t$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)\n")
    target.write("else\n")
    target.write("\tmake check_edge_sw\n")
    target.write("\tmake check-vitis\n")
    target.write(
        "\t$(XILINX_VITIS)/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-g++ -o $@ $^ $(CXXFLAGS) $(LDFLAGS)\n"
    )
    target.write("endif\n\n")
    target.write("emconfig:$(EMCONFIG_DIR)/emconfig.json\n")
    target.write("$(EMCONFIG_DIR)/emconfig.json:\n")
    target.write("\temconfigutil --platform $(PLATFORM) --od $(EMCONFIG_DIR)")
    if "num_devices" in data:
        target.write(" --nd ")
        target.write(data["num_devices"])
    target.write("\n\n")
    return


def mk_clean(target, data):
    target.write(
        "############################## Cleaning Rules ##############################\n"
    )

    target.write("# Cleaning stuff\n")
    target.write("clean:\n")
    target.write("\t-$(RMDIR) $(EXECUTABLE) *.xclbin/{*sw_emu*,*hw_emu*} \n")
    target.write("\t-$(RMDIR) profile_* TempConfig system_estimate.xtxt *.rpt *.csv \n")
    target.write(
        "\t-$(RMDIR) src/*.ll *v++* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb\n"
    )
    target.write("\n")

    target.write("cleanall: clean\n")
    target.write("\t-$(RMDIR) build_dir*")
    target.write(" sd_card*")
    target.write("\n")
    target.write("\t-$(RMDIR) package.*\n")
    target.write(
        "\t-$(RMDIR) _x* *xclbin.run_summary qemu-memory-_* emulation _vimage pl* start_simulation.sh *.xclbin\n"
    )
    if "output_files" in data:
        target.write("\t-$(RMDIR) ")
        for files in data["output_files"]:
            target.write("./")
            target.write(files)
            target.write(" ")
    target.write("\n")

    return


def mk_build_all(target, data):
    target.write(
        "############################## Setting Targets ##############################\n"
    )
    target.write(".PHONY: all clean cleanall docs emconfig\n")
    target.write("ifeq ($(EMU_PS), X86)\n")
    target.write("all: check-platform check-device $(EXECUTABLE) $(LINK_OUTPUT)")
    target.write(" emconfig\n")
    target.write("else\n")
    target.write(
        "all: check-platform check-device check_edge_sw $(EXECUTABLE) $(LINK_OUTPUT)"
    )
    target.write(" sd_card\n")
    target.write("endif\n\n")

    target.write(".PHONY: host\n")
    target.write("host: $(EXECUTABLE)\n")
    target.write("\n")

    target.write(".PHONY: build\n")
    target.write("build: check-vitis check-device $(LINK_OUTPUT)")
    target.write("\n\n")

    target.write(".PHONY: xclbin\n")
    target.write("xclbin: build\n")
    target.write("\n")

    rtl_counter = 0
    if "containers" in data:
        for con in data["containers"]:
            if "accelerators" in con:
                for acc in con["accelerators"]:
                    if "kernel_type" in acc:
                        if acc["kernel_type"] == "RTL":
                            rtl_counter = 1

    if rtl_counter == 1:
        building_kernel_rtl(target, data)
    else:
        building_kernel(target, data)
    mk_sdcard(target, data)
    building_host(target, data)
    return


def mk_run(target, data):
    target.write(
        "############################## Setting Essential Checks and Running Rules ##############################\n"
    )

    target.write("run: all\n")
    target.write("ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))\n")
    target.write("ifeq ($(EMU_PS), X86)\n")
    target.write("\tcp -rf $(EMCONFIG_DIR)/emconfig.json .\n")
    target.write("\tXCL_EMULATION_MODE=$(TARGET) $(EXECUTABLE) $(CMD_ARGS)\n")
    target.write("else\n")
    target.write(
        "\tbash -c '$(LAUNCH_EMULATOR) -run-app $(RUN_APP_SCRIPT) | tee run_app.log; exit $${PIPESTATUS[0]}'\n"
    )
    target.write("endif\n")

    if "post_launch" in data:
        for post_launch in data["post_launch"]:
            if "launch_cmd" in post_launch:
                target.write("\n")
                args = post_launch["launch_cmd"]
                args = args.replace("BUILD", "$(BUILD_DIR)")
                args = args.replace("REPO_DIR", "$(XF_PROJ_ROOT)")
                args = args.replace("HOST_EXE", "$(EXE_FILE)")
                target.write("\t" + args)
    target.write("else\n")
    target.write(
        '\t$(ECHO) "Please copy the content of sd_card folder and data to an SD Card and run on the board"'
    )
    target.write("\nendif\n")
    if "targets" in data:
        target.write("ifneq ($(TARGET),$(findstring $(TARGET),")
        args = data["targets"]
        for arg in args:
            target.write(" ")
            target.write(arg)
        target.write("))\n")
        target.write("$(error Application supports only")
        for arg in args:
            target.write(" ")
            target.write(arg)
        target.write(" TARGET. Please use the target for running the application)\n")
        target.write("endif\n")

    target.write("\n\n")

    target.write(".PHONY: test\n")
    target.write("test: $(EXECUTABLE)\n")
    target.write("ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))\n")
    target.write("ifeq ($(EMU_PS), X86)\n")
    target.write("\tXCL_EMULATION_MODE=$(TARGET) $(EXECUTABLE) $(CMD_ARGS)\n")
    target.write("else\n")
    target.write(
        "\tbash -c '$(LAUNCH_EMULATOR) -run-app $(RUN_APP_SCRIPT) | tee run_app.log; exit $${PIPESTATUS[0]}'\n"
    )
    target.write("endif\n")

    if "post_launch" in data:
        for post_launch in data["post_launch"]:
            if "launch_cmd" in post_launch:
                target.write("\n")
                args = post_launch["launch_cmd"]
                args = args.replace("BUILD", "$(BUILD_DIR)")
                args = args.replace("REPO_DIR", "$(XF_PROJ_ROOT)")
                args = args.replace("HOST_EXE", "$(EXE_FILE)")
                target.write("\t" + args)
    target.write("else\n")
    target.write(
        '\t$(ECHO) "Please copy the content of sd_card folder and data to an SD Card and run on the board"'
    )
    target.write("\nendif\n")
    if "targets" in data:
        target.write("ifneq ($(TARGET),$(findstring $(TARGET),")
        args = data["targets"]
        for arg in args:
            target.write(" ")
            target.write(arg)
        target.write("))\n")
        target.write("$(warning WARNING:Application supports only")
        for arg in args:
            target.write(" ")
            target.write(arg)
        target.write(" TARGET. Please use the target for running the application)\n")
        target.write("endif\n")
        target.write("\n")
    target.write("\n")
    target.write("check_edge_sw:\n")
    target.write("ifndef EDGE_COMMON_SW\n")
    target.write(
        "\t$(error EDGE_COMMON_SW variable is not set, please download and use the pre-built image from https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/embedded-platforms.html)\n"
    )
    target.write("endif\n\n")


def mk_sdcard(target, data):
    target.write(
        "############################## Preparing sdcard ##############################\n"
    )
    target.write(".PHONY: sd_card\n")
    target.write("sd_card:")
    target.write(" gen_run_app emconfig $(SD_CARD)\n")
    target.write("\n")

    target.write("$(SD_CARD):")
    target.write(" $(EXECUTABLE) $(LINK_OUTPUT)\n")
    extra_file_list = []
    if "launch" in data:
        if "cmd_args" in data["launch"][0]:
            args = data["launch"][0]["cmd_args"].split(" ")
            for arg in args:
                if "xclbin" not in arg:
                    arg = arg.replace("BUILD", "$(BUILD_DIR)")
                    arg = arg.replace("REPO_DIR", "$(XF_PROJ_ROOT)")
                    arg = arg.replace("PROJECT", ".")
                    extra_file_list.append(arg)
    if "containers" in data:
        for con in data["containers"]:
            target.write(
                "\tv++ -p $(VPP_PFLAGS) $(LINK_OUTPUT) $(VPP_FLAGS) -t $(TARGET) --platform $(PLATFORM) "
            )
            target.write(
                "--package.out_dir $(PACKAGE_OUT) --package.rootfs $(EDGE_COMMON_SW)/rootfs.ext4 --package.sd_file $(SD_IMAGE_FILE) --package.sd_file xrt.ini --package.sd_file $(RUN_APP_SCRIPT) --package.sd_file $(EXECUTABLE) --package.sd_file $(EMCONFIG_DIR)/emconfig.json"
            )
            for extra_filename in extra_file_list:
                if "-" not in extra_filename:
                    target.write(" --package.sd_file ")
                    target.write(extra_filename)
            target.write(" -o $(BUILD_DIR)/")
            target.write(con["name"])
            target.write(".xclbin\n")
    target.write("\n")


def mk_help(target):
    target.write(
        "\n############################## Help Section ##############################\n"
    )
    target.write("ifneq ($(findstring Makefile, $(MAKEFILE_LIST)), Makefile)\n")
    target.write("help:\n")
    target.write('\t$(ECHO) "Makefile Usage:"\n')
    target.write(
        '\t$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform> EDGE_COMMON_SW=<rootfs and kernel image path>."\n'
    )
    target.write(
        '\t$(ECHO) "      Command to generate the design for specified Target and Shell."\n'
    )
    target.write('\t$(ECHO) ""\n')
    target.write(
        '\t$(ECHO) "  make run TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform> EMU_PS=<X86/QEMU>'
    )
    target.write(" EDGE_COMMON_SW=<rootfs and kernel image path>")
    target.write('"\n')
    target.write(
        '\t$(ECHO) "      Command to run application in emulation..Default sw_emu will run on x86 ,to launch on qemu specify EMU_PS=QEMU."\n'
    )
    target.write('\t$(ECHO) ""\n')
    target.write(
        '\t$(ECHO) "  make build TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>'
    )
    target.write(" EDGE_COMMON_SW=<rootfs and kernel image path>")
    target.write('"\n')
    target.write('\t$(ECHO) "      Command to build xclbin application."\n')
    target.write('\t$(ECHO) ""\n')
    target.write('\t$(ECHO) "  make host PLATFORM=<FPGA platform>')
    target.write(" EDGE_COMMON_SW=<rootfs and kernel image path>")
    target.write('"\n')
    target.write('\t$(ECHO) "      Command to build host application."\n')
    target.write(
        '\t$(ECHO) "      EDGE_COMMON_SW is required for SoC shells. Please download and use the pre-built image from - "\n'
    )
    target.write(
        '\t$(ECHO) "      https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/embedded-platforms.html"\n'
    )
    target.write('\t$(ECHO) ""\n')
    target.write(
        '\t$(ECHO) "  make sd_card TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform> EDGE_COMMON_SW=<rootfs and kernel image path>"\n'
    )
    target.write('\t$(ECHO) "      Command to prepare sd_card files."\n')
    target.write('\t$(ECHO) ""\n')
    target.write('\t$(ECHO) "  make clean "\n')
    target.write(
        '\t$(ECHO) "      Command to remove the generated non-hardware files."\n'
    )
    target.write('\t$(ECHO) ""\n')
    target.write('\t$(ECHO) "  make cleanall"\n')
    target.write('\t$(ECHO) "      Command to remove all the generated files."\n')
    target.write('\t$(ECHO) ""\n')

    target.write("endif\n")
    target.write("\n")


def create_mk(target, data):
    mk_copyright(target)
    mk_help(target)
    create_params(target, data)
    add_host_flags(target, data)
    add_kernel_flags(target, data)
    mk_build_all(target, data)
    mk_run(target, data)
    mk_clean(target, data)
    return


def generate_makefile(desc_file, path):
    global data, init_cur_dir, cur_dir
    desc = open(desc_file, "r")
    data = json.load(desc)
    desc.close()

    file_name = "LICENSE"  # file to be searched
    cur_dir = os.getcwd()  # Dir from where search starts can be replaced with any path
    init_cur_dir = cur_dir

    while True:
        file_list = os.listdir(cur_dir)
        parent_dir = os.path.dirname(cur_dir)
        if file_name in file_list:
            break
        else:
            if cur_dir == parent_dir:  # if dir is root dir
                print("LICENSE file not found")
                break
            else:
                cur_dir = parent_dir

    if "match_makefile" in data and data["match_makefile"] == "false":
        print("Info:: Makefile Manually Edited:: AutoMakefile Generator Skipped")
    else:
        # print("Generating Auto-Makefile for %s" % data["name"])
        target = open(os.path.join(path, "makefile_zynqmp.mk"), "w")
        create_mk(target, data)

    if target:
        target.close
