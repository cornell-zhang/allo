# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import xml.etree.ElementTree as ET

from .utils import format_str, format_code
from ..ir.transform import find_func_in_module
from ..utils import get_func_inputs_outputs
from ..utils import ctype_map

header = """
import argparse, time, pynq, os
import numpy as np

from pynq.buffer import PynqBuffer

def main():
    parser = argparse.ArgumentParser(description="Host Program for kernel")
    parser.add_argument("bitstream", type=str, help="Path to the bitstream file")

    args = parser.parse_args()
    bitstream_path:str = args.bitstream

    if not os.path.exists(bitstream_path):
        raise FileNotFoundError(f"Cannot find the bitstream file at {bitstream_path}")
    hwh_path = os.path.splitext(bitstream_path)[0] + ".hwh"
    if not os.path.exists(hwh_path):
        raise FileNotFoundError(f"Cannot find the hwh file at {hwh_path}")

    print(f"Programming hardware with bitstream {bitstream_path}")
    overlay = pynq.Overlay(bitstream_path)

"""

footer = """
    print(f"Hardware kernel finished in {hw_time_ms} ms")

if __name__ == "__main__":
    main()
"""
ctype_map_pynq = ctype_map


def pointer_args_parse(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {"spirit": "http://www.spiritconsortium.org/XMLSchema/SPIRIT/1685-2009"}
    reg_elems = root.findall(".//spirit:register", ns)

    base_names = []
    for reg in reg_elems:
        name = reg.find("spirit:name", ns).text
        if name.endswith("_1") or name.endswith("_2"):
            base = name.rsplit("_", 1)[0]
            if base not in base_names:
                base_names.append(base)
    return base_names


# for scalar args but not sure if we need this right now
def scalar_args_parse(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {"spirit": "http://www.spiritconsortium.org/XMLSchema/SPIRIT/1685-2009"}
    reg_elems = root.findall(".//spirit:register", ns)

    base_names = []
    for reg in reg_elems:
        name = reg.find("spirit:name", ns).text
        if not (
            name.endswith("_1")
            or name.endswith("_2")
            or name in {"CTRL", "GIER", "IP_IER", "IP_ISR"}
        ):
            if name not in base_names:
                base_names.append(name)
    return base_names


def codegen_pynq_host(top, module, project):
    out_str = header

    func = find_func_in_module(module, top)
    inputs, outputs = get_func_inputs_outputs(func)

    with format_code(indent=4):

        # top function overlay
        out_str += format_str(f"top_hw = overlay.{top}_0")
        out_str += format_str("top_hw.register_map.CTRL.AP_START = 0")
        out_str += "\n"

        all_args = inputs + outputs

        # extract pointer args TODO: need to replace hardcoded xml path
        ptr_names = pointer_args_parse(
            f"{project}/out.prj/solution1/impl/ip/component.xml"
        )

        # generate input/output buffers
        argidx = 0
        for name in ptr_names:
            dtype, shape = all_args[argidx]
            np_dtype = ctype_map_pynq[dtype]
            out_str += format_str(
                f"{name}: PynqBuffer = pynq.allocate({shape}, dtype = {np_dtype})"
            )
            argidx += 1

        out_str += "\n"

        # run accelerator
        # sync the buffers
        for i in range(len(inputs)):
            out_str += format_str(f"{ptr_names[i]}.sync_to_device()")

        out_str += "\n"
        # allocate buffer addr to registers
        for name in ptr_names:
            addr = f"{name}.physical_address"
            out_str += format_str(f"top_hw.register_map.{name}_1 = {addr}")
            out_str += format_str(
                f"top_hw.register_map.{name}_2 = ({addr} >> 32) & 0xFFFFFFFF"
            )

        out_str += format_str(
            """
                              
top_hw.register_map.CTRL.AP_START = 1
start_time = time.perf_counter()
while not top_hw.register_map.CTRL.AP_DONE:
    pass
end_time = time.perf_counter()
hw_time_ms = (end_time - start_time) * 1e3
        """,
            strip=False,
        )

        for i in range(len(inputs), len(ptr_names)):
            out_str += format_str(f"{ptr_names[i]}.sync_from_device()")

    out_str += format_str(footer, 0, strip=False)

    return out_str


def postprocess_hls_code_pynq(hls_code, top=None, pragma=True):
    out_str = ""
    func_decl = False
    has_endif = False
    extern_decl = False
    func_args = []
    for line in hls_code.split("\n"):
        if line == "using namespace std;" or line.startswith("#ifndef"):
            out_str += line + "\n"
            out_str += '\nextern "C" {\n\n'
            extern_decl = True
        elif line.startswith(f"void {top}"):
            func_decl = True
            if not extern_decl:
                out_str += '\nextern "C" {\n\n'
                extern_decl = True
            out_str += line + "\n"
        elif func_decl and line.startswith(") {"):
            func_decl = False
            out_str += line + "\n"
            # Generate pragmas for all arguments
            if pragma:
                gmem_idx = 0
                for arg, argtype in func_args:
                    if argtype == "pointer":
                        out_str += f"  #pragma HLS interface m_axi port={arg} offset=slave bundle=gmem{gmem_idx}\n"
                        out_str += f"  #pragma HLS interface s_axilite port={arg} bundle=control\n"
                        gmem_idx += 1
                    elif argtype == "scalar":
                        out_str += f"  #pragma HLS interface s_axilite port={arg} bundle=control\n"
        elif func_decl:
            if pragma:
                dtype, var = line.strip().rsplit(" ", 1)
                var_clean = var.rstrip(",")
                # Pointer/array detection
                is_pointer = ("[" in var_clean) or ("*" in dtype) or ("*" in var_clean)
                if is_pointer:
                    # Extract base variable name (e.g. v12 from v12[128]) to avoid
                    # concatenating array sizes into the identifier (v12128).
                    base = var_clean.split("[")[0]
                    arg_name = (
                        base.replace("*", "").replace("&", "").replace(",", "").strip()
                    )
                    out_str += f"  {dtype} {var_clean},\n"
                    func_args.append((arg_name, "pointer"))
                else:
                    arg_name = var_clean.replace("&", "").strip()
                    out_str += f"  {dtype} {var_clean},\n"
                    func_args.append((arg_name, "scalar"))
            else:
                out_str += line + "\n"
        elif line.startswith("#endif"):
            out_str += '} // extern "C"\n\n'
            out_str += line + "\n"
            has_endif = True
        else:
            out_str += line + "\n"
    # Remove the last comma in the argument list if present
    out_str = out_str.replace(",\n) {", "\n) {")
    if not has_endif:
        out_str += '} // extern "C"\n'
    return out_str


def generate_description_file(top, src_path, dst_path, frequency):
    with open(src_path, "r", encoding="utf-8") as f:
        desc = f.read()
    desc = desc.replace("top", top)
    desc = json.loads(desc)
    desc["containers"][0]["ldclflags"] += f"  --kernel_frequency {frequency}"
    with open(dst_path, "w", encoding="utf-8") as outfile:
        json.dump(desc, outfile, indent=4)
