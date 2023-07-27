# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modification: Slapo. See https://github.com/awslabs/slapo/blob/main/scripts/lint/add_license_header.py

"""Helper tool to add license header to files."""
import os
import sys
import subprocess

usage = """
Usage: python3 scripts/lint/add_license_header.py <file1 file2 ...|all>
Add license header to the target files. If file is "all", add license header
to all git-tracked files.
Examples:
 - python3 scripts/lint/add_license_header.py slapo/schedule.py slapo/pipeline.py
 - python3 scripts/lint/add_license_header.py all
"""

header_cstyle = """
/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
""".strip()

header_mdstyle = """
<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->
""".strip()

header_pystyle = """
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
""".strip()

header_rststyle = """
..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
""".strip()

header_groovystyle = """
// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
""".strip()

header_cmdstyle = """
:: Copyright Allo authors. All Rights Reserved.
:: SPDX-License-Identifier: Apache-2.0
""".strip()

FMT_MAP = {
    "sh": header_pystyle,
    "cc": header_cstyle,
    "c": header_cstyle,
    "cu": header_cstyle,
    "cuh": header_cstyle,
    "mm": header_cstyle,
    "m": header_cstyle,
    "go": header_cstyle,
    "java": header_cstyle,
    "h": header_cstyle,
    "py": header_pystyle,
    "toml": header_pystyle,
    "yml": header_pystyle,
    "yaml": header_pystyle,
    "rs": header_cstyle,
    "md": header_mdstyle,
    "cmake": header_pystyle,
    "mk": header_pystyle,
    "rst": header_rststyle,
    "gradle": header_groovystyle,
    "tcl": header_pystyle,
    "xml": header_mdstyle,
    "storyboard": header_mdstyle,
    "pbxproj": header_cstyle,
    "plist": header_mdstyle,
    "xcworkspacedata": header_mdstyle,
    "html": header_mdstyle,
    "bat": header_cmdstyle,
}


def get_file_fmt(file_path):
    """Get the file format, or None if the file format is not supported."""
    suffix = file_path.split(".")[-1]
    if suffix in FMT_MAP:
        return FMT_MAP[suffix]
    elif os.path.basename(file_path) == "gradle.properties":
        return FMT_MAP["h"]
    return None


def has_license_header(lines):
    """Check if the file has the license header."""
    copyright = False
    license = False
    for line in lines:
        if line.find("Allo authors.") != -1:
            copyright = True
        elif line.find("SPDX-License-Identifier") != -1:
            license = True
        if copyright and license:
            return True
    return False


def add_header(file_path, header):
    """Add header to file"""
    if not os.path.exists(file_path):
        print("%s does not exist" % file_path)
        return

    lines = open(file_path).readlines()
    if has_license_header(lines):
        print("%s has license header...skipped" % file_path)
        return False

    print("%s misses license header...added" % file_path)
    with open(file_path, "w") as outfile:
        # Insert the license at the second line if the first line has a special usage.
        insert_line_idx = 0
        if lines and lines[0][:2] in ["#!", "<?", "<html>", "// !$"]:
            insert_line_idx = 1

        # Write the pre-license lines.
        for idx in range(insert_line_idx):
            outfile.write(lines[idx])

        # Write the license.
        outfile.write(header + "\n\n")

        # Wright the rest of the lines.
        outfile.write("".join(lines[insert_line_idx:]))
    return True


def main(args):
    if len(sys.argv) == 1:
        sys.stderr.write(usage)
        sys.stderr.flush()
        sys.exit(-1)

    file_list = args[1:]
    if len(file_list) == 1 and file_list[0] == "all":
        cmd = ["git", "ls-tree", "--full-tree", "--name-only", "-r", "HEAD"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out, _) = proc.communicate()
        assert proc.returncode == 0, f'{" ".join(cmd)} errored: {out}'
        file_list = out.decode("utf-8").split("\n")

    stats = {"added": 0, "skipped": 0, "unsupported": 0}
    for file_path in file_list:
        if not file_path:
            continue
        fmt = get_file_fmt(file_path)
        if fmt is None:
            print("Unsupported file type: %s" % file_path)
            stats["unsupported"] += 1
        elif add_header(file_path, fmt):
            stats["added"] += 1
        else:
            stats["skipped"] += 1

    print(
        f"Added {stats['added']}\nSkipped {stats['skipped']}\n"
        f"Unsupported {stats['unsupported']}"
    )


if __name__ == "__main__":
    main(sys.argv)
