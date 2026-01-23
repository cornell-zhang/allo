# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modification: Slapo. See https://github.com/awslabs/slapo/blob/main/scripts/lint/check_license_header.py

"""Helper tool to check license header."""
import os
import sys
import subprocess

from add_license_header import get_file_fmt, has_license_header

usage = """
Usage: python3 scripts/lint/check_license_header.py <commit|"all">
Run license header check that changed since <commit>. If <commit> is "all",
check all files.
Examples:
 - Compare last one commit: python3 scripts/lint/check_license_header.py HEAD~1
 - Compare against origin: python3 scripts/lint/check_license_header.py origin/main
 - Check all files: python3 scripts/lint/check_license_header.py all
"""


def check_license(fname):
    # Skip 3rdparty change and unsupported file format.
    if not os.path.isfile(fname) or get_file_fmt(fname) is None:
        return True

    lines = open(fname).readlines()
    return has_license_header(lines)


def main():
    if len(sys.argv) != 2:
        sys.stderr.write(usage)
        sys.stderr.flush()
        sys.exit(-1)

    commit = sys.argv[1]
    if commit == "all":
        cmd = ["git", "ls-tree", "--full-tree", "--name-only", "-r", "HEAD"]
    else:
        cmd = ["git", "diff", "--name-only", "--diff-filter=ACMRTUX", commit]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    assert proc.returncode == 0, f'{" ".join(cmd)} errored: {out}'
    res = out.decode("utf-8")

    error_list = []
    for fname in res.split():
        if fname.startswith(".github/ISSUE_TEMPLATE") or fname == "AGENTS.md":
            continue
        if not check_license(fname):
            error_list.append(fname)

    if error_list:
        report = "-----Check report-----\n"
        report += "\n".join(error_list) + "\n"
        report += (
            "-----Found %d files that cannot pass the license header check-----\n"
            % len(error_list)
        )
        sys.stderr.write(report)
        sys.stderr.flush()
        sys.exit(-1)

    print("all checks passed...")


if __name__ == "__main__":
    main()
