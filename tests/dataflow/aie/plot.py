#!/usr/bin/env python3
import re
import sys
import matplotlib.pyplot as plt


def parse_log():
    text = """
NPU execution time: 3149us
NPU execution time: 281us
NPU execution time: 273us
NPU execution time: 210us
NPU execution time: 208us
NPU execution time: 183us
NPU execution time: 142us
NPU execution time: 132us
NPU execution time: 126us
NPU execution time: 113us
NPU execution time: 2678us
NPU execution time: 159us
NPU execution time: 208us
NPU execution time: 137us
NPU execution time: 190us
NPU execution time: 128us
NPU execution time: 186us
NPU execution time: 115us
NPU execution time: 114us
NPU execution time: 114us
NPU execution time: 136us
NPU execution time: 129us
NPU execution time: 114us
NPU execution time: 110us
NPU execution time: 109us
NPU execution time: 113us
NPU execution time: 109us
NPU execution time: 109us
NPU execution time: 109us
NPU execution time: 108us
"""
    times = [int(x) for x in re.findall(r"NPU execution time: (\d+)us", text)]
    return times


def plot_times(times):
    plt.figure(figsize=(8, 4))
    plt.plot(times, marker="o", linestyle="-", label="NPU exec time (us)")
    plt.title("NPU Execution Time")
    plt.xlabel("Run index")
    plt.ylabel("Execution time (us)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot.png")


if __name__ == "__main__":
    times = parse_log()

    if not times:
        print("No NPU execution times found in the log.")
        sys.exit(1)

    print(f"Parsed {len(times)} execution times")
    plot_times(times)
