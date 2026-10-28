"""Render the original TA performance CSV plot without import-time I/O."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", default=".log/run_cmp_performance_${RUN_ID}/comparison_report_${RUN_ID}.csv")
    parser.add_argument("-o", "--output", default="time_tl-ascend_vs_ascendc.png")
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import pandas as pd

    file_path = args.csv_path

    df = pd.read_csv(file_path)

    df["TL_Time(ms)"] = df["TL_Time(us)"] / 1000.0
    df["AC_Time(ms)"] = df["AC_Time(us)"] / 1000.0

    x = df["S"]  # seqlen
    plt.plot(x, df["TL_Time(ms)"], marker="o", label="TL-Ascend")
    plt.plot(x, df["AC_Time(ms)"], marker="o", label="AscendC")

    plt.title("TL-Ascend vs AscendC")
    plt.xlabel("Seqlen (S)")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
