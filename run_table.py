#!/usr/bin/env python3
"""
Compile balance.c, run the experiment suite, and display results as a
formatted table matching the assignment report layout.
"""
import re
import subprocess
import sys
import os

BINARY = "balance_run.exe"
SOURCE = "balance.c"

# MSYS2 ucrt64 gcc conflicts with Git's mingw64 DLLs when Git bin is in PATH.
# Pass a clean PATH so only MSYS2 directories are searched.
MSYS2_CLEAN_ENV = {
    **os.environ,
    "PATH": r"C:\msys64\ucrt64\bin;C:\msys64\usr\bin;" + os.environ.get("PATH", ""),
}

GCC_CANDIDATES = [
    r"C:\msys64\ucrt64\bin\gcc.exe",
    r"C:\msys64\mingw64\bin\gcc.exe",
    "gcc",
]

def find_gcc():
    for candidate in GCC_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    import shutil
    g = shutil.which("gcc")
    if g:
        return g
    print("Could not find gcc. Install MSYS2 (pacman -S mingw-w64-ucrt-x86_64-gcc) or add gcc to PATH.")
    sys.exit(1)

def compile(script_dir):
    gcc = find_gcc()
    src = os.path.join(script_dir, SOURCE)
    out = os.path.join(script_dir, BINARY)
    cmd = [gcc, "-O2", "-o", out, src]
    result = subprocess.run(cmd, capture_output=True, text=True, env=MSYS2_CLEAN_ENV)
    if result.returncode != 0:
        print("Compilation failed:\n" + result.stderr)
        sys.exit(1)
    return out

def run_binary(exe_path):
    result = subprocess.run([exe_path], capture_output=True, text=True, env=MSYS2_CLEAN_ENV)
    if result.returncode != 0:
        print("Run failed:\n" + result.stderr)
        sys.exit(1)
    return result.stdout

# Matches lines like:
#   STRICT  k=5    | steady:20/20  balanced:12/20  avg_cycles=    25002  avg_spread=  1.65  worst_spread=3
ROW_RE = re.compile(
    r"(?P<strategy>STRICT|VARIANT)\s+k=(?P<k>\d+)\s+\|"
    r"\s+steady:\s*(?P<sc>\d+)/(?P<st>\d+)"
    r"\s+balanced:\s*(?P<bc>\d+)/(?P<bt>\d+)"
    r"\s+avg_cycles=\s*(?P<avg_cycles>[\d.]+)"
    r"\s+avg_spread=\s*(?P<avg_spread>[\d.]+)"
    r"\s+worst_spread=(?P<worst_spread>\d+)"
)

def parse(output):
    rows = []
    for line in output.splitlines():
        m = ROW_RE.search(line)
        if m:
            rows.append({
                "strategy":         m.group("strategy"),
                "k":                int(m.group("k")),
                "reached_steady":   f"{m.group('sc')}/{m.group('st')}",
                "reached_balanced": f"{m.group('bc')}/{m.group('bt')}",
                "avg_cycles":       float(m.group("avg_cycles")),
                "avg_spread":       float(m.group("avg_spread")),
                "worst_spread":     int(m.group("worst_spread")),
            })
    return rows


def export_png(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    headers = ["Strategy", "k", "Reached\nsteady", "Reached\nbalanced",
               "Avg cycles\nto steady", "Avg final\nspread", "Worst\nspread"]

    table_data = []
    for r in rows:
        table_data.append([
            r["strategy"],
            str(r["k"]),
            r["reached_steady"],
            r["reached_balanced"],
            f"{r['avg_cycles']:,.0f}",
            f"{r['avg_spread']:.2f}",
            str(r["worst_spread"]),
        ])

    n_rows = len(table_data)
    n_cols = len(headers)

    fig_w = 10
    row_h = 0.45
    header_h = 0.55
    fig_h = header_h + n_rows * row_h + 0.3

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    col_widths = [0.10, 0.06, 0.12, 0.13, 0.17, 0.14, 0.12]

    tbl = ax.table(
        cellText=table_data,
        colLabels=headers,
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    STRICT_BG  = "#fde8e8"
    VARIANT_BG = "#e8f0fe"
    HEADER_BG  = "#2c3e50"

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#aaaaaa")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_height(header_h / fig_h)
        else:
            strategy = table_data[row - 1][0]
            cell.set_facecolor(STRICT_BG if strategy == "STRICT" else VARIANT_BG)
            cell.set_height(row_h / fig_h)

    legend_patches = [
        mpatches.Patch(facecolor=STRICT_BG,  edgecolor="#aaaaaa", label="STRICT strategy"),
        mpatches.Patch(facecolor=VARIANT_BG, edgecolor="#aaaaaa", label="VARIANT strategy"),
    ]
    ax.legend(handles=legend_patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=9,
              framealpha=0.9)

    ax.set_title("CS231P HW8 — Ring Load-Balancing Results",
                 fontsize=12, fontweight="bold", pad=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Table saved to: {out_path}")

def plot_convergence(exe_path, script_dir, k=20, seed=12345,
                     Lmin=10, Lmax=1000, Dmin=100, Dmax=1000):
    """Figure 1: spread vs time for STRICT and VARIANT with the same seed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def get_series(strategy_int):
        cmd = [exe_path, str(k), str(Lmin), str(Lmax),
               str(Dmin), str(Dmax), str(seed), str(strategy_int), "timeseries"]
        result = subprocess.run(cmd, capture_output=True, text=True, env=MSYS2_CLEAN_ENV)
        times, spreads = [], []
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) == 2:
                times.append(int(parts[0]) / 1000)
                spreads.append(int(parts[1]))
        return times, spreads

    t_strict,  s_strict  = get_series(0)
    t_variant, s_variant = get_series(1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t_strict,  s_strict,  color="#c0392b", label="STRICT",  linewidth=1.2)
    ax.plot(t_variant, s_variant, color="#2980b9", label="VARIANT", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_xlabel("time-interval (x1000)")
    ax.set_ylabel("spread (max - min load)")
    ax.set_title(f"Convergence of load spread, k = {k}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(script_dir, "balance_convergence.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Convergence plot saved to: {out}")


def plot_imbalance(rows, script_dir):
    """Figure 2: grouped bar chart of avg final spread vs k."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    strict_rows  = [r for r in rows if r["strategy"] == "STRICT"]
    variant_rows = [r for r in rows if r["strategy"] == "VARIANT"]

    ks              = [r["k"]          for r in strict_rows]
    strict_spreads  = [r["avg_spread"] for r in strict_rows]
    variant_spreads = [r["avg_spread"] for r in variant_rows]

    x     = np.arange(len(ks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, strict_spreads,  width, label="STRICT",  color="#c0392b")
    ax.bar(x + width / 2, variant_spreads, width, label="VARIANT", color="#2980b9")
    ax.axhline(1, color="black", linewidth=1.2, linestyle="--",
               label="balanced threshold (≤1)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel("avg final spread")
    ax.set_title("Final imbalance vs number of processors")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    out = os.path.join(script_dir, "balance_imbalance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Imbalance plot saved to: {out}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"Compiling {SOURCE}...")
    exe = compile(script_dir)
    print("Running experiments (this may take a minute)...\n")
    output = run_binary(exe)
    rows = parse(output)
    if not rows:
        print("No data parsed. Raw output:\n")
        print(output)
        sys.exit(1)

    png_path = os.path.join(script_dir, "balance_results.png")
    export_png(rows, png_path)

    plot_convergence(exe, script_dir)
    plot_imbalance(rows, script_dir)

if __name__ == "__main__":
    main()
