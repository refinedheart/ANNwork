#!/usr/bin/env python3
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACK = ROOT / "results" / "report_pack"
X86_DIR = ROOT.parent / "results-x86-2"


def read_csv(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def f(v):
    return float(v)


def write_md(path, headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex(path, headers, rows):
    def esc(s):
        return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")
    lines = [
        "\\begin{tabular}{" + "l" * len(headers) + "}",
        "\\hline",
        " & ".join(esc(h) for h in headers) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(esc(cell) for cell in row) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    arm_q20 = {row["mode"]: row for row in read_csv(PACK / "ann2_selected.csv")}
    arm_q10 = {row["mode"]: row for row in read_csv(PACK / "thread_sweep.csv")}
    arm_q200 = {row["mode"]: row for row in read_csv(PACK / "batch_scale.csv")}

    x86_q20 = {row["mode"]: row for row in read_csv(X86_DIR / "cross_platform_x86_q20_fixed.csv")}
    x86_q10 = {row["mode"]: row for row in read_csv(X86_DIR / "cross_platform_x86_q10_fixed.csv")}
    x86_q200 = {row["mode"]: row for row in read_csv(X86_DIR / "cross_platform_x86_q200_fixed.csv")}

    out_dir = PACK / "cross_platform"
    out_dir.mkdir(parents=True, exist_ok=True)

    q20_rows = []
    mappings_q20 = [
        ("ivf_pq", "ivf_pq"),
        ("ivf_pq_pthread", "thread_ivf_pq"),
        ("ivf_pq_openmp", "openmp_ivf_pq"),
        ("hnsw", "hnsw"),
        ("hnsw_pthread", "thread_hnsw"),
        ("hnsw_openmp", "openmp_hnsw"),
    ]
    for arm_mode, x86_mode in mappings_q20:
        a = arm_q20[arm_mode]
        x = x86_q20[x86_mode]
        q20_rows.append([
            arm_mode,
            a["threads"],
            a["recall"],
            f"{f(a['total_us'])/1000.0:.3f}",
            x86_mode,
            x["threads"],
            x["avg_recall"],
            f"{f(x['total_time_us'])/1000.0:.3f}",
        ])

    headers_q20 = [
        "arm_mode", "arm_threads", "arm_recall", "arm_total_ms",
        "x86_mode", "x86_threads", "x86_recall", "x86_total_ms"
    ]
    write_md(out_dir / "cross_platform_q20.md", headers_q20, q20_rows)
    write_tex(out_dir / "cross_platform_q20.tex", headers_q20, q20_rows)

    q10_rows = []
    mappings_q10 = [
        ("ivf_pq_pthread", "thread_ivf_pq"),
        ("ivf_pq_openmp", "openmp_ivf_pq"),
        ("hnsw_pthread", "thread_hnsw"),
        ("hnsw_openmp", "openmp_hnsw"),
    ]
    for arm_mode, x86_mode in mappings_q10:
        a_candidates = [row for row in read_csv(PACK / "thread_sweep.csv") if row["mode"] == arm_mode and row["threads"] == "8"]
        x = x86_q10[x86_mode]
        if not a_candidates:
            continue
        a = a_candidates[0]
        q10_rows.append([
            arm_mode,
            a["threads"],
            a["recall"],
            a["total_us"],
            x86_mode,
            x["threads"],
            x["avg_recall"],
            x["total_time_us"],
        ])
    headers_q10 = [
        "arm_mode", "arm_threads", "arm_recall", "arm_total_us",
        "x86_mode", "x86_threads", "x86_recall", "x86_total_us"
    ]
    write_md(out_dir / "cross_platform_q10_threads.md", headers_q10, q10_rows)

    q200_rows = []
    mappings_q200 = [
        ("ivf_pq_pthread", "thread_ivf_pq"),
        ("hnsw_pthread", "thread_hnsw"),
    ]
    for arm_mode, x86_mode in mappings_q200:
        x = x86_q200[x86_mode]
        q200_rows.append([
            arm_mode,
            x86_mode,
            x["avg_recall"],
            x["avg_latency_us"],
            x["total_time_us"],
        ])
    write_md(
        out_dir / "x86_q200_summary.md",
        ["arm_reference_mode", "x86_mode", "x86_recall", "x86_latency_us", "x86_total_us"],
        q200_rows
    )

    notes = [
        "# Cross-Platform Writeup Draft",
        "",
        "## Safe Conclusions",
        "",
        "- HNSW remains the lowest-latency family on both ARM and x86.",
        "- IVF-PQ remains a strong recall/throughput compromise on both ARM and x86.",
        "- On x86 q20, serial `hnsw` total time is about 1.187 ms, compared with ARM q20 `hnsw` total time about 4.847 ms.",
        "- On x86 q20, serial `ivf_pq` total time is about 10.715 ms, compared with ARM q20 `ivf_pq` total time about 12.828 ms.",
        "- On x86 q20, `thread_ivf_pq` and `openmp_ivf_pq` reduce total time to about 2.719 ms and 3.166 ms.",
        "- On x86 q20, `thread_hnsw` and `openmp_hnsw` stay close to serial `hnsw`, indicating limited additional throughput gain at small query batch size.",
        "",
        "## Caution",
        "",
        "- The current ARM Flat baseline in ANN2 is PQ-based approximate search, while x86 `avx16/thread_flat/openmp_flat` are exact flat search. These should not be directly compared as the same line in the final report.",
    ]
    (out_dir / "cross_platform_writeup.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
