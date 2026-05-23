#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACK = ROOT / "results" / "report_pack"
FIG_DIR = PACK / "figures"
TAB_DIR = PACK / "tables"


def read_csv(path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(row, key):
    return float(row[key])


def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)


def svg_header(width, height):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        'text { font-family: Menlo, Monaco, Consolas, monospace; fill: #202124; }',
        '.title { font-size: 22px; font-weight: 700; }',
        '.label { font-size: 14px; }',
        '.tick { font-size: 12px; fill: #5f6368; }',
        '.legend { font-size: 13px; }',
        '.grid { stroke: #e5e7eb; stroke-width: 1; }',
        '.axis { stroke: #374151; stroke-width: 1.5; }',
        '</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def write_svg(path, lines):
    path.write_text("\n".join(lines + ["</svg>"]), encoding="utf-8")


def nice_label(value):
    if abs(value) >= 1000:
        return f"{value / 1000:.1f}k"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"


def line_chart(path, title, x_label, y_label, series, colors, x_key, y_key, x_values):
    width = 960
    height = 540
    left = 90
    right = 40
    top = 70
    bottom = 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_y = []
    for points in series.values():
        for row in points:
            all_y.append(to_float(row, y_key))
    y_min = 0.0
    y_max = max(all_y) if all_y else 1.0
    y_max *= 1.12
    if y_max == 0:
        y_max = 1.0

    x_min = min(x_values)
    x_max = max(x_values)
    x_span = max(x_max - x_min, 1)

    def sx(x):
        return left + (x - x_min) / x_span * plot_w

    def sy(y):
        return top + plot_h - (y - y_min) / (y_max - y_min) * plot_h

    lines = svg_header(width, height)
    lines.append(f'<text x="{left}" y="36" class="title">{title}</text>')
    lines.append(f'<text x="{width / 2:.1f}" y="{height - 22}" text-anchor="middle" class="label">{x_label}</text>')
    lines.append(f'<text x="24" y="{top + plot_h / 2:.1f}" transform="rotate(-90 24 {top + plot_h / 2:.1f})" text-anchor="middle" class="label">{y_label}</text>')

    for i in range(6):
        value = y_min + (y_max - y_min) * i / 5
        y = sy(value)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{nice_label(value)}</text>')

    for x in x_values:
        px = sx(x)
        lines.append(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top + plot_h}" class="grid"/>')
        lines.append(f'<text x="{px:.1f}" y="{top + plot_h + 20}" text-anchor="middle" class="tick">{x}</text>')

    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')

    legend_x = left + plot_w - 150
    legend_y = top + 10
    legend_idx = 0

    for name, points in series.items():
        sorted_points = sorted(points, key=lambda row: int(float(row[x_key])))
        coords = " ".join(f"{sx(int(float(row[x_key]))):.1f},{sy(to_float(row, y_key)):.1f}" for row in sorted_points)
        color = colors[name]
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{coords}"/>')
        for row in sorted_points:
            x = sx(int(float(row[x_key])))
            y = sy(to_float(row, y_key))
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')
        ly = legend_y + legend_idx * 20
        lines.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 18}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x + 24}" y="{ly + 4}" class="legend">{name}</text>')
        legend_idx += 1

    write_svg(path, lines)


def bar_chart(path, title, y_label, rows, y_key, color_map):
    width = 1100
    height = 560
    left = 90
    right = 30
    top = 70
    bottom = 140
    plot_w = width - left - right
    plot_h = height - top - bottom

    y_max = max(to_float(row, y_key) for row in rows) * 1.15

    def sy(y):
        return top + plot_h - y / y_max * plot_h

    bar_w = plot_w / max(len(rows), 1) * 0.72
    gap = plot_w / max(len(rows), 1)

    lines = svg_header(width, height)
    lines.append(f'<text x="{left}" y="36" class="title">{title}</text>')
    lines.append(f'<text x="24" y="{top + plot_h / 2:.1f}" transform="rotate(-90 24 {top + plot_h / 2:.1f})" text-anchor="middle" class="label">{y_label}</text>')

    for i in range(6):
        value = y_max * i / 5
        y = sy(value)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{nice_label(value)}</text>')

    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')

    for idx, row in enumerate(rows):
        x = left + idx * gap + (gap - bar_w) / 2
        y = sy(to_float(row, y_key))
        h = top + plot_h - y
        label = f"{row['mode']} ({row['threads']})"
        color = color_map.get(row["mode"], "#4f46e5")
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" rx="4"/>')
        lines.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" class="tick">{to_float(row, y_key) / 1000:.2f}</text>')
        tick_x = x + bar_w / 2
        tick_y = top + plot_h + 16
        lines.append(f'<text x="{tick_x:.1f}" y="{tick_y:.1f}" transform="rotate(35 {tick_x:.1f} {tick_y:.1f})" text-anchor="start" class="tick">{label}</text>')

    write_svg(path, lines)


def group_by(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return dict(grouped)


def write_table_md(path, headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table_tex(path, headers, rows):
    def esc(text):
        return (
            str(text)
            .replace("\\", "\\textbackslash{}")
            .replace("_", "\\_")
            .replace("%", "\\%")
            .replace("&", "\\&")
            .replace("#", "\\#")
        )

    cols = "l" * len(headers)
    lines = [
        "\\begin{tabular}{" + cols + "}",
        "\\hline",
        " & ".join(esc(h) for h in headers) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(esc(cell) for cell in row) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_float(value, digits=3):
    return f"{float(value):.{digits}f}"


def build_tables(selected_rows, thread_rows, batch_rows, trade_rows, ref_rows):
    selected_map = {row["mode"]: row for row in selected_rows}
    seq_total = to_float(selected_map["seq"], "total_us")
    table_rows = []
    keep = ["seq", "pthread", "openmp", "ivf_pq", "ivf_pq_pthread", "hnsw", "hnsw_pthread"]
    for mode in keep:
        row = selected_map[mode]
        speedup = seq_total / to_float(row, "total_us")
        table_rows.append([
            mode,
            row["threads"],
            format_float(row["recall"], 3),
            format_float(to_float(row, "latency_us"), 2),
            format_float(to_float(row, "total_us") / 1000.0, 3),
            format_float(speedup, 2),
        ])

    headers = ["mode", "threads", "recall", "latency_us", "total_ms", "speedup_vs_seq"]
    write_table_md(TAB_DIR / "selected_summary.md", headers, table_rows)
    write_table_tex(TAB_DIR / "selected_summary.tex", headers, table_rows)

    by_variant = defaultdict(list)
    for row in ref_rows:
        by_variant[row["variant"]].append(row)
    variant_rows = []
    for variant, rows in by_variant.items():
        avg_latency = sum(to_float(row, "latency_us") for row in rows) / len(rows)
        avg_recall = sum(to_float(row, "recall") for row in rows) / len(rows)
        label = rows[0]["label"]
        variant_rows.append((variant, label, avg_recall, avg_latency))
    variant_rows.sort(key=lambda item: item[3], reverse=True)

    simd_rows = [
        [variant, label, format_float(recall, 5), format_float(latency, 2)]
        for variant, label, recall, latency in variant_rows
    ]
    write_table_md(TAB_DIR / "simd_reference_summary.md", ["variant", "label", "avg_recall", "avg_latency_us"], simd_rows)

    best_hnsw_thread = min(
        [row for row in thread_rows if row["mode"] == "hnsw_pthread"],
        key=lambda row: to_float(row, "total_us"),
    )
    best_pthread_thread = min(
        [row for row in thread_rows if row["mode"] == "pthread"],
        key=lambda row: to_float(row, "total_us"),
    )
    notes = [
        f"Best hnsw_pthread total_us in thread sweep: threads={best_hnsw_thread['threads']}, total_us={best_hnsw_thread['total_us']}.",
        f"Best pthread total_us in thread sweep: threads={best_pthread_thread['threads']}, total_us={best_pthread_thread['total_us']}.",
        f"Batch-scale rows: {len(batch_rows)}.",
        f"Tradeoff rows: {len(trade_rows)}.",
    ]
    (TAB_DIR / "quick_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")


def build_notes(selected_rows, thread_rows, batch_rows, trade_rows):
    selected = {row["mode"]: row for row in selected_rows}

    def speedup(a, b):
        return to_float(selected[a], "total_us") / to_float(selected[b], "total_us")

    thread_hnsw_best = min(
        [row for row in thread_rows if row["mode"] == "hnsw_pthread"],
        key=lambda row: to_float(row, "total_us"),
    )
    thread_hnsw_openmp_best = min(
        [row for row in thread_rows if row["mode"] == "hnsw_openmp"],
        key=lambda row: to_float(row, "total_us"),
    )

    pq_trade = {int(row["param_value"]): row for row in trade_rows if row["family"] == "flat_pq"}
    hnsw_trade = {int(row["param_value"]): row for row in trade_rows if row["family"] == "hnsw"}

    lines = [
        "# Writeup Notes",
        "",
        "## Core Findings",
        "",
        f"- `pthread` 8-thread total time is `{selected['pthread']['total_us']} us`, about `{speedup('seq', 'pthread'):.2f}x` faster than `seq` for the same query batch.",
        f"- `openmp` 8-thread total time is `{selected['openmp']['total_us']} us`, about `{speedup('seq', 'openmp'):.2f}x` faster than `seq`.",
        f"- `ivf_pq_pthread` 8-thread total time is `{selected['ivf_pq_pthread']['total_us']} us`, about `{to_float(selected['ivf_pq'], 'total_us') / to_float(selected['ivf_pq_pthread'], 'total_us'):.2f}x` faster than serial `ivf_pq`.",
        f"- `hnsw_pthread` 8-thread total time is `{selected['hnsw_pthread']['total_us']} us`, about `{to_float(selected['hnsw'], 'total_us') / to_float(selected['hnsw_pthread'], 'total_us'):.2f}x` faster than serial `hnsw`.",
        "",
        "## Thread Sweep",
        "",
        f"- In the 10-query thread sweep, `hnsw_pthread` reaches its best total time at `{thread_hnsw_best['threads']}` threads with `{thread_hnsw_best['total_us']} us`.",
        f"- In the same sweep, `hnsw_openmp` reaches its best total time at `{thread_hnsw_openmp_best['threads']}` threads with `{thread_hnsw_openmp_best['total_us']} us`.",
        "- For Flat query-level parallelism, throughput improves with more threads, but average per-query latency stays in roughly the same range instead of decreasing proportionally.",
        "- `OpenMP` shows clear speedup too, but the curve is less monotonic than `pthread`, which is suitable for discussing scheduling and runtime overhead.",
        "",
        "## Tradeoff",
        "",
        f"- For Flat-PQ, increasing `p` from `200` to `550` raises recall from `{pq_trade[200]['recall']}` to `{pq_trade[550]['recall']}`, while total time rises from `{pq_trade[200]['total_us']} us` to `{pq_trade[550]['total_us']} us`.",
        f"- For HNSW, increasing `ef` from `32` to `128` raises recall from `{hnsw_trade[32]['recall']}` to `{hnsw_trade[128]['recall']}`, while total time rises from `{hnsw_trade[32]['total_us']} us` to `{hnsw_trade[128]['total_us']} us`.",
        "- For IVF/IVF-PQ, `nprobe` increases query cost significantly; IVF-PQ keeps lower latency than IVF-NEON at the tested settings.",
        "",
        "## Suggested Discussion Angles",
        "",
        "- Separate single-query latency and batch throughput when interpreting results.",
        "- Explain why query-level parallelism is easier to scale than fine-grained partition strategies in some modes.",
        "- Point out that graph-based HNSW is the strongest low-latency option in the current experiments, while IVF-PQ offers a strong throughput/accuracy compromise.",
    ]
    (PACK / "WRITEUP_NOTES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_figures(selected_rows, thread_rows, batch_rows, trade_rows, ref_rows):
    palette = {
        "seq": "#1f77b4",
        "pthread": "#d62728",
        "openmp": "#ff7f0e",
        "ivf_pq_pthread": "#2ca02c",
        "ivf_pq_openmp": "#86bf91",
        "hnsw_pthread": "#9467bd",
        "hnsw_openmp": "#c5b0d5",
        "ivf_pq": "#17becf",
        "hnsw": "#8c564b",
    }

    representative = [row for row in selected_rows if row["mode"] in {"seq", "pthread", "openmp", "ivf_pq_pthread", "hnsw_pthread"}]
    representative.sort(key=lambda row: to_float(row, "total_us"), reverse=True)
    bar_chart(
        FIG_DIR / "main_total_time.svg",
        "Representative Total Query Time",
        "total_us",
        representative,
        "total_us",
        palette,
    )

    thread_series = group_by(thread_rows, "mode")
    line_chart(
        FIG_DIR / "thread_sweep_total.svg",
        "Thread Sweep: Total Time",
        "threads",
        "total_us",
        thread_series,
        {
            "pthread": "#d62728",
            "openmp": "#ff7f0e",
            "ivf_pq_pthread": "#2ca02c",
            "ivf_pq_openmp": "#86bf91",
            "hnsw_pthread": "#9467bd",
            "hnsw_openmp": "#c5b0d5",
        },
        "threads",
        "total_us",
        [1, 2, 4, 8],
    )

    batch_series = group_by(batch_rows, "mode")
    line_chart(
        FIG_DIR / "batch_scale_total.svg",
        "Batch Scale: Total Time",
        "query_limit",
        "total_us",
        batch_series,
        {
            "pthread": "#d62728",
            "hnsw_pthread": "#9467bd",
            "ivf_pq_pthread": "#2ca02c",
        },
        "query_limit",
        "total_us",
        [5, 10, 20, 50],
    )

    trade_by_family = group_by(trade_rows, "family")

    for family, filename, label, xvals in [
        ("flat_pq", "tradeoff_flat_pq.svg", "ANN_PQ_RERANK", sorted({int(row["param_value"]) for row in trade_by_family["flat_pq"]})),
        ("hnsw", "tradeoff_hnsw.svg", "ANN_HNSW_EF", sorted({int(row["param_value"]) for row in trade_by_family["hnsw"]})),
    ]:
        family_rows = trade_by_family[family]
        series = {family_rows[0]["mode"]: family_rows}
        colors = {family_rows[0]["mode"]: "#d62728" if family == "flat_pq" else "#9467bd"}
        line_chart(
            FIG_DIR / filename,
            f"{family} Tradeoff: Recall",
            label,
            "recall",
            series,
            colors,
            "param_value",
            "recall",
            xvals,
        )

    ivf_rows = [row for row in trade_rows if row["family"] in {"ivf", "ivf_pq"}]
    line_chart(
        FIG_DIR / "tradeoff_ivf_total.svg",
        "IVF / IVF-PQ Tradeoff: Total Time",
        "ANN_IVF_NPROBE",
        "total_us",
        group_by(ivf_rows, "mode"),
        {
            "ivf_neon": "#1f77b4",
            "ivf_pq": "#2ca02c",
        },
        "param_value",
        "total_us",
        sorted({int(row["param_value"]) for row in ivf_rows}),
    )

    by_variant = defaultdict(list)
    for row in ref_rows:
        by_variant[row["variant"]].append(row)
    ref_summary = []
    for variant, rows in by_variant.items():
        ref_summary.append({
            "mode": variant,
            "threads": "SIMD",
            "total_us": str(sum(to_float(row, "latency_us") for row in rows) / len(rows)),
        })
    keep_variants = {"SERIAL", "NEON_4P", "NEON_16P"}
    ref_summary = [row for row in ref_summary if row["mode"] in keep_variants]
    ref_summary.sort(key=lambda row: to_float(row, "total_us"), reverse=True)
    bar_chart(
        FIG_DIR / "simd_reference_latency.svg",
        "SIMD Reference Latency",
        "latency_us",
        ref_summary,
        "total_us",
        {
            "SERIAL": "#1f77b4",
            "NEON_4P": "#2ca02c",
            "NEON_16P": "#9467bd",
        },
    )


def main():
    ensure_dirs()
    selected_rows = read_csv(PACK / "ann2_selected.csv")
    thread_rows = read_csv(PACK / "thread_sweep.csv")
    batch_rows = read_csv(PACK / "batch_scale.csv")
    trade_rows = read_csv(PACK / "tradeoff_sweep.csv")
    ref_rows = read_csv(PACK / "reference" / "strategy_results.csv")
    build_figures(selected_rows, thread_rows, batch_rows, trade_rows, ref_rows)
    build_tables(selected_rows, thread_rows, batch_rows, trade_rows, ref_rows)
    build_notes(selected_rows, thread_rows, batch_rows, trade_rows)


if __name__ == "__main__":
    main()
