#!/usr/bin/env python3
import csv
import html
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


RESULT_DIR = Path(__file__).resolve().parent
TABLE_DIR = RESULT_DIR / "tables"
FIGURE_DIR = RESULT_DIR / "figures"


INT_FIELDS = {
    "case_order",
    "nranks",
    "n1",
    "n2",
    "n3",
    "m",
    "nsys",
    "nrow_min",
    "nrow_max",
    "iter",
    "iterations",
}

FLOAT_FIELDS = {
    "total_s_max",
    "total_s_avg",
    "local_compute_s_max",
    "forward_exchange_s_max",
    "reduced_compute_s_max",
    "backward_exchange_s_max",
    "update_compute_s_max",
    "compute_s_max",
    "communication_s_max",
    "solution_sum",
    "solution_l2",
    "solution_linf",
    "sample_z0",
    "sample_zmid",
    "sample_zlast",
}

COLORS = {
    "blue": "#2563eb",
    "orange": "#ea580c",
    "green": "#16a34a",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "teal": "#0f766e",
    "gray": "#64748b",
    "yellow": "#ca8a04",
}


def latest(pattern):
    matches = sorted(RESULT_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"missing {pattern} under {RESULT_DIR}")
    return matches[-1]


def as_int(value):
    return int(float(value))


def load_csv(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            if value == "":
                continue
            try:
                if key in INT_FIELDS:
                    row[key] = as_int(value)
                elif key in FLOAT_FIELDS:
                    row[key] = float(value)
            except ValueError:
                pass
        if "study_tags" in row:
            row["tags"] = set(tag for tag in row["study_tags"].split(";") if tag)
    return rows


def load_environment(path):
    data = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if "=" in line and not line.startswith(" "):
            key, value = line.split("=", 1)
            if re.match(r"^[A-Za-z0-9_]+$", key):
                data[key] = value.strip()
    gpu_count = len(re.findall(r"NVIDIA H200", text))
    if gpu_count:
        data["gpu_count"] = str(gpu_count)
        data["gpu_name"] = "NVIDIA H200"
    driver_match = re.search(r"Driver Version:\s*([0-9.]+)", text)
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", text)
    nvcc_match = re.search(r"release\s+([0-9.]+),\s+V([0-9.]+)", text)
    mpi_match = re.search(r"mpirun \(Open MPI\)\s+([^\s]+)", text)
    if driver_match:
        data["driver_version"] = driver_match.group(1)
    if cuda_match:
        data["driver_cuda_version"] = cuda_match.group(1)
    if nvcc_match:
        data["nvcc_release"] = nvcc_match.group(1)
        data["nvcc_version"] = nvcc_match.group(2)
    if mpi_match:
        data["openmpi_version"] = mpi_match.group(1)
    return data


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def stdev(values):
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def safe_div(a, b):
    if b == 0 or math.isnan(b):
        return float("nan")
    return a / b


def fmt_float(value, digits=3):
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_sci(value, digits=3):
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.{digits}e}"


def fmt_ms(value):
    return fmt_float(value * 1000.0, 3)


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines) + "\n"


def write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_csv(path, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def esc(text):
    return html.escape(str(text), quote=True)


def svg_header(width, height):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}",
        ".title{font-size:18px;font-weight:700}",
        ".axis{stroke:#52606d;stroke-width:1}",
        ".grid{stroke:#d9e2ec;stroke-width:1}",
        ".tick{font-size:11px;fill:#52606d}",
        ".label{font-size:12px;fill:#334e68}",
        ".point-label{font-size:11px;fill:#243b53;font-weight:600}",
        ".legend{font-size:12px;fill:#243b53}",
        "</style>",
    ]


def svg_footer():
    return ["</svg>"]


def nice_ticks(ymax, count=5):
    if ymax <= 0 or math.isnan(ymax):
        return [0, 1]
    raw = ymax / count
    exponent = math.floor(math.log10(raw))
    fraction = raw / (10 ** exponent)
    if fraction <= 1:
        step = 1 * 10 ** exponent
    elif fraction <= 2:
        step = 2 * 10 ** exponent
    elif fraction <= 5:
        step = 5 * 10 ** exponent
    else:
        step = 10 * 10 ** exponent
    top = math.ceil(ymax / step) * step
    ticks = []
    current = 0.0
    while current <= top + step * 0.5:
        ticks.append(current)
        current += step
    return ticks


def draw_axes(lines, width, height, left, right, top, bottom, y_ticks, y_max, title, xlabel, ylabel):
    plot_w = width - left - right
    plot_h = height - top - bottom
    lines.append(f'<text x="{width / 2}" y="28" text-anchor="middle" class="title">{esc(title)}</text>')
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    for tick in y_ticks:
        y = top + plot_h - safe_div(tick, y_max) * plot_h if y_max else top + plot_h
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">{fmt_float(tick, 2)}</text>')
    lines.append(f'<text x="{left + plot_w / 2}" y="{height - 12}" text-anchor="middle" class="label">{esc(xlabel)}</text>')
    lines.append(f'<text x="16" y="{top + plot_h / 2}" text-anchor="middle" class="label" transform="rotate(-90 16 {top + plot_h / 2})">{esc(ylabel)}</text>')


def log_ticks(vmin, vmax):
    if vmin <= 0 or vmax <= 0:
        return []
    candidates = []
    start = math.floor(math.log10(vmin))
    end = math.ceil(math.log10(vmax))
    for exponent in range(start, end + 1):
        for multiplier in (1, 2, 5):
            value = multiplier * (10 ** exponent)
            if vmin <= value <= vmax:
                candidates.append(value)
    if not candidates:
        return [vmin, vmax]
    if candidates[0] > vmin:
        candidates.insert(0, vmin)
    if candidates[-1] < vmax:
        candidates.append(vmax)
    return candidates


def fmt_log_tick(value):
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:g}"
    return f"{value:.3g}"


def line_chart(path, title, xlabel, ylabel, series, x_tick_formatter=None):
    width, height = 920, 540
    left, right, top, bottom = 94, 34, 62, 92
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_points = [p for item in series for p in item[2] if p[0] > 0 and p[1] > 0]
    if not all_points:
        return
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    x_min_plot = xmin / 1.12
    x_max_plot = xmax * 1.12
    y_min_plot = ymin / 1.18
    y_max_plot = ymax * 1.18
    lxmin, lxmax = math.log10(x_min_plot), math.log10(x_max_plot)
    lymin, lymax = math.log10(y_min_plot), math.log10(y_max_plot)
    lines = svg_header(width, height)
    lines.append(f'<text x="{width / 2}" y="28" text-anchor="middle" class="title">{esc(title)}</text>')
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')

    def sx(x):
        return left + safe_div(math.log10(x) - lxmin, lxmax - lxmin) * plot_w

    def sy(y):
        return top + plot_h - safe_div(math.log10(y) - lymin, lymax - lymin) * plot_h

    for x in sorted(set(xs)):
        x_pos = sx(x)
        label = x_tick_formatter(x) if x_tick_formatter else str(x)
        lines.append(f'<line x1="{x_pos:.1f}" y1="{top}" x2="{x_pos:.1f}" y2="{top + plot_h}" class="grid"/>')
        lines.append(f'<line x1="{x_pos:.1f}" y1="{top + plot_h}" x2="{x_pos:.1f}" y2="{top + plot_h + 5}" class="axis"/>')
        lines.append(f'<text x="{x_pos:.1f}" y="{top + plot_h + 23}" text-anchor="middle" class="tick">{esc(label)}</text>')

    for tick in log_ticks(y_min_plot, y_max_plot):
        y_pos = sy(tick)
        lines.append(f'<line x1="{left}" y1="{y_pos:.1f}" x2="{left + plot_w}" y2="{y_pos:.1f}" class="grid"/>')
        lines.append(f'<text x="{left - 8}" y="{y_pos + 4:.1f}" text-anchor="end" class="tick">{esc(fmt_log_tick(tick))}</text>')

    lines.append(f'<text x="{left + plot_w / 2}" y="{height - 14}" text-anchor="middle" class="label">{esc(xlabel)} (log scale)</text>')
    lines.append(f'<text x="16" y="{top + plot_h / 2}" text-anchor="middle" class="label" transform="rotate(-90 16 {top + plot_h / 2})">{esc(ylabel)} (log scale)</text>')

    legend_x = left + 8
    legend_y = top + 16
    for idx, item in enumerate(series):
        name, color, points = item[:3]
        style = item[3] if len(item) > 3 else {}
        points = sorted([p for p in points if p[0] > 0 and p[1] > 0])
        dash = ' stroke-dasharray="7 5"' if style.get("dash") else ""
        opacity = f' opacity="{style.get("opacity")}"' if style.get("opacity") else ""
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{style.get("stroke_width", 2.5)}"{dash}{opacity}/>')
        if style.get("markers", True):
            for x, y in points:
                lines.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"{opacity}/>')
        for x, y in points:
            label = style.get("point_labels", {}).get(x)
            if label:
                lines.append(f'<text x="{sx(x) + 7:.1f}" y="{sy(y) - 7:.1f}" class="point-label">{esc(label)}</text>')
        y0 = legend_y + idx * 20
        lines.append(f'<line x1="{legend_x}" y1="{y0 - 3}" x2="{legend_x + 16}" y2="{y0 - 3}" stroke="{color}" stroke-width="{style.get("stroke_width", 2.5)}"{dash}{opacity}/>')
        lines.append(f'<text x="{legend_x + 22}" y="{y0 + 2}" class="legend">{esc(name)}</text>')

    lines += svg_footer()
    write_text(path, "\n".join(lines) + "\n")


def grouped_bar_chart(path, title, xlabel, ylabel, labels, series):
    width, height = 940, 540
    left, right, top, bottom = 94, 34, 62, 112
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_values = [v for _, _, values in series for v in values]
    y_ticks = nice_ticks(max(all_values) * 1.15 if all_values else 1)
    y_max = y_ticks[-1]
    lines = svg_header(width, height)
    draw_axes(lines, width, height, left, right, top, bottom, y_ticks, y_max, title, xlabel, ylabel)
    group_w = plot_w / max(len(labels), 1)
    gap = group_w * 0.18
    bar_area = group_w - gap
    bar_w = bar_area / max(len(series), 1)
    for i, label in enumerate(labels):
        x0 = left + i * group_w + gap / 2
        for j, (_, color, values) in enumerate(series):
            value = values[i]
            h = safe_div(value, y_max) * plot_h
            x = x0 + j * bar_w
            y = top + plot_h - h
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bar_w - 3, 1):.1f}" height="{h:.1f}" fill="{color}"/>')
        lx = left + i * group_w + group_w / 2
        lines.append(f'<text x="{lx:.1f}" y="{top + plot_h + 24}" text-anchor="middle" class="tick">{esc(label)}</text>')
    legend_x = left + 8
    legend_y = top + 16
    for idx, (name, color, _) in enumerate(series):
        y0 = legend_y + idx * 20
        lines.append(f'<rect x="{legend_x}" y="{y0 - 9}" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text x="{legend_x + 18}" y="{y0 + 2}" class="legend">{esc(name)}</text>')
    lines += svg_footer()
    write_text(path, "\n".join(lines) + "\n")


def stacked_bar_chart(path, title, xlabel, ylabel, labels, stacks):
    width, height = 940, 540
    left, right, top, bottom = 94, 34, 62, 112
    plot_w = width - left - right
    plot_h = height - top - bottom
    totals = [sum(values[i] for _, _, values in stacks) for i in range(len(labels))]
    y_ticks = nice_ticks(max(totals) * 1.15 if totals else 1)
    y_max = y_ticks[-1]
    lines = svg_header(width, height)
    draw_axes(lines, width, height, left, right, top, bottom, y_ticks, y_max, title, xlabel, ylabel)
    group_w = plot_w / max(len(labels), 1)
    bar_w = group_w * 0.54
    for i, label in enumerate(labels):
        x = left + i * group_w + (group_w - bar_w) / 2
        cursor = top + plot_h
        for _, color, values in stacks:
            value = values[i]
            h = safe_div(value, y_max) * plot_h
            cursor -= h
            lines.append(f'<rect x="{x:.1f}" y="{cursor:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        lx = left + i * group_w + group_w / 2
        lines.append(f'<text x="{lx:.1f}" y="{top + plot_h + 24}" text-anchor="middle" class="tick">{esc(label)}</text>')
    legend_x = left + 8
    legend_y = top + 16
    for idx, (name, color, _) in enumerate(stacks):
        y0 = legend_y + idx * 20
        lines.append(f'<rect x="{legend_x}" y="{y0 - 9}" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text x="{legend_x + 18}" y="{y0 + 2}" class="legend">{esc(name)}</text>')
    lines += svg_footer()
    write_text(path, "\n".join(lines) + "\n")


def case_lookup_rows(case_rows):
    lookup = {}
    expected = []
    for row in case_rows:
        base = {
            "case_order": row["case_order"],
            "run_case_id": row["run_case_id"],
            "study_tags": row["study_tags"],
            "variant": row["variant"],
            "nranks": row["nranks"],
            "n1": row["n1"],
            "n2": row["n2"],
            "n3": row["n3"],
            "m": row["m"],
            "notes": row["notes"],
        }
        if str(row.get("run_fortran", "0")) == "1":
            item = dict(base, implementation="fortran-original", mpi_mode="device")
            expected.append(item)
            lookup[("fortran-original", "device", row["nranks"], row["n1"], row["n2"], row["n3"], row["m"])] = item
        if str(row.get("run_cxx", "0")) == "1":
            for mode in str(row.get("cxx_mpi_modes", "device")).split():
                item = dict(base, implementation="cuda-cxx", mpi_mode=mode)
                expected.append(item)
                lookup[("cuda-cxx", mode, row["nranks"], row["n1"], row["n2"], row["n3"], row["m"])] = item
    return lookup, expected


def enrich_rows(rows, lookup):
    for row in rows:
        key = (row["implementation"], row["mpi_mode"], row["nranks"], row["n1"], row["n2"], row["n3"], row["m"])
        meta = lookup.get(key)
        if meta:
            row.update({
                "case_order": meta["case_order"],
                "run_case_id": meta["run_case_id"],
                "study_tags": meta["study_tags"],
                "study_notes": meta["notes"],
                "tags": set(meta["study_tags"].split(";")),
            })
        else:
            row.setdefault("case_order", 9999)
            row.setdefault("run_case_id", f"{row['implementation']}_{row['nranks']}_{row['n1']}x{row['n2']}x{row['n3']}_m{row['m']}_{row['mpi_mode']}")
            row.setdefault("study_tags", "")
            row.setdefault("study_notes", "")
            row.setdefault("tags", set())


def group_timing(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["implementation"],
            row["mpi_mode"],
            row["nranks"],
            row["n1"],
            row["n2"],
            row["n3"],
            row["m"],
        )
        groups[key].append(row)
    summaries = []
    for key, group_rows in groups.items():
        stable = [r for r in group_rows if r["iter"] >= 1] or list(group_rows)
        first = next((r for r in group_rows if r["iter"] == 0), group_rows[0])
        base = stable[0]
        total_values = [r["total_s_max"] for r in stable]
        local_values = [r["local_compute_s_max"] for r in stable]
        forward_values = [r["forward_exchange_s_max"] for r in stable]
        reduced_values = [r["reduced_compute_s_max"] for r in stable]
        backward_values = [r["backward_exchange_s_max"] for r in stable]
        update_values = [r["update_compute_s_max"] for r in stable]
        compute_values = [r["compute_s_max"] for r in stable]
        communication_values = [r["communication_s_max"] for r in stable]
        total_mean = mean(total_values)
        ncell = base["n1"] * base["n2"] * base["n3"]
        work_units = base["nsys"] * base["n3"] * (base["m"] ** 3)
        row = {
            "implementation": key[0],
            "mpi_mode": key[1],
            "nranks": key[2],
            "n1": key[3],
            "n2": key[4],
            "n3": key[5],
            "m": key[6],
            "nsys": base["nsys"],
            "nrow_min": base["nrow_min"],
            "nrow_max": base["nrow_max"],
            "case_order": base.get("case_order", 9999),
            "run_case_id": base.get("run_case_id", ""),
            "study_tags": base.get("study_tags", ""),
            "study_notes": base.get("study_notes", ""),
            "stable_iterations": len(stable),
            "total_s_mean": total_mean,
            "total_s_std": stdev(total_values),
            "total_s_min": min(total_values),
            "total_s_max_observed": max(total_values),
            "total_s_iter0": first["total_s_max"],
            "warmup_ratio": safe_div(first["total_s_max"], total_mean),
            "local_compute_s_mean": mean(local_values),
            "forward_exchange_s_mean": mean(forward_values),
            "reduced_compute_s_mean": mean(reduced_values),
            "backward_exchange_s_mean": mean(backward_values),
            "update_compute_s_mean": mean(update_values),
            "compute_s_mean": mean(compute_values),
            "communication_s_mean": mean(communication_values),
            "throughput_mcells_s": safe_div(ncell, total_mean) / 1.0e6,
            "work_throughput_gunits_s": safe_div(work_units, total_mean) / 1.0e9,
            "cv_percent": safe_div(stdev(total_values), total_mean) * 100.0,
            "comm_percent": safe_div(mean(communication_values), total_mean) * 100.0,
            "payload_m2_nsys": base["m"] * base["m"] * base["nsys"],
        }
        summaries.append(row)
    summaries.sort(key=lambda r: (r["case_order"], r["implementation"], r["mpi_mode"]))
    return summaries


def actual_key(row):
    return (row["implementation"], row["mpi_mode"], row["nranks"], row["n1"], row["n2"], row["n3"], row["m"])


def expected_key(row):
    return (row["implementation"], row["mpi_mode"], row["nranks"], row["n1"], row["n2"], row["n3"], row["m"])


def has_tag(row, tag):
    return tag in set(str(row.get("study_tags", "")).split(";"))


def write_summary_csv(timing):
    headers = [
        "run_case_id",
        "implementation",
        "mpi_mode",
        "nranks",
        "n1",
        "n2",
        "n3",
        "m",
        "nsys",
        "nrow_max",
        "stable_iterations",
        "total_s_mean",
        "total_s_std",
        "total_s_iter0",
        "warmup_ratio",
        "local_compute_s_mean",
        "forward_exchange_s_mean",
        "reduced_compute_s_mean",
        "backward_exchange_s_mean",
        "update_compute_s_mean",
        "compute_s_mean",
        "communication_s_mean",
        "throughput_mcells_s",
        "work_throughput_gunits_s",
        "cv_percent",
        "comm_percent",
        "payload_m2_nsys",
        "study_tags",
    ]
    rows = []
    for r in timing:
        rows.append([
            r["run_case_id"],
            r["implementation"],
            r["mpi_mode"],
            r["nranks"],
            r["n1"],
            r["n2"],
            r["n3"],
            r["m"],
            r["nsys"],
            r["nrow_max"],
            r["stable_iterations"],
            f"{r['total_s_mean']:.12g}",
            f"{r['total_s_std']:.12g}",
            f"{r['total_s_iter0']:.12g}",
            f"{r['warmup_ratio']:.12g}",
            f"{r['local_compute_s_mean']:.12g}",
            f"{r['forward_exchange_s_mean']:.12g}",
            f"{r['reduced_compute_s_mean']:.12g}",
            f"{r['backward_exchange_s_mean']:.12g}",
            f"{r['update_compute_s_mean']:.12g}",
            f"{r['compute_s_mean']:.12g}",
            f"{r['communication_s_mean']:.12g}",
            f"{r['throughput_mcells_s']:.12g}",
            f"{r['work_throughput_gunits_s']:.12g}",
            f"{r['cv_percent']:.12g}",
            f"{r['comm_percent']:.12g}",
            r["payload_m2_nsys"],
            r["study_tags"],
        ])
    write_csv(TABLE_DIR / "summary_by_case.csv", headers, rows)


def table_data_coverage(profile_rows, signature_rows, case_rows, timing, expected):
    actual = {actual_key(r) for r in timing}
    missing = [r for r in expected if expected_key(r) not in actual]
    expected_counts = defaultdict(int)
    actual_counts = defaultdict(int)
    for r in expected:
        expected_counts[r["implementation"]] += 1
    for r in timing:
        actual_counts[r["implementation"]] += 1
    rows = [
        ["profile rows", len(profile_rows)],
        ["profile cases", len(timing)],
        ["signature rows", len(signature_rows)],
        ["case matrix rows", len(case_rows)],
        ["expected implementation cases", len(expected)],
        ["observed implementation cases", len(timing)],
        ["missing expected implementation cases", len(missing)],
        ["implementations in profile", ", ".join(sorted({r["implementation"] for r in profile_rows}))],
        ["MPI modes in profile", ", ".join(sorted({r["mpi_mode"] for r in profile_rows}))],
        ["block sizes m", ", ".join(str(x) for x in sorted({r["m"] for r in profile_rows}))],
        ["rank counts", ", ".join(str(x) for x in sorted({r["nranks"] for r in profile_rows}))],
        ["Fortran timing rows present", "yes" if any(r["implementation"] == "fortran-original" for r in profile_rows) else "no"],
        ["expected cuda-cxx cases", expected_counts.get("cuda-cxx", 0)],
        ["observed cuda-cxx cases", actual_counts.get("cuda-cxx", 0)],
        ["expected fortran-original cases", expected_counts.get("fortran-original", 0)],
        ["observed fortran-original cases", actual_counts.get("fortran-original", 0)],
    ]
    write_text(TABLE_DIR / "0_data_coverage.md", md_table(["item", "value"], rows))


def table_expected_vs_observed(timing, expected):
    actual = {actual_key(r) for r in timing}
    rows = []
    for r in expected:
        observed = "yes" if expected_key(r) in actual else "no"
        if observed == "no" or r["implementation"] == "fortran-original":
            rows.append([
                r["run_case_id"],
                r["implementation"],
                r["mpi_mode"],
                r["nranks"],
                f"{r['n1']}x{r['n2']}x{r['n3']}",
                r["m"],
                observed,
            ])
    intro = (
        "The case matrix requested Fortran and CUDA C++ device runs for 28 base cases, "
        "plus 3 CUDA C++ host-fallback runs. The actual timing CSV contains CUDA C++ rows only.\n\n"
    )
    write_text(
        TABLE_DIR / "10_expected_vs_observed_runs.md",
        intro + md_table(["case", "implementation", "mode", "np", "grid", "m", "observed"], rows),
    )


def table_signature(signature_rows):
    groups = defaultdict(list)
    for row in signature_rows:
        key = (row["implementation"], row["mpi_mode"], row["n1"], row["n2"], row["n3"], row["m"])
        groups[key].append(row)
    rows = []
    for key, group in sorted(groups.items(), key=lambda item: (item[0][3], item[0][4], item[0][5], item[0][0])):
        sums = [r["solution_sum"] for r in group]
        l2s = [r["solution_l2"] for r in group]
        linfs = [r["solution_linf"] for r in group]
        rows.append([
            key[0],
            key[1],
            f"{key[2]}x{key[3]}x{key[4]}",
            key[5],
            ",".join(str(r["nranks"]) for r in sorted(group, key=lambda r: r["nranks"])),
            fmt_sci(max(sums) - min(sums), 3),
            fmt_sci(max(l2s) - min(l2s), 3),
            fmt_float(mean(linfs), 6),
            fmt_float(mean(r["sample_zmid"] for r in group), 6),
        ])
    intro = (
        "These rows are solution signatures, not a manufactured-solution correctness proof. "
        "They are useful for checking that repeated rank decompositions and MPI modes produce the same first-solve signature.\n\n"
    )
    write_text(
        TABLE_DIR / "1_signature_summary.md",
        intro
        + md_table(
            ["implementation", "mode", "grid", "m", "np values", "sum range", "l2 range", "linf mean", "zmid mean"],
            rows,
        ),
    )


def central_rows(timing):
    return sorted(
        [
            r
            for r in timing
            if r["n1"] == 64 and r["n2"] == 64 and r["n3"] == 2048 and r["m"] == 5
        ],
        key=lambda r: (r["mpi_mode"], r["nranks"]),
    )


def table_central(timing):
    rows = []
    for r in central_rows(timing):
        rows.append([
            r["implementation"],
            r["mpi_mode"],
            r["nranks"],
            fmt_ms(r["total_s_mean"]),
            fmt_ms(r["local_compute_s_mean"]),
            fmt_ms(r["forward_exchange_s_mean"]),
            fmt_ms(r["reduced_compute_s_mean"]),
            fmt_ms(r["backward_exchange_s_mean"]),
            fmt_ms(r["update_compute_s_mean"]),
            fmt_float(r["throughput_mcells_s"], 1),
            fmt_float(r["work_throughput_gunits_s"], 2),
        ])
    write_text(
        TABLE_DIR / "2_central_case_timing.md",
        md_table(
            [
                "implementation",
                "mode",
                "np",
                "total_ms",
                "local_ms",
                "forward_ms",
                "reduced_ms",
                "backward_ms",
                "update_ms",
                "Mcells_s",
                "work_Gunits_s",
            ],
            rows,
        ),
    )


def table_phase_breakdown(timing):
    rows = []
    for r in central_rows(timing):
        if r["mpi_mode"] != "device":
            continue
        rows.append([
            r["nranks"],
            fmt_ms(r["local_compute_s_mean"]),
            fmt_ms(r["forward_exchange_s_mean"]),
            fmt_ms(r["reduced_compute_s_mean"]),
            fmt_ms(r["backward_exchange_s_mean"]),
            fmt_ms(r["update_compute_s_mean"]),
            fmt_float(r["comm_percent"], 2),
        ])
    write_text(
        TABLE_DIR / "4_phase_breakdown.md",
        md_table(["np", "local_ms", "forward_ms", "reduced_ms", "backward_ms", "update_ms", "comm_percent"], rows),
    )


def table_strong(timing):
    rows = []
    by_m = defaultdict(list)
    for r in timing:
        if r["mpi_mode"] == "device" and r["n1"] == 64 and r["n2"] == 64 and r["n3"] == 2048:
            by_m[r["m"]].append(r)
    for m in sorted(by_m):
        subset = sorted(by_m[m], key=lambda r: r["nranks"])
        base = next((r for r in subset if r["nranks"] == 2), None)
        if not base:
            continue
        for r in subset:
            speedup = safe_div(base["total_s_mean"], r["total_s_mean"])
            efficiency = safe_div(speedup, r["nranks"] / 2.0) * 100.0
            rows.append([
                m,
                r["nranks"],
                fmt_ms(r["total_s_mean"]),
                fmt_float(speedup, 3),
                fmt_float(efficiency, 1),
                fmt_float(r["work_throughput_gunits_s"], 2),
                fmt_float(r["comm_percent"], 2),
            ])
    write_text(
        TABLE_DIR / "3_strong_scaling.md",
        md_table(["m", "np", "total_ms", "speedup_2base", "efficiency_percent", "work_Gunits_s", "comm_percent"], rows),
    )


def table_m_sensitivity(timing):
    rows = []
    subset = [
        r
        for r in timing
        if r["mpi_mode"] == "device" and r["n1"] == 64 and r["n2"] == 64 and r["n3"] == 2048
    ]
    for r in sorted(subset, key=lambda r: (r["nranks"], r["m"])):
        rows.append([
            r["nranks"],
            r["m"],
            r["payload_m2_nsys"],
            fmt_ms(r["total_s_mean"]),
            fmt_ms(r["compute_s_mean"]),
            fmt_ms(r["communication_s_mean"]),
            fmt_float(r["work_throughput_gunits_s"], 2),
        ])
    write_text(
        TABLE_DIR / "5_m_sensitivity.md",
        md_table(["np", "m", "m2_nsys_payload", "total_ms", "compute_ms", "comm_ms", "work_Gunits_s"], rows),
    )


def table_nsys(timing):
    rows = []
    subset = [
        r
        for r in timing
        if r["mpi_mode"] == "device" and r["nranks"] == 8 and r["n3"] == 2048 and r["nrow_max"] == 256
    ]
    for r in sorted(subset, key=lambda r: (r["m"], r["nsys"])):
        rows.append([
            r["m"],
            r["nsys"],
            f"{r['n1']}x{r['n2']}x{r['n3']}",
            r["payload_m2_nsys"],
            fmt_ms(r["total_s_mean"]),
            fmt_float(r["throughput_mcells_s"], 1),
            fmt_float(r["work_throughput_gunits_s"], 2),
        ])
    write_text(
        TABLE_DIR / "6_nsys_sensitivity.md",
        md_table(["m", "nsys", "grid", "m2_nsys_payload", "total_ms", "Mcells_s", "work_Gunits_s"], rows),
    )


def table_weak_nrow(timing):
    rows = []
    by_m = defaultdict(list)
    for r in timing:
        if r["mpi_mode"] == "device" and r["nsys"] == 4096 and r["nrow_max"] == 512:
            by_m[r["m"]].append(r)
    for m in sorted(by_m):
        subset = sorted(by_m[m], key=lambda r: r["nranks"])
        base = next((r for r in subset if r["nranks"] == 2), None)
        if not base:
            continue
        for r in subset:
            weak_eff = safe_div(base["total_s_mean"], r["total_s_mean"]) * 100.0
            rows.append([
                m,
                r["nranks"],
                f"{r['n1']}x{r['n2']}x{r['n3']}",
                r["nrow_max"],
                fmt_ms(r["total_s_mean"]),
                fmt_float(weak_eff, 1),
                fmt_float(r["work_throughput_gunits_s"], 2),
            ])
    write_text(
        TABLE_DIR / "7_weak_nrow_scaling.md",
        md_table(["m", "np", "grid", "local_nrow", "total_ms", "weak_efficiency_percent", "work_Gunits_s"], rows),
    )


def table_mpi_mode(timing):
    rows = []
    central = central_rows(timing)
    by_np_mode = defaultdict(dict)
    for r in central:
        by_np_mode[r["nranks"]][r["mpi_mode"]] = r
    for np in sorted(by_np_mode):
        device = by_np_mode[np].get("device")
        host = by_np_mode[np].get("host")
        for mode in ("device", "host"):
            r = by_np_mode[np].get(mode)
            if not r:
                continue
            ratio = safe_div(r["total_s_mean"], device["total_s_mean"]) if device and mode == "host" else float("nan")
            rows.append([
                np,
                mode,
                fmt_ms(r["total_s_mean"]),
                fmt_ms(r["compute_s_mean"]),
                fmt_ms(r["communication_s_mean"]),
                fmt_float(r["comm_percent"], 2),
                "-" if math.isnan(ratio) else fmt_float(ratio, 3),
            ])
    write_text(
        TABLE_DIR / "8_mpi_mode_comparison.md",
        md_table(["np", "mode", "total_ms", "compute_ms", "comm_ms", "comm_percent", "host_over_device"], rows),
    )


def table_warmup(timing):
    central = [r for r in central_rows(timing) if r["mpi_mode"] == "device"]
    rows = []
    for r in central:
        rows.append([
            r["nranks"],
            fmt_ms(r["total_s_iter0"]),
            fmt_ms(r["total_s_mean"]),
            fmt_float(r["warmup_ratio"], 1),
            fmt_float(r["cv_percent"], 2),
        ])
    top = sorted(timing, key=lambda r: r["warmup_ratio"], reverse=True)[:8]
    rows2 = [
        [
            r["run_case_id"],
            r["mpi_mode"],
            r["nranks"],
            f"{r['n1']}x{r['n2']}x{r['n3']}",
            r["m"],
            fmt_float(r["warmup_ratio"], 1),
        ]
        for r in top
    ]
    content = md_table(["np", "iter0_ms", "iter1_9_mean_ms", "iter0_over_stable", "stable_cv_percent"], rows)
    content += "\nTop warm-up ratios across all observed cases:\n\n"
    content += md_table(["case", "mode", "np", "grid", "m", "iter0_over_stable"], rows2)
    write_text(TABLE_DIR / "9_warmup_effect.md", content)


def make_figures(timing):
    strong_series_time = []
    strong_series_work = []
    color_by_m = {2: COLORS["blue"], 3: COLORS["green"], 5: COLORS["orange"], 8: COLORS["purple"]}
    for m in (2, 3, 5, 8):
        subset = sorted(
            [
                r
                for r in timing
                if r["mpi_mode"] == "device"
                and r["n1"] == 64
                and r["n2"] == 64
                and r["n3"] == 2048
                and r["m"] == m
            ],
            key=lambda r: r["nranks"],
        )
        if not subset:
            continue
        base = next((r for r in subset if r["nranks"] == 2), subset[0])
        labels = {}
        for r in subset:
            speedup = safe_div(base["total_s_mean"], r["total_s_mean"])
            efficiency = safe_div(speedup, r["nranks"] / 2.0) * 100.0
            labels[r["nranks"]] = f"{efficiency:.0f}%"
        measured_time = [(r["nranks"], r["total_s_mean"] * 1000) for r in subset]
        ideal_time = [(r["nranks"], base["total_s_mean"] * 1000 * safe_div(2.0, r["nranks"])) for r in subset]
        measured_work = [(r["nranks"], r["work_throughput_gunits_s"]) for r in subset]
        ideal_work = [(r["nranks"], base["work_throughput_gunits_s"] * safe_div(r["nranks"], 2.0)) for r in subset]
        strong_series_time.append((f"m={m}", color_by_m[m], measured_time, {"point_labels": labels}))
        strong_series_time.append((f"m={m} ideal", color_by_m[m], ideal_time, {"dash": True, "markers": False, "opacity": "0.42", "stroke_width": 2}))
        strong_series_work.append((f"m={m}", color_by_m[m], measured_work, {"point_labels": labels}))
        strong_series_work.append((f"m={m} ideal", color_by_m[m], ideal_work, {"dash": True, "markers": False, "opacity": "0.42", "stroke_width": 2}))
    line_chart(FIGURE_DIR / "1_strong_scaling_time.svg", "Strong Scaling by Block Size", "MPI ranks / GPUs", "total_s_max mean (ms)", strong_series_time, lambda x: f"{int(x)}")
    line_chart(FIGURE_DIR / "2_strong_scaling_work_throughput.svg", "Strong Scaling Work-Weighted Throughput", "MPI ranks / GPUs", "nsys*N*m^3 / s (G units)", strong_series_work, lambda x: f"{int(x)}")

    central_device = [r for r in central_rows(timing) if r["mpi_mode"] == "device"]
    labels = [f"np={r['nranks']}" for r in central_device]
    stacked_bar_chart(
        FIGURE_DIR / "3_phase_breakdown_central_m5_device.svg",
        "Central m=5 Device Mode: Phase Breakdown",
        "64x64x2048, m=5",
        "phase time mean (ms)",
        labels,
        [
            ("local", COLORS["blue"], [r["local_compute_s_mean"] * 1000 for r in central_device]),
            ("forward", COLORS["orange"], [r["forward_exchange_s_mean"] * 1000 for r in central_device]),
            ("reduced", COLORS["purple"], [r["reduced_compute_s_mean"] * 1000 for r in central_device]),
            ("backward", COLORS["red"], [r["backward_exchange_s_mean"] * 1000 for r in central_device]),
            ("update", COLORS["green"], [r["update_compute_s_mean"] * 1000 for r in central_device]),
        ],
    )

    ms = [2, 3, 5, 8]
    m_subset = {
        np: {
            r["m"]: r
            for r in timing
            if r["mpi_mode"] == "device" and r["n1"] == 64 and r["n2"] == 64 and r["n3"] == 2048 and r["nranks"] == np
        }
        for np in (2, 4, 8)
    }
    grouped_bar_chart(
        FIGURE_DIR / "4_m_sensitivity.svg",
        "Block Size Sensitivity",
        "block size m",
        "total_s_max mean (ms)",
        [f"m={m}" for m in ms],
        [
            (f"np={np}", color, [m_subset[np][m]["total_s_mean"] * 1000 for m in ms])
            for np, color in [(2, COLORS["blue"]), (4, COLORS["green"]), (8, COLORS["purple"])]
        ],
    )

    nsys_series = []
    for m, color in [(2, COLORS["blue"]), (5, COLORS["orange"]), (8, COLORS["purple"])]:
        subset = sorted(
            [
                r
                for r in timing
                if r["mpi_mode"] == "device" and r["nranks"] == 8 and r["n3"] == 2048 and r["m"] == m
            ],
            key=lambda r: r["nsys"],
        )
        nsys_series.append((f"m={m}", color, [(r["nsys"], r["total_s_mean"] * 1000) for r in subset]))
    line_chart(FIGURE_DIR / "5_nsys_sensitivity.svg", "Nsys Sensitivity at np=8", "nsys = n1*n2", "total_s_max mean (ms)", nsys_series, lambda x: f"{int(x)}")

    weak_series = []
    for m, color in [(2, COLORS["blue"]), (5, COLORS["orange"]), (8, COLORS["purple"])]:
        subset = sorted(
            [
                r
                for r in timing
                if r["mpi_mode"] == "device" and r["nsys"] == 4096 and r["nrow_max"] == 512 and r["m"] == m
            ],
            key=lambda r: r["nranks"],
        )
        if not subset:
            continue
        base = next((r for r in subset if r["nranks"] == 2), subset[0])
        labels = {}
        for r in subset:
            efficiency = safe_div(base["total_s_mean"], r["total_s_mean"]) * 100.0
            labels[r["nranks"]] = f"{efficiency:.0f}%"
        measured_time = [(r["nranks"], r["total_s_mean"] * 1000) for r in subset]
        ideal_time = [(r["nranks"], base["total_s_mean"] * 1000) for r in subset]
        weak_series.append((f"m={m}", color, measured_time, {"point_labels": labels}))
        weak_series.append((f"m={m} ideal", color, ideal_time, {"dash": True, "markers": False, "opacity": "0.42", "stroke_width": 2}))
    line_chart(FIGURE_DIR / "6_weak_nrow_scaling.svg", "Weak Nrow Scaling", "MPI ranks / GPUs", "total_s_max mean (ms)", weak_series, lambda x: f"{int(x)}")

    by_np_mode = defaultdict(dict)
    for r in central_rows(timing):
        by_np_mode[r["nranks"]][r["mpi_mode"]] = r
    np_labels = [f"np={np}" for np in sorted(by_np_mode)]
    grouped_bar_chart(
        FIGURE_DIR / "7_mpi_mode_device_vs_host.svg",
        "MPI Mode Comparison",
        "64x64x2048, m=5",
        "total_s_max mean (ms)",
        np_labels,
        [
            ("device", COLORS["blue"], [by_np_mode[np]["device"]["total_s_mean"] * 1000 for np in sorted(by_np_mode)]),
            ("host", COLORS["red"], [by_np_mode[np]["host"]["total_s_mean"] * 1000 for np in sorted(by_np_mode)]),
        ],
    )

    grouped_bar_chart(
        FIGURE_DIR / "8_warmup_effect_central_m5_device.svg",
        "Warm-up Effect: First Iteration vs Stable Mean",
        "64x64x2048, m=5, device mode",
        "total_s_max (ms)",
        [f"np={r['nranks']}" for r in central_device],
        [
            ("iter 0", COLORS["red"], [r["total_s_iter0"] * 1000 for r in central_device]),
            ("iter 1-9 mean", COLORS["blue"], [r["total_s_mean"] * 1000 for r in central_device]),
        ],
    )


def make_outputs():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    profile_path = latest("btdma_total_profile_*.csv")
    signature_path = latest("btdma_solution_signature_*.csv")
    case_path = latest("btdma_full_case_list_*.csv")
    env_path = latest("btdma_environment_*.txt")

    profile_rows = load_csv(profile_path)
    signature_rows = load_csv(signature_path)
    case_rows = load_csv(case_path)
    env = load_environment(env_path)

    lookup, expected = case_lookup_rows(case_rows)
    enrich_rows(profile_rows, lookup)
    enrich_rows(signature_rows, lookup)
    timing = group_timing(profile_rows)

    write_summary_csv(timing)
    table_data_coverage(profile_rows, signature_rows, case_rows, timing, expected)
    table_signature(signature_rows)
    table_central(timing)
    table_strong(timing)
    table_phase_breakdown(timing)
    table_m_sensitivity(timing)
    table_nsys(timing)
    table_weak_nrow(timing)
    table_mpi_mode(timing)
    table_warmup(timing)
    table_expected_vs_observed(timing, expected)
    make_figures(timing)

    lines = [
        "# BTDMA analysis inputs",
        "",
        f"profile_csv={profile_path.name}",
        f"signature_csv={signature_path.name}",
        f"case_list_csv={case_path.name}",
        f"environment_txt={env_path.name}",
        f"date={env.get('date', 'unknown')}",
        f"host={env.get('hostname', 'unknown')}",
        f"gpu={env.get('gpu_count', '?')} x {env.get('gpu_name', 'unknown')}",
        f"nvcc={env.get('nvcc_release', 'unknown')}",
        f"mpi={env.get('openmpi_version', 'unknown')}",
        f"profile_rows={len(profile_rows)}",
        f"profile_cases={len(timing)}",
        f"signature_rows={len(signature_rows)}",
        f"expected_implementation_cases={len(expected)}",
        f"observed_implementation_cases={len(timing)}",
    ]
    write_text(TABLE_DIR / "analysis_inputs.md", "\n".join(lines) + "\n")

    print(f"wrote {TABLE_DIR}")
    print(f"wrote {FIGURE_DIR}")


if __name__ == "__main__":
    make_outputs()
