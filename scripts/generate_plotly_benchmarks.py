#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Generate an interactive benchmark dashboard using Plotly.js.

Reads one or more dispenso benchmark JSON files and produces a single
self-contained HTML file with interactive charts, dark/light theme,
sidebar navigation, platform switching, and suite-specific visualizations.

Usage:
    python generate_plotly_benchmarks.py results/macos.json
    python generate_plotly_benchmarks.py results/macos.json results/linux.json
    python generate_plotly_benchmarks.py results/*.json -o docs/benchmarks/index.html
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# ─── Color palette (consistent with generate_charts.py) ────────────────────

LIBRARY_COLORS = {
    # Primary libraries - must be visually distinct from each other
    "dispenso": "#2ecc71",  # Green
    "tbb": "#3498db",  # Blue
    "omp": "#e74c3c",  # Red
    "taskflow": "#f39c12",  # Orange
    "folly": "#9b59b6",  # Purple
    "serial": "#95a5a6",  # Gray
    "std": "#1abc9c",  # Teal
    "deque": "#e67e22",  # Dark orange
    "async": "#f1c40f",  # Yellow
    # Dispenso variants - each distinct, avoiding primary library colors
    # Must not use: blue (tbb), red (omp), orange (taskflow), purple (folly)
    "dispenso_static_chunk": "#1e8449",  # Dark forest green
    "dispenso_auto_chunk": "#00bcd4",  # Cyan (distinct from green family)
    "dispenso_static": "#1e8449",  # Dark forest green
    "dispenso_auto": "#00bcd4",  # Cyan (distinct from green family)
    "dispenso_bulk": "#1e8449",  # Dark green, dashed in line charts
    "dispenso_par": "#27ae60",  # Darker green (variant of dispenso)
    "dispenso_mostly_idle": "#48c9b0",  # Teal-green
    "dispenso_very_idle": "#76d7c4",  # Light teal
    "dispenso_mixed": "#58d68d",  # Light green
    "dispenso2": "#27ae60",  # Darker green
    # Cascading parallel for variants
    "dispenso_blocking": "#2ecc71",  # Green (same as dispenso)
    "dispenso_cascaded": "#2ecc71",  # Green (distinguished by dash)
    "tbb_task_group": "#3498db",  # Blue (distinguished by dash)
    # for_each container variants - each distinct
    "for_each_n": "#2ecc71",  # Green (dispenso default)
    "for_each_n_deque": "#e67e22",  # Dark orange
    "for_each_n_list": "#3498db",  # Blue
    "for_each_n_set": "#9b59b6",  # Purple
}

IMPL_COLORS = {
    "serial": "#95a5a6",
    "std::async": "#e74c3c",
    "folly::Future": "#e67e22",
    "dispenso::Future": "#3498db",
    "dispenso::TaskSet": "#2ecc71",
    "dispenso::TaskSet (bulk)": "#1e8449",
    "dispenso::when_all": "#9b59b6",
    "std::vector": "#808080",
    "std::deque": "#FF8C00",
    "tbb::concurrent_vector": "#4285F4",
    "dispenso::ConcurrentVector": "#34A853",
    "NopMutex": "#95a5a6",
    "std::shared_mutex": "#e74c3c",
    "dispenso::RWLock": "#2ecc71",
    "std::function": "#e74c3c",
    "dispenso::OnceFunction": "#2ecc71",
    "InlineFunction": "#3498db",
    "malloc/free": "#e74c3c",
    "PoolAllocator": "#2ecc71",
    "Arena": "#27ae60",
    "NoLock": "#3498db",
    "NoLock Arena": "#2980b9",
    "SmallBufferAllocator": "#2ecc71",
    "new/delete": "#e74c3c",
}


def get_color(name):
    """Get color for a library/implementation name.

    Tries exact match first, then progressively longer prefix matches
    to ensure dispenso_static gets a different color than dispenso.
    """
    if name in IMPL_COLORS:
        return IMPL_COLORS[name]
    nl = name.lower()
    # Try exact match in LIBRARY_COLORS
    if nl in LIBRARY_COLORS:
        return LIBRARY_COLORS[nl]
    # Try longest-prefix match so 'dispenso_static' beats 'dispenso'
    best_match = ""
    best_color = "#7f8c8d"
    for key, color in LIBRARY_COLORS.items():
        if (nl.startswith(key) or key in nl) and len(key) > len(best_match):
            best_match = key
            best_color = color
    return best_color


# ─── Benchmark name parsing ────────────────────────────────────────────────


def parse_benchmark_name(name):
    result = {"library": name, "threads": None, "worksize": None, "raw": name}
    if "<" in name and ">" in name:
        m = re.match(r"([^<]+)<([^>]+)>", name)
        if m:
            result["library"] = m.group(1)
            result["worksize"] = m.group(2)
        return result
    parts = name.split("/")
    if parts and parts[-1] == "real_time":
        parts = parts[:-1]
    if len(parts) >= 3:
        result["library"] = parts[0]
        try:
            result["threads"] = int(parts[1])
            result["worksize"] = parts[2]
        except (ValueError, IndexError):
            pass
    elif len(parts) == 2:
        result["library"] = parts[0]
        try:
            result["threads"] = int(parts[1])
            result["worksize"] = "default"
        except ValueError:
            pass
    return result


# ─── Shared helpers ────────────────────────────────────────────────────────


def _format_worksize(ws):
    """Format a numeric worksize string into a compact label (e.g. '1000000' → '1M')."""
    ws_int = int(ws) if ws.isdigit() else 0
    if ws_int >= 1_000_000_000:
        return f"{ws_int // 1_000_000_000}B"
    if ws_int >= 1_000_000:
        return f"{ws_int // 1_000_000}M"
    if ws_int >= 1_000:
        return f"{ws_int // 1_000}K"
    return ws


def _lib_sort_key(lib):
    """Sort key: serial first, dispenso, then dispenso variants, then others."""
    ll = lib.lower()
    if ll == "serial":
        return (0, ll)
    if ll == "dispenso":
        return (1, ll)
    if ll.startswith("dispenso"):
        return (2, ll)
    return (3, ll)


# ─── Suite-specific chart builders ──────────────────────────────────────────
# Each returns a list of chart config dicts consumable by the JS renderer.


def _parse_line_benchmarks(benchmarks):
    """Parse benchmarks into grouped thread-scaling data and serial baselines."""
    size_map = {
        "kSmallSize": "1000",
        "kMediumSize": "1000000",
        "kLargeSize": "100000000",
        "100": "100",
    }
    grouped = defaultdict(dict)
    serial_points = {}

    for bm in benchmarks:
        if bm.get("run_type") != "iteration":
            continue
        name = bm["name"].replace("BM_", "")
        parsed = parse_benchmark_name(name)
        lib = parsed["library"]

        # Filter out auto_chunk, static_chunk, and other variants (auto, static)
        # to keep charts clean - matches generate_charts.py behavior
        lib_lower = lib.lower()
        if "auto" in lib_lower or "static" in lib_lower:
            continue

        if parsed.get("worksize") and "<" not in bm["name"]:
            # Slash-delimited: lib/threads/worksize
            threads = parsed["threads"]
            if threads is not None:
                grouped[parsed["worksize"]][(lib, threads)] = bm["real_time"]
        elif "<" in bm["name"]:
            # Template: serial<kSmallSize>
            m = re.match(r"BM_([^<]+)<([^>]+)>", bm["name"])
            if m:
                ws_key = m.group(2)
                serial_points[size_map.get(ws_key, ws_key)] = bm["real_time"]

    return grouped, serial_points


def build_line_charts(benchmarks, suite):
    """Thread-scaling line charts for simple_for, summing_for, trivial_compute, nested_for."""
    grouped, serial_points = _parse_line_benchmarks(benchmarks)

    charts = []
    for ws in sorted(grouped.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        by_lib = defaultdict(list)
        thread_set = set()
        for (lib, threads), time_ns in grouped[ws].items():
            by_lib[lib].append((threads, time_ns))
            thread_set.add(threads)

        traces = []
        # Add serial baseline first (acts as reference point)
        if ws in serial_points:
            max_t = max(thread_set) if thread_set else 1
            traces.append(
                {
                    "name": "serial",
                    "x": [1, max_t],
                    "y": [serial_points[ws], serial_points[ws]],
                    "color": get_color("serial"),
                    "dash": "dash",
                    "baseline": True,
                }
            )

        for lib in sorted(by_lib.keys(), key=_lib_sort_key):
            points = sorted(by_lib[lib])
            trace = {
                "name": lib,
                "x": [p[0] for p in points],
                "y": [p[1] for p in points],
                "color": get_color(lib),
            }
            # In cascading_parallel_for, distinguish blocking vs non-blocking with dashing.
            # Blocking variants (tbb parallel_for, dispenso blocking) get dashed lines;
            # non-blocking variants (tbb_task_group, dispenso cascaded) get solid lines.
            if suite == "cascading_parallel_for":
                ll = lib.lower()
                if ll == "tbb" or "blocking" in ll:
                    trace["dash"] = "dash"
            traces.append(trace)

        label = _format_worksize(ws)
        suite_display = suite.replace("_", " ").title()
        if ws == "default":
            title = suite_display
        else:
            title = f"{suite_display} - {label} Elements"
        chart_cfg = {
            "id": f"{suite}_{ws}",
            "suite": suite,
            "type": "line",
            "title": title,
            "traces": traces,
            "xaxis": "Threads",
            "yaxis_unit": "ns",
        }
        # simple_for has extreme outliers (taskflow ~7000x slower) where
        # auto-zoom hides the comparison entirely.  Show full range so the
        # user can see where taskflow sits, then zoom in interactively.
        if suite == "simple_for":
            chart_cfg["no_auto_zoom"] = True
        charts.append(chart_cfg)
    return charts


_CV_LIB_PREFIXES = ["std_", "deque_", "tbb_", "dispenso_"]
_CV_LIB_MAP = {
    "std": "std::vector",
    "deque": "std::deque",
    "tbb": "tbb::concurrent_vector",
    "dispenso": "dispenso::ConcurrentVector",
}
_CV_LIB_ORDER = [
    "std::vector",
    "std::deque",
    "tbb::concurrent_vector",
    "dispenso::ConcurrentVector",
]


def _parse_concurrent_vector_benchmarks(benchmarks):
    """Parse concurrent_vector benchmarks into (operation, library) -> time."""
    data = {}
    for bm in benchmarks:
        if bm.get("run_type") != "iteration":
            continue
        clean = bm["name"].replace("BM_", "").replace("/real_time", "")
        for prefix in _CV_LIB_PREFIXES:
            if clean.startswith(prefix):
                lib_key = prefix.rstrip("_")
                op = clean[len(prefix) :]
                lib = _CV_LIB_MAP.get(lib_key, lib_key)
                data[(op, lib)] = bm["real_time"]
                break
    return data


def _build_cv_group_chart(data, suite, group_name, ops_list, suffix, title_suffix):
    """Build a single concurrent vector chart for a serial or parallel group."""
    ops = [op for op in ops_list if any((op, lib) in data for lib in _CV_LIB_ORDER)]
    libs = [lib for lib in _CV_LIB_ORDER if any((op, lib) in data for op in ops)]
    if not ops or not libs:
        return None

    group_data = []
    for lib in libs:
        values = [data.get((op, lib), 0) for op in ops]
        group_data.append({"name": lib, "values": values, "color": get_color(lib)})

    label = "Serial/Access" if group_name == "serial" else "Parallel"
    return {
        "id": f"concurrent_vector_{group_name}{suffix}",
        "suite": suite,
        "type": "grouped_bar_h",
        "title": f"Concurrent Vector - {label} Operations{title_suffix}",
        "categories": [op.replace("_", " ") for op in ops],
        "groups": group_data,
        "xaxis_unit": "ns",
        "log_scale": True,
    }


def build_concurrent_vector_charts(benchmarks, suite):
    """Grouped horizontal bar charts split into serial/parallel operations."""
    serial_ops = [
        "push_back_serial",
        "push_back_serial_reserve",
        "iterate",
        "iterate_reverse",
        "lower_bound",
        "index",
        "random",
    ]
    parallel_ops = [
        "parallel",
        "parallel_reserve",
        "parallel_clear",
        "parallel_grow_by_10",
        "parallel_grow_by_100",
        "parallel_grow_by_max",
    ]

    data = _parse_concurrent_vector_benchmarks(benchmarks)
    suffix = "_tcmalloc" if "tcmalloc" in suite else ""
    title_suffix = " (tcmalloc)" if suffix else ""

    charts = []
    for group_name, ops_list in [("serial", serial_ops), ("parallel", parallel_ops)]:
        chart = _build_cv_group_chart(
            data, suite, group_name, ops_list, suffix, title_suffix
        )
        if chart:
            charts.append(chart)
    return charts


def _parse_future_benchmarks(benchmarks, impl_map, sizes):
    """Parse future benchmarks into data dict and error set."""
    data = {}
    error_set = set()
    for bm in benchmarks:
        clean = bm["name"].replace("BM_", "").replace("/real_time", "")
        size = None
        for sz in ["kSmallSize", "kMediumSize", "kLargeSize"]:
            if sz in clean:
                size = sz.replace("k", "").replace("Size", "")
                clean = clean.replace(f"<{sz}>", "")
                break
        impl = impl_map.get(clean, clean.replace("_", " "))
        if bm.get("error_occurred"):
            if size:
                error_set.add((impl, size))
            continue
        if bm.get("run_type") != "iteration":
            continue
        if size:
            data[(impl, size)] = bm["real_time"]
    return data, error_set


def _build_error_aware_groups(impls, available_sizes, data, error_set):
    """Build grouped bar data with error placeholders at max category height."""
    cat_maxes = {}
    for s in available_sizes:
        vals = [data.get((i, s), 0) for i in impls]
        cat_maxes[s] = max(vals) if vals else 0

    groups = []
    for impl in impls:
        values = []
        error_indices = []
        for idx, s in enumerate(available_sizes):
            if (impl, s) in error_set:
                values.append(cat_maxes[s])
                error_indices.append(idx)
            else:
                values.append(data.get((impl, s), 0))
        group = {"name": impl, "values": values, "color": get_color(impl)}
        if error_indices:
            group["error_indices"] = error_indices
        groups.append(group)
    return groups


def build_future_charts(benchmarks, suite):
    """Grouped bar chart by tree size for future benchmark."""
    impl_map = {
        "serial_tree": "serial",
        "std_tree": "std::async",
        "folly_tree": "folly::Future",
        "dispenso_tree": "dispenso::Future",
        "dispenso_tree_when_all": "dispenso::when_all",
        "dispenso_taskset_tree": "dispenso::TaskSet",
        "dispenso_taskset_tree_bulk": "dispenso::TaskSet (bulk)",
    }
    impl_order = [
        "serial",
        "std::async",
        "folly::Future",
        "dispenso::Future",
        "dispenso::TaskSet",
        "dispenso::TaskSet (bulk)",
        "dispenso::when_all",
    ]
    sizes = ["Small", "Medium", "Large"]

    data, error_set = _parse_future_benchmarks(benchmarks, impl_map, sizes)

    impls = [
        i
        for i in impl_order
        if any((i, s) in data or (i, s) in error_set for s in sizes)
    ]
    available_sizes = [
        s for s in sizes if any((i, s) in data or (i, s) in error_set for i in impls)
    ]

    groups = _build_error_aware_groups(impls, available_sizes, data, error_set)

    return [
        {
            "id": "future",
            "suite": suite,
            "type": "grouped_bar_v",
            "title": "Future/Async Tree Build Benchmark",
            "categories": available_sizes,
            "groups": groups,
            "yaxis_unit": "ns",
            "xaxis": "Tree Size",
        }
    ]


def build_graph_charts(benchmarks, suite):
    """Horizontal bar chart with logical grouping for graph/graph_scene."""
    # graph_scene: dispenso apples-to-apples with taskflow first,
    # then partial_revaluation last (unique dispenso feature)
    graph_scene_order = [
        "scene graph parallel for",
        "scene graph taskflow",
        "scene graph concurrent task set",
        "scene graph partial revaluation",
    ]
    graph_order = [
        "taskflow build big tree",
        "build big tree (Graph)",
        "build big tree (BiProp)",
        "build bi prop dependency chain",
        "build bi prop dependency group",
        "build dependency chain (Graph)",
        "build dependency chain (BiProp)",
        "execute dependency chain (Graph)",
        "execute dependency chain (BiProp)",
        "forward propagator node (Graph)",
        "forward propagator node (BiProp)",
    ]
    group_order = graph_scene_order if suite == "graph_scene" else graph_order

    items = []
    for bm in benchmarks:
        if bm.get("run_type") != "iteration":
            continue
        clean = bm["name"].replace("BM_", "").replace("/real_time", "")
        clean = clean.replace("<dispenso::BiPropGraph>", " (BiProp)")
        clean = clean.replace("<dispenso::Graph>", " (Graph)")
        clean = clean.replace("_", " ")
        items.append({"name": clean, "time_ns": bm["real_time"]})

    # Sort by group_order, then alphabetical for unknowns
    def sort_key(item):
        try:
            return (group_order.index(item["name"]), "")
        except ValueError:
            return (len(group_order), item["name"])

    items.sort(key=sort_key)

    def color_for(name):
        if "taskflow" in name.lower():
            return "#f39c12"
        if "BiProp" in name:
            return "#27ae60"
        return "#2ecc71"

    suite_display = suite.replace("_", " ").title()
    return [
        {
            "id": f"{suite}_bar",
            "suite": suite,
            "type": "bar_h_colored",
            "title": f"{suite_display} Benchmark",
            "items": [
                {
                    "name": i["name"],
                    "time_ns": i["time_ns"],
                    "color": color_for(i["name"]),
                }
                for i in items
            ],
            "xaxis_unit": "ns",
        }
    ]


def _parse_rw_lock_benchmarks(benchmarks):
    """Parse rw_lock benchmarks into (operation, mutex, threads, contention) → time."""
    data = {}
    for bm in benchmarks:
        if bm.get("run_type") != "iteration":
            continue
        clean = bm["name"].replace("BM_", "").replace("/real_time", "")
        mutex = None
        for m in ["NopMutex", "std::shared_mutex", "dispenso::RWLock"]:
            if m in clean:
                mutex = m
                clean = clean.replace(f"<{m}>", "")
                break
        parts = clean.split("/")
        operation = parts[0]
        if operation == "serial":
            cont = parts[1] if len(parts) > 1 else None
            data[(operation, mutex, None, cont)] = bm["real_time"]
        else:
            threads = parts[1] if len(parts) > 1 else None
            cont = parts[2] if len(parts) > 2 else None
            data[(operation, mutex, threads, cont)] = bm["real_time"]
    return data


def _build_grouped_bar_chart(chart_id, suite, title, categories, mutexes, data, key_fn):
    """Build a grouped horizontal bar chart config for a set of mutexes."""
    groups = []
    for mutex in mutexes:
        values = [data.get(key_fn(mutex, cat), 0) for cat in categories]
        if any(v > 0 for v in values):
            groups.append({"name": mutex, "values": values, "color": get_color(mutex)})
    if not groups:
        return None
    return {
        "id": chart_id,
        "suite": suite,
        "type": "grouped_bar_h",
        "title": title,
        "categories": [str(c) for c in categories],
        "groups": groups,
        "xaxis_unit": "ns",
    }


def _build_rw_lock_parallel_charts(data, suite, mutexes):
    """Build per-contention parallel RW lock charts."""
    parallel_keys = [k for k in data if k[0] == "parallel"]
    parallel_conts = sorted(set(k[3] for k in parallel_keys if k[3]), key=int)
    parallel_threads = sorted(set(k[2] for k in parallel_keys if k[2]), key=int)
    charts = []
    for cont in parallel_conts:
        chart = _build_grouped_bar_chart(
            f"rw_lock_parallel_{cont}",
            suite,
            f"RW Lock - Parallel ({cont} iterations)",
            parallel_threads,
            mutexes,
            data,
            lambda mutex, t, c=cont: ("parallel", mutex, t, c),
        )
        if chart:
            chart["categories"] = [f"{t} threads" for t in parallel_threads]
            charts.append(chart)
    return charts


def build_rw_lock_charts(benchmarks, suite):
    """Serial + per-contention parallel grouped bar charts."""
    data = _parse_rw_lock_benchmarks(benchmarks)

    mutexes_serial = ["NopMutex", "std::shared_mutex", "dispenso::RWLock"]
    mutexes_parallel = ["std::shared_mutex", "dispenso::RWLock"]
    charts = []

    # Serial chart
    serial_conts = sorted(set(k[3] for k in data if k[0] == "serial" and k[3]), key=int)
    if serial_conts:
        chart = _build_grouped_bar_chart(
            "rw_lock_serial",
            suite,
            "RW Lock - Serial Operations",
            serial_conts,
            mutexes_serial,
            data,
            lambda mutex, c: ("serial", mutex, None, c),
        )
        if chart:
            chart["categories"] = [f"{c} iterations" for c in serial_conts]
            charts.append(chart)

    charts.extend(_build_rw_lock_parallel_charts(data, suite, mutexes_parallel))
    return charts


_OF_SIZE_ORDER = ["kSmallSize", "kMediumSize", "kLargeSize", "kExtraLargeSize"]
_OF_SIZE_LABELS = {
    "kSmallSize": "Small",
    "kMediumSize": "Medium",
    "kLargeSize": "Large",
    "kExtraLargeSize": "Extra Large",
}
_OF_FUNC_COLORS = {
    "Std Function": "#e74c3c",
    "Once Function": "#2ecc71",
    "Inline Function": "#3498db",
}


def _parse_once_function_benchmarks(benchmarks):
    """Parse once_function benchmarks into (op, func_type, size) -> time."""
    data = {}
    for bm in benchmarks:
        if bm.get("run_type") != "iteration":
            continue
        clean = bm["name"].replace("BM_", "").replace("/real_time", "")
        if clean.startswith("move_"):
            op = "move"
            clean = clean[5:]
        elif clean.startswith("queue_"):
            op = "queue"
            clean = clean[6:]
        else:
            continue
        if "<" not in clean:
            continue
        func_type, size = clean.split("<")
        size = size.rstrip(">")
        func_type = func_type.replace("_", " ").title()
        data[(op, func_type, size)] = bm["real_time"]
    return data


def build_once_function_charts(benchmarks, suite):
    """Move vs queue operations, grouped by size."""
    data = _parse_once_function_benchmarks(benchmarks)

    charts = []
    for op in ["move", "queue"]:
        func_types = sorted(set(k[1] for k in data if k[0] == op))
        sizes = [
            s for s in _OF_SIZE_ORDER if any((op, ft, s) in data for ft in func_types)
        ]
        if not sizes or not func_types:
            continue
        groups = []
        for ft in func_types:
            values = [data.get((op, ft, s), 0) for s in sizes]
            groups.append(
                {
                    "name": ft,
                    "values": values,
                    "color": _OF_FUNC_COLORS.get(ft, "#9b59b6"),
                }
            )
        charts.append(
            {
                "id": f"once_function_{op}",
                "suite": suite,
                "type": "grouped_bar_h",
                "title": f"Once Function - {op.title()} Operations",
                "categories": [_OF_SIZE_LABELS.get(s, s) for s in sizes],
                "groups": groups,
                "xaxis_unit": "ns",
            }
        )
    return charts


_ALLOC_PREFIXES = [
    ("mallocfree_threaded", "malloc/free", 19),
    ("mallocfree", "malloc/free", 10),
    ("nl_pool_allocator_arena", "NoLock Arena", 23),
    ("nl_pool_allocator", "NoLock", 17),
    ("pool_allocator_arena", "Arena", 20),
    ("pool_allocator_threaded", "PoolAllocator", 23),
    ("pool_allocator", "PoolAllocator", 14),
]


def _parse_pool_allocator_benchmarks(benchmarks):
    """Parse pool allocator benchmarks into (alloc_type, size, threads, iters) → time."""
    data = {}
    for bm in benchmarks:
        if bm.get("run_type") != "iteration":
            continue
        clean = bm["name"].replace("BM_", "").replace("/real_time", "")
        at = None
        rest = ""
        for prefix, alloc_type, prefix_len in _ALLOC_PREFIXES:
            if clean.startswith(prefix):
                at = alloc_type
                rest = clean[prefix_len:]
                break
        if at is None:
            continue

        size = "unknown"
        threads = 1
        iters = 0
        if "<" in rest:
            params, iters_str = rest.split(">")
            params = params.lstrip("<")
            if "," in params:
                size, threads = params.split(",")
                threads = int(threads)
            else:
                size = params
            if "/" in iters_str:
                iters = int(iters_str.lstrip("/"))
        elif "/" in rest:
            iters = int(rest.lstrip("/"))
        data[(at, size, threads, iters)] = bm["real_time"]
    return data


_PA_SIZE_ORDER = ["kSmallSize", "kMediumSize", "kLargeSize"]
_PA_SIZE_LABELS = {"kSmallSize": "S", "kMediumSize": "M", "kLargeSize": "L"}
_PA_ALLOC_SINGLE = ["malloc/free", "PoolAllocator", "Arena", "NoLock", "NoLock Arena"]
_PA_ALLOC_MULTI = ["malloc/free", "PoolAllocator"]


def _build_pool_allocator_chart(data, suite, tc):
    """Build a single pool allocator chart for a given thread count."""
    if tc == 1:
        alloc_types = _PA_ALLOC_SINGLE
        title_suffix = "Single-threaded"
    else:
        alloc_types = _PA_ALLOC_MULTI
        title_suffix = f"{tc} Threads"

    alloc_types = [
        a for a in alloc_types if any(k[0] == a and k[2] == tc for k in data)
    ]
    iters_list = sorted(set(k[3] for k in data if k[2] == tc))

    categories = []
    for size in _PA_SIZE_ORDER:
        for it in iters_list:
            il = f"{it // 1000}K" if it >= 1000 else str(it)
            categories.append(f"{_PA_SIZE_LABELS.get(size, size)}/{il}")

    groups = []
    for at in alloc_types:
        values = [
            data.get((at, size, tc, it), 0)
            for size in _PA_SIZE_ORDER
            for it in iters_list
        ]
        groups.append({"name": at, "values": values, "color": get_color(at)})

    if not groups:
        return None
    return {
        "id": f"pool_allocator_{tc}t",
        "suite": suite,
        "type": "grouped_bar_h",
        "title": f"Pool Allocator - {title_suffix}",
        "categories": categories,
        "groups": groups,
        "xaxis_unit": "ns",
        "log_scale": True,
    }


def build_pool_allocator_charts(benchmarks, suite):
    """Per-thread-count grouped bar charts."""
    data = _parse_pool_allocator_benchmarks(benchmarks)
    thread_counts = sorted(set(k[2] for k in data))
    charts = []
    for tc in thread_counts:
        chart = _build_pool_allocator_chart(data, suite, tc)
        if chart:
            charts.append(chart)
    return charts


def build_small_buffer_charts(benchmarks, suite):
    """Horizontal bars, grouped side-by-side by allocator type."""
    items = []
    for bm in benchmarks:
        if bm.get("run_type") != "iteration":
            continue
        clean = bm["name"].replace("BM_", "").replace("/real_time", "")
        clean = (
            clean.replace("<kSmallSize>", "[S]")
            .replace("<kMediumSize>", "[M]")
            .replace("<kLargeSize>", "[L]")
        )
        clean = (
            clean.replace("<kSmallSize,", "[S,")
            .replace("<kMediumSize,", "[M,")
            .replace("<kLargeSize,", "[L,")
        )
        clean = clean.replace(">", "]").replace("_", " ")
        is_sb = "small buffer" in clean.lower()
        items.append(
            {
                "name": clean,
                "time_ns": bm["real_time"],
                "color": "#2ecc71" if is_sb else "#e74c3c",
            }
        )

    # Sort by size, iterations, threads, allocator type (to interleave)
    def sort_key(item):
        n = item["name"]
        s = 0 if "[S]" in n or "[S," in n else (1 if "[M]" in n or "[M," in n else 2)
        it_m = re.search(r"/(\d+)", n)
        it = int(it_m.group(1)) if it_m else 0
        th_m = re.search(r"threads:(\d+)", n)
        th = int(th_m.group(1)) if th_m else 1
        is_sb = 1 if "small buffer" in n.lower() else 0
        return (s, it, th, is_sb)

    items.sort(key=sort_key)

    return [
        {
            "id": "small_buffer",
            "suite": suite,
            "type": "bar_h_colored",
            "title": "Small Buffer Allocator Benchmark",
            "items": items,
            "xaxis_unit": "ns",
            "legend": [
                {"name": "SmallBufferAllocator", "color": "#2ecc71"},
                {"name": "new/delete", "color": "#e74c3c"},
            ],
        }
    ]


def _parse_timed_task_benchmark(bm):
    """Parse a single timed_task benchmark entry into a row dict, or None."""
    mean_err = bm.get("mean")
    stddev_err = bm.get("stddev")
    if mean_err is None or stddev_err is None:
        return None
    m = re.match(r"BM_(\w+)<(.+)>", bm["name"])
    if not m:
        return None
    func_name = m.group(1)
    template_args = m.group(2).strip()

    if "mixed" in func_name:
        library = func_name.replace("_mixed", "")
        steady = template_args.strip().lower() == "true"
        config_label = f"Mixed ({'Steady' if steady else 'Normal'})"
    else:
        library = func_name
        parts = [p.strip() for p in template_args.split(",")]
        if len(parts) != 2:
            return None
        try:
            period_ms = int(parts[0])
        except ValueError:
            return None
        steady = parts[1].lower() == "true"
        config_label = f"{period_ms}ms ({'Steady' if steady else 'Normal'})"

    return {
        "library": library,
        "config": config_label,
        "mean_us": mean_err * 1e6,
        "stddev_us": stddev_err * 1e6,
    }


def build_timed_task_charts(benchmarks, suite):
    """Scheduling jitter bar chart with error bars."""
    rows = []
    for bm in benchmarks:
        row = _parse_timed_task_benchmark(bm)
        if row:
            rows.append(row)

    if not rows:
        return []

    # Order configs
    configs = []
    seen = set()
    for r in sorted(rows, key=lambda r: (r["config"].startswith("Mixed"), r["config"])):
        if r["config"] not in seen:
            configs.append(r["config"])
            seen.add(r["config"])

    libraries = sorted(set(r["library"] for r in rows))
    groups = []
    for lib in libraries:
        values = []
        errors = []
        for cfg in configs:
            match = [r for r in rows if r["library"] == lib and r["config"] == cfg]
            if match:
                values.append(match[0]["mean_us"])
                errors.append(match[0]["stddev_us"])
            else:
                values.append(0)
                errors.append(0)
        groups.append(
            {
                "name": lib,
                "values": values,
                "errors": errors,
                "color": get_color(lib),
            }
        )

    return [
        {
            "id": "timed_task",
            "suite": suite,
            "type": "grouped_bar_h_error",
            "title": "Timed Task - Scheduling Jitter",
            "categories": configs,
            "groups": groups,
            "xaxis_label": "Mean Scheduling Error (us)",
        }
    ]


def build_pipeline_charts(benchmarks, suite):
    """Bar chart for pipeline benchmark, grouped by serial vs parallel."""
    items = []
    for bm in benchmarks:
        if bm.get("run_type") != "iteration":
            continue
        clean = bm["name"].replace("BM_", "").replace("/real_time", "")
        clean = clean.replace("_", " ")
        color = get_color(clean.split()[0] if clean.split() else "")
        items.append({"name": clean, "time_ns": bm["real_time"], "color": color})

    # Sort: serial first, then by library within each group
    def sort_key(item):
        n = item["name"].lower()
        is_par = "par" in n or "parallel" in n
        lib_order = 0
        if "serial" in n:
            lib_order = 0
        elif "dispenso" in n:
            lib_order = 1
        elif "tbb" in n:
            lib_order = 2
        elif "taskflow" in n:
            lib_order = 3
        else:
            lib_order = 4
        return (1 if is_par else 0, lib_order, n)

    items.sort(key=sort_key)

    suite_display = suite.replace("_", " ").title()
    return [
        {
            "id": f"{suite}_bar",
            "suite": suite,
            "type": "bar_h_colored",
            "title": f"{suite_display} Benchmark",
            "items": items,
            "xaxis_unit": "ns",
        }
    ]


def _parse_generic_line_benchmarks(benchmarks):
    """Parse benchmarks into thread-scaling data and error tracking.

    Returns (grouped, errored) where:
    - grouped: {worksize: {(lib, threads): real_time}}
    - errored: {worksize: set of libs with error_occurred}
    """
    grouped = defaultdict(dict)
    errored = defaultdict(set)

    for bm in benchmarks:
        if bm.get("run_type") != "iteration":
            continue
        parsed = parse_benchmark_name(bm["name"].replace("BM_", ""))
        lib = parsed["library"]
        threads = parsed.get("threads")
        ws = parsed.get("worksize")
        if threads is not None and ws is not None:
            if bm.get("error_occurred"):
                errored[ws].add(lib)
                continue
            grouped[ws][(lib, threads)] = bm["real_time"]

    return grouped, errored


def _skipped_libs_list(errored_libs, valid_libs):
    """Build skipped_libs chart config for libraries that only have errors."""
    skipped = [lib for lib in sorted(errored_libs) if lib not in valid_libs]
    if not skipped:
        return None
    return [{"name": lib, "color": get_color(lib)} for lib in skipped]


def build_generic_line_charts(benchmarks, suite):
    """Generic line charts for suites with thread/worksize structure."""
    grouped, errored = _parse_generic_line_benchmarks(benchmarks)

    charts = []
    for ws in sorted(grouped.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        by_lib = defaultdict(list)
        for (lib, threads), t in grouped[ws].items():
            by_lib[lib].append((threads, t))

        traces = []
        for lib in sorted(by_lib.keys(), key=_lib_sort_key):
            points = sorted(by_lib[lib])
            trace = {
                "name": lib,
                "x": [p[0] for p in points],
                "y": [p[1] for p in points],
                "color": get_color(lib),
            }
            if "bulk" in lib.lower():
                trace["dash"] = "dash"
            traces.append(trace)

        label = _format_worksize(ws)
        suite_display = suite.replace("_", " ").title()
        chart_cfg = {
            "id": f"{suite}_{ws}",
            "suite": suite,
            "type": "line",
            "title": suite_display
            if ws == "default"
            else f"{suite_display} - {label} Elements",
            "traces": traces,
            "xaxis": "Threads",
            "yaxis_unit": "ns",
        }
        skipped = _skipped_libs_list(errored.get(ws, set()), by_lib)
        if skipped:
            chart_cfg["skipped_libs"] = skipped
        charts.append(chart_cfg)

    # Also generate charts for worksizes where ALL libraries errored
    # (no entries in grouped but present in errored)
    for ws in sorted(errored.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        if ws in grouped:
            continue
        label = _format_worksize(ws)
        suite_display = suite.replace("_", " ").title()
        charts.append(
            {
                "id": f"{suite}_{ws}",
                "suite": suite,
                "type": "line",
                "title": f"{suite_display} - {label} Elements",
                "traces": [],
                "xaxis": "Threads",
                "yaxis_unit": "ns",
                "skipped_libs": [
                    {"name": lib, "color": get_color(lib)}
                    for lib in sorted(errored[ws])
                ],
            }
        )
    return charts


def build_generic_bar_charts(benchmarks, suite):
    """Generic bar chart, items grouped so alternatives for same case are side-by-side."""
    items = []
    for bm in benchmarks:
        if bm.get("run_type") != "iteration":
            continue
        clean = bm["name"].replace("BM_", "").replace("/real_time", "")
        clean_display = clean.replace("_", " ")
        items.append(
            {
                "name": clean_display,
                "time_ns": bm["real_time"],
                "color": get_color(clean_display),
            }
        )

    # Sort: serial first, then group by operation
    def sort_key(item):
        n = item["name"].lower()
        is_par = "par" in n or "parallel" in n
        lib_order = 0
        if "serial" in n:
            lib_order = 0
        elif "dispenso" in n:
            lib_order = 1
        elif "tbb" in n:
            lib_order = 2
        elif "taskflow" in n:
            lib_order = 3
        elif "omp" in n:
            lib_order = 4
        elif "folly" in n:
            lib_order = 5
        elif "std" in n:
            lib_order = 6
        else:
            lib_order = 7
        return (1 if is_par else 0, lib_order, n)

    items.sort(key=sort_key)

    suite_display = suite.replace("_", " ").title()
    return [
        {
            "id": f"{suite}_bar",
            "suite": suite,
            "type": "bar_h_colored",
            "title": f"{suite_display} Benchmark",
            "items": items,
            "xaxis_unit": "ns",
        }
    ]


# ─── Chart dispatch ─────────────────────────────────────────────────────────

SPECIALIZED_LINE = {
    "simple_for",
    "summing_for",
    "trivial_compute",
    "nested_for",
    "cascading_parallel_for",
}


def build_charts_for_suite(benchmarks, suite):
    """Route suite to appropriate chart builder."""
    if suite in SPECIALIZED_LINE:
        return build_line_charts(benchmarks, suite)
    if suite in ("concurrent_vector", "concurrent_vector_tcmalloc"):
        return build_concurrent_vector_charts(benchmarks, suite)
    if suite == "future":
        return build_future_charts(benchmarks, suite)
    if suite in ("graph", "graph_scene"):
        return build_graph_charts(benchmarks, suite)
    if suite == "rw_lock":
        return build_rw_lock_charts(benchmarks, suite)
    if suite == "once_function":
        return build_once_function_charts(benchmarks, suite)
    if suite == "pool_allocator":
        return build_pool_allocator_charts(benchmarks, suite)
    if suite == "small_buffer":
        return build_small_buffer_charts(benchmarks, suite)
    if suite == "timed_task":
        return build_timed_task_charts(benchmarks, suite)
    if suite == "pipeline":
        return build_pipeline_charts(benchmarks, suite)
    # Generic fallback: check if data has thread structure
    has_threads = any(
        parse_benchmark_name(bm["name"].replace("BM_", "")).get("threads") is not None
        for bm in benchmarks
        if bm.get("run_type") == "iteration"
    )
    if has_threads:
        return build_generic_line_charts(benchmarks, suite)
    return build_generic_bar_charts(benchmarks, suite)


# ─── HTML generation ────────────────────────────────────────────────────────


def generate_html(platform_data, output_path):
    """Generate the full interactive HTML page.

    platform_data: list of {"id": str, "label": str, "machine": dict, "charts": list}
    """
    # Build per-platform JSON blob
    platforms_json = json.dumps(platform_data)

    # Collect all suites across platforms for navigation
    all_suites = []
    seen = set()
    for pd_item in platform_data:
        for chart in pd_item["charts"]:
            s = chart["suite"]
            if s not in seen:
                all_suites.append(s)
                seen.add(s)

    nav_items = "\n".join(
        f'<a href="#section-{s}" class="nav-link" data-suite="{s}">'
        f"{s.replace('_', ' ').title()}</a>"
        for s in all_suites
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dispenso Benchmark Results</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {{
  --bg-primary: #0d1117; --bg-secondary: #161b22; --bg-tertiary: #21262d;
  --bg-card: #1c2128; --text-primary: #e6edf3; --text-secondary: #8b949e;
  --text-muted: #6e7681; --border: #30363d; --accent: #2ecc71;
  --accent-hover: #27ae60; --accent-dim: rgba(46,204,113,0.15);
  --shadow: rgba(0,0,0,0.3); --nav-width: 240px; --header-height: 56px;
}}
html[data-theme="light"] {{
  --bg-primary: #ffffff; --bg-secondary: #f6f8fa; --bg-tertiary: #eaeef2;
  --bg-card: #ffffff; --text-primary: #1f2328; --text-secondary: #656d76;
  --text-muted: #8b949e; --border: #d0d7de; --accent: #1a7f37;
  --accent-hover: #116329; --accent-dim: rgba(26,127,55,0.1);
  --shadow: rgba(31,35,40,0.12);
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif; background: var(--bg-primary); color: var(--text-primary); line-height: 1.5; }}
.header {{ position:fixed; top:0; left:0; right:0; height:var(--header-height); background:var(--bg-secondary); border-bottom:1px solid var(--border); display:flex; align-items:center; padding:0 20px; z-index:100; }}
.header-logo {{ display:flex; align-items:center; gap:10px; font-size:18px; font-weight:600; }}
.header-logo .icon {{ width:28px; height:28px; background:var(--accent); border-radius:6px; display:flex; align-items:center; justify-content:center; font-size:15px; color:#fff; font-weight:700; }}
.header-right {{ margin-left:auto; display:flex; align-items:center; gap:12px; }}
.btn {{ background:var(--bg-tertiary); border:1px solid var(--border); color:var(--text-secondary); padding:5px 12px; border-radius:6px; cursor:pointer; font-size:13px; transition:all .2s; }}
.btn:hover {{ color:var(--text-primary); border-color:var(--accent); }}
.btn.active {{ background:var(--accent-dim); border-color:var(--accent); color:var(--accent); font-weight:500; }}
.platform-select {{ background:var(--bg-tertiary); border:1px solid var(--border); color:var(--text-primary); padding:5px 12px; border-radius:6px; font-size:13px; cursor:pointer; }}
.platform-select option {{ background:var(--bg-secondary); color:var(--text-primary); }}
.sidebar {{ position:fixed; top:var(--header-height); left:0; bottom:0; width:var(--nav-width); background:var(--bg-secondary); border-right:1px solid var(--border); overflow-y:auto; padding:12px 0; z-index:50; scrollbar-width:thin; scrollbar-color:var(--border) transparent; }}
.search-box {{ margin:0 14px 12px; }}
.search-input {{ width:100%; padding:6px 10px; background:var(--bg-tertiary); border:1px solid var(--border); border-radius:6px; color:var(--text-primary); font-size:12px; outline:none; }}
.search-input:focus {{ border-color:var(--accent); }}
.search-input::placeholder {{ color:var(--text-muted); }}
.nav-section-title {{ padding:6px 14px; font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:.5px; color:var(--text-muted); }}
.nav-link {{ display:block; padding:5px 14px; color:var(--text-secondary); text-decoration:none; font-size:13px; transition:all .15s; border-left:3px solid transparent; }}
.nav-link:hover {{ color:var(--text-primary); background:var(--bg-tertiary); }}
.nav-link.active {{ color:var(--accent); background:var(--accent-dim); border-left-color:var(--accent); font-weight:500; }}
.main {{ margin-left:var(--nav-width); margin-top:var(--header-height); padding:24px; max-width:1400px; }}
.info-cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(220px,1fr)); gap:12px; margin-bottom:24px; }}
.info-card {{ background:var(--bg-card); border:1px solid var(--border); border-radius:10px; padding:14px; transition:border-color .2s; }}
.info-card:hover {{ border-color:var(--accent); }}
.info-card .label {{ font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:.5px; color:var(--text-muted); margin-bottom:4px; }}
.info-card .value {{ font-size:16px; font-weight:600; }}
.info-card .detail {{ font-size:12px; color:var(--text-secondary); margin-top:2px; }}
.stats-bar {{ display:flex; gap:20px; margin-bottom:24px; padding:12px 16px; background:var(--bg-card); border:1px solid var(--border); border-radius:10px; font-size:13px; }}
.stat-item {{ display:flex; align-items:center; gap:6px; }}
.stat-dot {{ width:7px; height:7px; border-radius:50%; }}
.stat-dot.ok {{ background:#2ecc71; }} .stat-dot.info {{ background:#3498db; }}
.benchmark-section {{ margin-bottom:40px; scroll-margin-top:calc(var(--header-height) + 20px); }}
.section-title {{ font-size:20px; font-weight:600; margin-bottom:16px; padding-bottom:8px; border-bottom:1px solid var(--border); }}
.chart-container {{ background:var(--bg-card); border:1px solid var(--border); border-radius:10px; padding:12px; margin-bottom:12px; transition:border-color .2s; }}
.chart-container:hover {{ border-color:var(--accent); box-shadow:0 2px 8px var(--shadow); }}
.chart {{ width:100%; min-height:400px; }}
.chart .modebar {{ left: 0 !important; right: auto !important; }}
@media (max-width:768px) {{ .sidebar {{ display:none; }} .main {{ margin-left:0; }} }}
::-webkit-scrollbar {{ width:6px; }} ::-webkit-scrollbar-track {{ background:transparent; }}
::-webkit-scrollbar-thumb {{ background:var(--border); border-radius:3px; }}
</style>
</head>
<body>

<header class="header">
  <div class="header-logo"><div class="icon">D</div><span>Dispenso Benchmarks</span></div>
  <div class="header-right">
    <select class="platform-select" id="platformSelect" onchange="switchPlatform(this.value)"></select>
    <button class="btn" onclick="toggleTheme()" id="themeBtn">Light</button>
  </div>
</header>

<nav class="sidebar">
  <div class="search-box"><input type="text" class="search-input" placeholder="Filter..." oninput="filterNav(this.value)"></div>
  <div class="nav-section-title">Benchmark Suites</div>
  {nav_items}
</nav>

<main class="main">
  <div class="info-cards" id="infoCards"></div>
  <div class="stats-bar" id="statsBar"></div>
  <div id="chartSections"></div>
</main>

<script>
const PLATFORMS = {platforms_json};
let currentPlatformIdx = 0;
let currentTheme = 'dark';

// ─── Time unit helpers ─────────────────────────────────────────
function autoUnit(maxNs) {{
  if (maxNs >= 1e9) return {{ scale: 1e9, label: 's' }};
  if (maxNs >= 1e6) return {{ scale: 1e6, label: 'ms' }};
  if (maxNs >= 1e3) return {{ scale: 1e3, label: 'us' }};
  return {{ scale: 1, label: 'ns' }};
}}

// ─── Theme ─────────────────────────────────────────────────────
function themeLayout() {{
  const dark = currentTheme === 'dark';
  return {{
    paper_bgcolor: dark ? '#1c2128' : '#fff',
    plot_bgcolor: dark ? '#0d1117' : '#f6f8fa',
    font: {{ color: dark ? '#e6edf3' : '#1f2328', family: '-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif', size: 12 }},
    xaxis: {{ gridcolor: dark ? '#21262d' : '#eaeef2', zerolinecolor: dark ? '#30363d' : '#d0d7de' }},
    yaxis: {{ gridcolor: dark ? '#21262d' : '#eaeef2', zerolinecolor: dark ? '#30363d' : '#d0d7de' }},
    legend: {{ bgcolor: dark ? 'rgba(22,27,34,0.9)' : 'rgba(255,255,255,0.95)', bordercolor: dark ? '#30363d' : '#d0d7de', borderwidth: 1 }},
    modebar: {{ orientation: 'h', bgcolor: 'transparent', activecolor: dark ? '#58a6ff' : '#0969da' }},
  }};
}}

const plotlyConfig = {{ responsive: true, scrollZoom: true, displayModeBar: true, modeBarButtonsToRemove: ['lasso2d','select2d'], displaylogo: false }};

// Pin the time axis lower bound to zero on any zoom/pan.
// Preserves the zoom span so repeated zoom in/out doesn't drift.
function pinZeroAxis(chartId, axis) {{
  const el = document.getElementById('chart-' + chartId);
  if (!el) return;
  let pinning = false;
  el.on('plotly_relayout', function(ed) {{
    if (pinning) return;
    const key0 = axis + '.range[0]';
    const key1 = axis + '.range[1]';
    // Check if the lower bound moved away from zero
    if (key0 in ed && ed[key0] !== 0) {{
      const lo = ed[key0];
      const hi = key1 in ed ? ed[key1] : el.layout[axis].range[1];
      const span = hi - lo;  // preserve intended zoom level
      pinning = true;
      Plotly.relayout(el, {{ [key0]: 0, [key1]: span }}).then(() => {{ pinning = false; }});
    }}
  }});
}}

// ─── Renderers ─────────────────────────────────────────────────
function renderLine(cfg) {{
  const allY = cfg.traces.flatMap(t => t.baseline ? [] : t.y);  // exclude serial baseline
  const maxY = allY.length > 0 ? Math.max(...allY) : 1;
  const u = autoUnit(maxY);
  const scaledAll = allY.map(v => v / u.scale);

  // Auto-zoom: compare per-trace medians to detect when one trace dominates
  const traceMedians = cfg.traces.filter(t => !t.baseline).map(t => {{
    const sv = [...t.y].filter(v => v > 0).sort((a,b) => a - b);
    return sv.length > 0 ? sv[Math.floor(sv.length / 2)] / u.scale : 0;
  }}).filter(v => v > 0).sort((a,b) => a - b);
  const maxTraceMedian = traceMedians.length > 0 ? traceMedians[traceMedians.length - 1] : 1;
  const minTraceMedian = traceMedians.length > 0 ? traceMedians[0] : 1;
  const autoZoom = !cfg.no_auto_zoom && traceMedians.length >= 2 && maxTraceMedian / minTraceMedian > 5;
  // Zoom to show the competitive traces: 3x the second-highest trace median
  const secondMax = traceMedians.length >= 2 ? traceMedians[traceMedians.length - 2] : maxTraceMedian;
  const zoomMax = secondMax * 3;

  const traces = cfg.traces.map(t => ({{
    x: t.x, y: t.y.map(v => v / u.scale), name: t.name,
    type: 'scatter', mode: t.baseline ? 'lines' : 'lines+markers',
    line: {{ color: t.color, width: t.baseline ? 1.5 : 2.5, dash: t.dash || 'solid' }},
    marker: {{ size: t.baseline ? 0 : 5 }},
    hovertemplate: '%{{x}} threads: %{{y:.4g}} ' + u.label + '<extra>%{{fullData.name}}</extra>',
  }}));

  // Add invisible legend entries for skipped (cannot-complete) libraries
  const skipped = cfg.skipped_libs || [];
  for (const sl of skipped) {{
    traces.push({{
      x: [null], y: [null], name: sl.name + ' (CANNOT COMPLETE)',
      type: 'scatter', mode: 'markers',
      marker: {{ size: 12, color: sl.color, symbol: 'x', line: {{ width: 2, color: sl.color }} }},
      hoverinfo: 'name',
      showlegend: true,
    }});
  }}

  const tl = themeLayout();
  // Add text annotation when libraries are skipped
  const annotations = [];
  if (skipped.length > 0) {{
    const names = skipped.map(s => s.name).join(', ');
    annotations.push({{
      text: names + ': cannot complete (out of memory)',
      xref: 'paper', yref: 'paper', x: 0.5, y: -0.12,
      showarrow: false,
      font: {{ size: 11, color: skipped.length === 1 ? skipped[0].color : '#e74c3c' }},
    }});
  }}
  const layout = {{
    ...tl, title: {{ text: cfg.title + (autoZoom ? ' (auto-zoomed, double-click to reset)' : ''), font: {{ size: 14 }} }},
    xaxis: {{ ...tl.xaxis, title: cfg.xaxis || 'Threads', fixedrange: true }},
    yaxis: {{ ...tl.yaxis, title: 'Time (' + u.label + ')', rangemode: 'tozero',
              ...(autoZoom ? {{ range: [0, zoomMax] }} : {{}}) }},
    legend: {{ ...tl.legend, orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'right', x: 1 }},
    margin: {{ l: 60, r: 20, t: 50, b: skipped.length > 0 ? 70 : 50 }}, hovermode: 'x unified',
    ...(annotations.length > 0 ? {{ annotations }} : {{}}),
  }};
  Plotly.newPlot('chart-' + cfg.id, traces, layout, plotlyConfig);
  pinZeroAxis(cfg.id, 'yaxis');
}}

function renderGroupedBarH(cfg) {{
  const allV = cfg.groups.flatMap(g => g.values);
  const maxV = Math.max(...allV.filter(v => v > 0));
  const u = autoUnit(maxV);
  const useLog = cfg.log_scale && maxV / Math.min(...allV.filter(v => v > 0)) > 50;
  // Reverse so first category is at top
  const cats = [...cfg.categories].reverse();
  const traces = cfg.groups.map(g => ({{
    type: 'bar', orientation: 'h', name: g.name,
    y: cats, x: [...g.values].reverse().map(v => useLog ? v : v / u.scale),
    marker: {{ color: g.color, opacity: 0.85 }},
    hovertemplate: useLog
      ? '%{{y}}: %{{x:.4g}} ns<extra>%{{fullData.name}}</extra>'
      : '%{{y}}: %{{x:.4g}} ' + u.label + '<extra>%{{fullData.name}}</extra>',
  }}));
  const tl = themeLayout();
  const el = document.getElementById('chart-' + cfg.id);
  if (el) el.style.minHeight = Math.max(350, cfg.categories.length * 40 + 100) + 'px';
  Plotly.newPlot('chart-' + cfg.id, traces, {{
    ...tl, barmode: 'group', bargap: 0.15, bargroupgap: 0.1,
    title: {{ text: cfg.title, font: {{ size: 14 }} }},
    xaxis: {{ ...tl.xaxis, title: useLog ? 'Time (ns) - log scale' : 'Time (' + u.label + ')',
              type: useLog ? 'log' : 'linear' }},
    yaxis: {{ ...tl.yaxis, automargin: true, fixedrange: true }},
    legend: {{ ...tl.legend, orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'right', x: 1 }},
    margin: {{ l: 180, r: 20, t: 60, b: 50 }},
  }}, plotlyConfig);
  if (!useLog) pinZeroAxis(cfg.id, 'xaxis');
}}

function renderGroupedBarV(cfg) {{
  const allV = cfg.groups.flatMap(g => g.values);
  const maxV = Math.max(...allV.filter(v => v > 0));
  const u = autoUnit(maxV);

  // Auto-zoom: compare per-group MAX values to detect outlier groups.
  // Using max (not median) ensures we don't clip the tallest competitive bars.
  // Exclude error-placeholder values from zoom calculation.
  const groupMaxes = cfg.groups.map(g => {{
    const errSet = new Set(g.error_indices || []);
    const pos = g.values.filter((v, i) => v > 0 && !errSet.has(i));
    return pos.length > 0 ? Math.max(...pos) : 0;
  }}).filter(v => v > 0).sort((a,b) => a - b);
  const maxGroupMax = groupMaxes.length > 0 ? groupMaxes[groupMaxes.length - 1] : 1;
  const secondGroupMax = groupMaxes.length >= 2 ? groupMaxes[groupMaxes.length - 2] : maxGroupMax;
  const autoZoom = groupMaxes.length >= 2 && maxGroupMax / secondGroupMax > 5;
  const zoomMax = (secondGroupMax * 1.5) / u.scale;

  const traces = cfg.groups.map(g => {{
    const n = g.values.length;
    const errSet = new Set(g.error_indices || []);
    const hasErrors = errSet.size > 0;

    // Per-bar colors and patterns for error bars
    const colors = g.values.map((_, i) => errSet.has(i) ? 'rgba(200,200,200,0.25)' : g.color);
    const patterns = hasErrors ? {{
      shape: g.values.map((_, i) => errSet.has(i) ? '/' : ''),
      fgcolor: g.values.map((_, i) => errSet.has(i) ? '#e74c3c' : g.color),
      solidity: 0.4,
    }} : undefined;
    // Custom hover: show "CANNOT COMPLETE" for error bars
    const customdata = g.values.map((_, i) => errSet.has(i) ? 'CANNOT COMPLETE' : '');
    const hovertemplate = g.values.map((_, i) =>
      errSet.has(i)
        ? '%{{x}}: <b>CANNOT COMPLETE</b><extra>' + g.name + '</extra>'
        : '%{{x}}: %{{y:.4g}} ' + u.label + '<extra>' + g.name + '</extra>'
    );

    return {{
      type: 'bar', name: g.name,
      x: cfg.categories, y: g.values.map(v => v / u.scale),
      marker: {{ color: colors, opacity: 0.85, ...(patterns ? {{ pattern: patterns }} : {{}}) }},
      customdata, hovertemplate,
      // Text label inside error bars
      text: g.values.map((_, i) => errSet.has(i) ? 'CANNOT<br>COMPLETE' : ''),
      textposition: 'inside', insidetextanchor: 'middle', textangle: -90,
      textfont: {{ color: '#e74c3c', size: 11 }},
    }};
  }});
  const tl = themeLayout();
  Plotly.newPlot('chart-' + cfg.id, traces, {{
    ...tl, barmode: 'group', bargap: 0.2,
    title: {{ text: cfg.title + (autoZoom ? ' (auto-zoomed, double-click to reset)' : ''), font: {{ size: 14 }} }},
    xaxis: {{ ...tl.xaxis, title: cfg.xaxis || '', fixedrange: true }},
    yaxis: {{ ...tl.yaxis, title: 'Time (' + u.label + ')', rangemode: 'tozero',
              ...(autoZoom ? {{ range: [0, zoomMax] }} : {{}}) }},
    legend: {{ ...tl.legend, title: {{ text: 'Implementation' }} }},
    margin: {{ l: 60, r: 20, t: 50, b: 50 }},
  }}, plotlyConfig);
  pinZeroAxis(cfg.id, 'yaxis');
}}

function renderBarHColored(cfg) {{
  const maxV = Math.max(...cfg.items.map(i => i.time_ns));
  const minV = Math.min(...cfg.items.filter(i => i.time_ns > 0).map(i => i.time_ns));
  const u = autoUnit(maxV);
  const useLog = maxV / minV > 100;
  // Reverse for top-to-bottom display
  const items = [...cfg.items].reverse();
  const trace = {{
    type: 'bar', orientation: 'h',
    y: items.map(i => i.name), x: items.map(i => useLog ? i.time_ns : i.time_ns / u.scale),
    marker: {{ color: items.map(i => i.color), opacity: 0.85 }},
    hovertemplate: useLog
      ? '%{{y}}: %{{x:.4g}} ns<extra></extra>'
      : '%{{y}}: %{{x:.4g}} ' + u.label + '<extra></extra>',
  }};
  const tl = themeLayout();
  const el = document.getElementById('chart-' + cfg.id);
  if (el) el.style.minHeight = Math.max(350, cfg.items.length * 24 + 100) + 'px';

  const layout = {{
    ...tl, showlegend: false,
    title: {{ text: cfg.title, font: {{ size: 14 }} }},
    xaxis: {{ ...tl.xaxis, title: useLog ? 'Time (ns) - log scale' : 'Time (' + u.label + ')',
              type: useLog ? 'log' : 'linear' }},
    yaxis: {{ ...tl.yaxis, automargin: true, tickfont: {{ size: 10 }}, fixedrange: true }},
    margin: {{ l: 250, r: 20, t: 50, b: 50 }}, bargap: 0.3,
  }};

  // Add manual legend if provided
  if (cfg.legend) {{
    layout.showlegend = true;
    // Add invisible traces for legend
    const traces = [trace];
    cfg.legend.forEach(l => {{
      traces.push({{
        type: 'bar', orientation: 'h', x: [null], y: [null],
        marker: {{ color: l.color }}, name: l.name, showlegend: true,
      }});
    }});
    trace.showlegend = false;
    Plotly.newPlot('chart-' + cfg.id, traces, layout, plotlyConfig);
    if (!useLog) pinZeroAxis(cfg.id, 'xaxis');
    return;
  }}
  Plotly.newPlot('chart-' + cfg.id, [trace], layout, plotlyConfig);
  if (!useLog) pinZeroAxis(cfg.id, 'xaxis');
}}

function renderGroupedBarHError(cfg) {{
  // Reverse for top-to-bottom
  const cats = [...cfg.categories].reverse();
  const traces = cfg.groups.map(g => {{
    const vals = [...g.values].reverse();
    const errs = [...g.errors].reverse();
    // Clamp lower error bars so they don't go below zero (scheduling error can't be negative)
    const errMinus = vals.map((v, i) => Math.min(errs[i], v));
    return {{
      type: 'bar', orientation: 'h', name: g.name,
      y: cats, x: vals,
      error_x: {{ type: 'data', array: errs, arrayminus: errMinus, visible: true }},
      marker: {{ color: g.color, opacity: 0.85 }},
      hovertemplate: '%{{y}}: %{{x:.2f}} +/- %{{error_x.array:.2f}} us<extra>%{{fullData.name}}</extra>',
    }};
  }});
  const tl = themeLayout();
  Plotly.newPlot('chart-' + cfg.id, traces, {{
    ...tl, barmode: 'group', bargap: 0.2,
    title: {{ text: cfg.title, font: {{ size: 14 }} }},
    xaxis: {{ ...tl.xaxis, title: cfg.xaxis_label || 'Mean Error (us)' }},
    yaxis: {{ ...tl.yaxis, automargin: true, fixedrange: true }},
    legend: {{ ...tl.legend }},
    margin: {{ l: 200, r: 20, t: 50, b: 50 }},
  }}, plotlyConfig);
  pinZeroAxis(cfg.id, 'xaxis');
}}

function renderChart(cfg) {{
  switch (cfg.type) {{
    case 'line': renderLine(cfg); break;
    case 'grouped_bar_h': renderGroupedBarH(cfg); break;
    case 'grouped_bar_v': renderGroupedBarV(cfg); break;
    case 'bar_h_colored': renderBarHColored(cfg); break;
    case 'grouped_bar_h_error': renderGroupedBarHError(cfg); break;
  }}
}}

// ─── Platform switching ────────────────────────────────────────
function buildInfoCards(machine) {{
  const compiler = machine.compiler || {{}};
  document.getElementById('infoCards').innerHTML = `
    <div class="info-card"><div class="label">Platform</div><div class="value">${{machine.platform}} ${{machine.architecture||''}}</div><div class="detail">${{machine.platform_release||''}}</div></div>
    <div class="info-card"><div class="label">Processor</div><div class="value">${{machine.cpu_model}}</div><div class="detail">${{machine.cpu_cores}} cores &middot; ${{Math.round(machine.memory_gb||0)}} GB RAM</div></div>
    <div class="info-card"><div class="label">Compiler</div><div class="value">${{compiler.compiler_summary||'Unknown'}}</div><div class="detail">C++${{compiler.cxx_standard||'??'}} &middot; ${{compiler.build_type||''}}</div></div>
    <div class="info-card"><div class="label">Run Date</div><div class="value">${{(machine.timestamp||'').slice(0,10)}}</div><div class="detail">${{(machine.timestamp||'').slice(11,19)}}</div></div>
  `;
}}

function buildSections(charts) {{
  // Group by suite
  const suites = new Map();
  charts.forEach(c => {{
    if (!suites.has(c.suite)) suites.set(c.suite, []);
    suites.get(c.suite).push(c);
  }});

  let html = '';
  suites.forEach((chartList, suite) => {{
    const display = suite.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
    html += `<section id="section-${{suite}}" class="benchmark-section"><h2 class="section-title">${{display}}</h2>`;
    chartList.forEach(c => {{
      html += `<div class="chart-container"><div id="chart-${{c.id}}" class="chart"></div></div>`;
    }});
    html += '</section>';
  }});
  document.getElementById('chartSections').innerHTML = html;
}}

function switchPlatform(idx) {{
  currentPlatformIdx = parseInt(idx);
  const p = PLATFORMS[currentPlatformIdx];
  buildInfoCards(p.machine);
  document.getElementById('statsBar').innerHTML = `
    <div class="stat-item"><span class="stat-dot info"></span><span>Suites: <strong>${{new Set(p.charts.map(c=>c.suite)).size}}</strong></span></div>
    <div class="stat-item"><span class="stat-dot ok"></span><span>Charts: <strong>${{p.charts.length}}</strong></span></div>
  `;
  buildSections(p.charts);
  // Small delay so DOM is ready
  requestAnimationFrame(() => {{
    p.charts.forEach(renderChart);
    observeSections();
  }});
}}

function toggleTheme() {{
  currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', currentTheme);
  document.getElementById('themeBtn').textContent = currentTheme === 'dark' ? 'Light' : 'Dark';
  const p = PLATFORMS[currentPlatformIdx];
  p.charts.forEach(renderChart);
}}

function filterNav(q) {{
  q = q.toLowerCase();
  document.querySelectorAll('.nav-link').forEach(l => {{
    l.style.display = l.textContent.toLowerCase().includes(q) ? 'block' : 'none';
  }});
}}

let observer;
function observeSections() {{
  if (observer) observer.disconnect();
  observer = new IntersectionObserver(entries => {{
    entries.forEach(e => {{
      if (e.isIntersecting) {{
        const id = e.target.id.replace('section-', '');
        document.querySelectorAll('.nav-link').forEach(l => {{
          l.classList.toggle('active', l.dataset.suite === id);
        }});
      }}
    }});
  }}, {{ rootMargin: '-20% 0px -70% 0px' }});
  document.querySelectorAll('.benchmark-section').forEach(s => observer.observe(s));
}}

// ─── Init ──────────────────────────────────────────────────────
const sel = document.getElementById('platformSelect');
PLATFORMS.forEach((p, i) => {{
  const opt = document.createElement('option');
  opt.value = i; opt.textContent = p.label;
  sel.appendChild(opt);
}});
switchPlatform(0);
</script>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Generated: {output_path}")


# ─── Main ───────────────────────────────────────────────────────────────────


def load_platform(json_path):
    """Load a JSON file and build chart configs."""
    with open(json_path) as f:
        data = json.load(f)

    machine = data["machine_info"]
    platform_id = machine.get("platform_id", Path(json_path).stem)
    label = (
        f"{machine.get('cpu_model', 'Unknown')} ({machine.get('cpu_cores', '?')} cores)"
    )

    charts = []
    for result in data["results"]:
        if not result.get("success") or "data" not in result:
            continue
        suite = result["name"].replace(".exe", "").replace("_benchmark", "")
        benchmarks = result["data"].get("benchmarks", [])
        charts.extend(build_charts_for_suite(benchmarks, suite))

    return {
        "id": platform_id,
        "label": label,
        "machine": machine,
        "charts": charts,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive benchmark dashboard"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark JSON file(s)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("/tmp/dispenso-benchmarks.html"),
        help="Output HTML file",
    )
    args = parser.parse_args()

    platforms = []
    for path in args.inputs:
        print(f"Loading: {path}")
        platforms.append(load_platform(path))
        print(
            f"  Platform: {platforms[-1]['label']}, Charts: {len(platforms[-1]['charts'])}"
        )

    generate_html(platforms, args.output)
    print(f"\nTotal platforms: {len(platforms)}")


if __name__ == "__main__":
    main()
