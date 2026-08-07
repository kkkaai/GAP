#!/usr/bin/env python3
"""Generate a Markdown report and 4xN visual index for the three baselines."""

from __future__ import annotations

import argparse
import html
import json
import os
import statistics as stats
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = REPO_ROOT / "outputs" / "runs_random10_three_baselines"
DEFAULT_REPORT = REPO_ROOT / "docs" / "retargeting_baselines_report.md"


ROUTE_NAMES = {
    "position_only": "指尖位置匹配",
    "position_force": "指尖位置匹配 + 力闭合引导",
    "contact_heatmap": "接触热力图匹配",
}


def rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return os.path.relpath(path.resolve(), base.resolve())


def md_link(label: str, path: Path, base: Path) -> str:
    return f"[{label}]({rel(path, base)})"


def route_metric(entry: dict[str, Any]) -> str:
    route = entry["route"]
    if route == "position_only":
        return f"tip {entry.get('mean_fingertip_error_mm', 0.0):.2f} mm"
    if route == "position_force":
        fc = "FC" if entry.get("strict_force_closure") else "no FC"
        return (
            f"tip {entry.get('mean_fingertip_error_mm', 0.0):.2f} mm / "
            f"contact {entry.get('mean_contact_error_mm', 0.0):.2f} mm / {fc}"
        )
    if route == "contact_heatmap":
        return (
            f"MSE {entry.get('heatmap_mse', 0.0):.5f} / "
            f"high-dist {entry.get('mean_high_contact_distance_mm', 0.0):.2f} mm / "
            f"tip {entry.get('mean_fingertip_error_mm', 0.0):.2f} mm"
        )
    return ""


def load_summary(run_root: Path) -> list[dict[str, Any]]:
    return json.loads((run_root / "summary.json").read_text(encoding="utf-8"))


def group_by_frame(summary: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for entry in summary:
        grouped[entry["frame_id"]][entry["route"]] = entry
    return dict(grouped)


def mean(values: list[float]) -> float:
    return float(stats.mean(values)) if values else 0.0


def median(values: list[float]) -> float:
    return float(stats.median(values)) if values else 0.0


def route_stats(summary: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| baseline | runs | mean tip error | median tip error | extra metric |",
        "|---|---:|---:|---:|---|",
    ]
    for route in ("position_only", "position_force", "contact_heatmap"):
        rows = [entry for entry in summary if entry["route"] == route]
        tips = [float(entry["mean_fingertip_error_mm"]) for entry in rows if entry.get("mean_fingertip_error_mm") is not None]
        extra = ""
        if route == "position_force":
            contacts = [float(entry["mean_contact_error_mm"]) for entry in rows if entry.get("mean_contact_error_mm") is not None]
            fc_count = sum(1 for entry in rows if entry.get("strict_force_closure"))
            extra = f"mean contact {mean(contacts):.2f} mm, strict FC {fc_count}/{len(rows)}"
        elif route == "contact_heatmap":
            mses = [float(entry["heatmap_mse"]) for entry in rows if entry.get("heatmap_mse") is not None]
            high_dist = [
                float(entry["mean_high_contact_distance_mm"])
                for entry in rows
                if entry.get("mean_high_contact_distance_mm") is not None
            ]
            extra = f"mean heatmap MSE {mean(mses):.5f}, high-contact dist {mean(high_dist):.2f} mm"
        lines.append(
            f"| {ROUTE_NAMES[route]} | {len(rows)} | {mean(tips):.2f} mm | {median(tips):.2f} mm | {extra} |"
        )
    return lines


def representative_frames(grouped: dict[str, dict[str, dict[str, Any]]]) -> list[tuple[str, str]]:
    complete = [
        frame_id
        for frame_id, routes in grouped.items()
        if all(route in routes for route in ("position_only", "position_force", "contact_heatmap"))
    ]
    if not complete:
        return []
    by_force = sorted(complete, key=lambda frame: grouped[frame]["position_force"].get("mean_fingertip_error_mm", 1e9))
    by_heat = sorted(complete, key=lambda frame: grouped[frame]["contact_heatmap"].get("heatmap_mse", 1e9))
    picks = [
        ("position-force fingertip best", by_force[0]),
        ("position-force fingertip median", by_force[len(by_force) // 2]),
        ("contact-heatmap MSE best", by_heat[0]),
        ("contact-heatmap MSE worst", by_heat[-1]),
    ]
    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label, frame in picks:
        if frame not in seen:
            deduped.append((label, frame))
            seen.add(frame)
    return deduped


def frame_table(grouped: dict[str, dict[str, dict[str, Any]]], report_base: Path) -> list[str]:
    lines = [
        "| frame | original | position-only | position-force | contact-heatmap |",
        "|---|---|---|---|---|",
    ]
    for frame_id in sorted(grouped):
        routes = grouped[frame_id]
        frame_dir = Path(routes[next(iter(routes))]["result_json"]).parents[1]
        original = frame_dir / "original.jpg"
        cells = [f"`{frame_id}`", f'<img src="{rel(original, report_base)}" width="180">']
        for route in ("position_only", "position_force", "contact_heatmap"):
            entry = routes.get(route)
            if not entry:
                cells.append("missing")
                continue
            label = route_metric(entry)
            cells.append(f"{md_link(label, Path(entry['scene_html']), report_base)}")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def representative_table(
    grouped: dict[str, dict[str, dict[str, Any]]],
    picks: list[tuple[str, str]],
    report_base: Path,
) -> list[str]:
    lines = [
        "| reason | frame | original | position-only | position-force | contact-heatmap |",
        "|---|---|---|---|---|---|",
    ]
    for reason, frame_id in picks:
        routes = grouped[frame_id]
        frame_dir = Path(routes[next(iter(routes))]["result_json"]).parents[1]
        original = frame_dir / "original.jpg"
        cells = [reason, f"`{frame_id}`", f'<img src="{rel(original, report_base)}" width="220">']
        for route in ("position_only", "position_force", "contact_heatmap"):
            entry = routes[route]
            cells.append(md_link(route_metric(entry), Path(entry["scene_html"]), report_base))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_visual_index(run_root: Path, grouped: dict[str, dict[str, dict[str, Any]]]) -> Path:
    rows = []
    for frame_id in sorted(grouped):
        routes = grouped[frame_id]
        frame_dir = Path(routes[next(iter(routes))]["result_json"]).parents[1]
        original = frame_dir / "original.jpg"
        cells = [
            f'<div class="cell original"><img src="{html.escape(rel(original, run_root))}" alt="original"></div>',
        ]
        for route in ("position_only", "position_force", "contact_heatmap"):
            entry = routes.get(route)
            if entry:
                src = html.escape(rel(Path(entry["scene_html"]), run_root))
                cells.append(f'<div class="cell"><iframe src="{src}" loading="lazy"></iframe></div>')
            else:
                cells.append('<div class="cell missing">missing</div>')
        rows.append(
            f"""
      <section class="frame">
        <h2>{html.escape(frame_id)}</h2>
        <div class="grid">
          {''.join(cells)}
        </div>
      </section>"""
        )

    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Random10 Three Baselines Visual Index</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f6f7f9; color: #1f2937; }}
    header {{ position: sticky; top: 0; z-index: 2; background: #fff; border-bottom: 1px solid #d9dee8; padding: 12px 16px; }}
    h1 {{ margin: 0; font-size: 20px; }}
    .labels, .grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 8px; }}
    .labels {{ margin-top: 8px; color: #475569; font-size: 13px; font-weight: 700; }}
    .frame {{ padding: 14px 16px 22px; border-bottom: 1px solid #e1e5ec; }}
    h2 {{ margin: 0 0 8px; font-size: 14px; font-weight: 700; color: #334155; }}
    .cell {{ height: 360px; background: #fff; border: 1px solid #d8dee9; overflow: hidden; }}
    .original {{ display: flex; align-items: center; justify-content: center; }}
    .original img {{ width: 100%; height: 100%; object-fit: contain; }}
    iframe {{ width: 100%; height: 100%; border: 0; background: #fff; }}
    .missing {{ display: flex; align-items: center; justify-content: center; color: #64748b; }}
  </style>
</head>
<body>
  <header>
    <h1>Random10 Three Baselines Visual Index</h1>
    <div class="labels"><div>Original RGB</div><div>Position-only</div><div>Position-force</div><div>Contact-heatmap</div></div>
  </header>
  {''.join(rows)}
</body>
</html>
"""
    output = run_root / "visual_index.html"
    output.write_text(content, encoding="utf-8")
    return output


def write_report(report: Path, run_root: Path, visual_index: Path, summary: list[dict[str, Any]]) -> None:
    grouped = group_by_frame(summary)
    picks = representative_frames(grouped)
    base = report.parent
    lines: list[str] = [
        "# 机械手动作重定向三条基线记录",
        "",
        "本文档记录当前项目中的三条 retargeting baseline：实现入口、输入输出、loss 组成、关键权重，以及 random10 批量实验结果。",
        "",
        "## 代码入口",
        "",
        "| baseline | script | route |",
        "|---|---|---|",
        f"| 指尖位置匹配 | `{rel(REPO_ROOT / 'tools/retarget/position.py', REPO_ROOT)}` | `position_only_fingertips` |",
        f"| 指尖位置匹配 + 力闭合引导 | `{rel(REPO_ROOT / 'tools/retarget/position_force.py', REPO_ROOT)}` | `position_force_closure` |",
        f"| 接触热力图匹配 | `{rel(REPO_ROOT / 'tools/retarget/contact_heatmap.py', REPO_ROOT)}` | `contact_heatmap_matching` |",
        f"| 批量实验脚本 | `{rel(REPO_ROOT / 'tools/retarget/batch_three_baselines.py', REPO_ROOT)}` | random10 x 3 baselines |",
        "",
        "## 输入输出",
        "",
        "三条路线使用同一组输入：HOI4D sequence、frame、RGB、CAD mesh、hand pickle、MANO root、robot profile。当前 random10 实验统一使用 `folding_hand_right`。",
        "",
        "每个 frame 的输出目录结构如下：",
        "",
        "```text",
        "outputs/runs_random10_three_baselines/",
        "  random_..._frameXXXXX/",
        "    original.jpg",
        "    frame_index.json",
        "    position_only/",
        "    position_force/",
        "    contact_heatmap/",
        "```",
        "",
        "每条 baseline 子目录下保存对应的 `retargeted_*.json` 和 `scene_*.html`。",
        "",
        "## Baseline 1: 指尖位置匹配",
        "",
        "实现路径：`tools/retarget/position.py` 调用 `Phase6ProstheticAction`。该路线从 MANO/HOI4D hand annotation 构造 `Phase5ManoResult`，提取五个 MANO fingertip 作为目标，再优化机器人 action，使机器人五个 fingertip 在 wrist frame 中匹配目标 fingertip。",
        "",
        "主要目标：",
        "",
        "```text",
        "L = mean_i || tip_robot_i(q) - tip_mano_i ||^2 + regularization",
        "```",
        "",
        "random10 批量参数：`optimization_restarts=2`，`max_nfev=60`。",
        "",
        "可视化对齐：当前已统一为 `rigid only, no scale`，与 Baseline 2/3 保持一致；旧的 `scale + rigid` 显示可通过 `position.py --scale-visualization` 手动开启。",
        "",
        "## Baseline 2: 指尖位置匹配 + 力闭合引导",
        "",
        "实现路径：`tools/retarget/position_force.py`。该路线先运行 Baseline 1 得到 action 初始化，再从初始机器人 fingertip 投影到物体表面生成 contact target；随后联合优化 wrist pose delta 和 action。",
        "",
        "当前 loss 组成：",
        "",
        "```text",
        "L = 25.0 * L_fingertip",
        "  + contact_weight * L_contact",
        "  + surface_attraction_weight * L_surface",
        "  + normal_weight * L_normal",
        "  + fc_weight * L_dex_fc",
        "  + penetration_weight * L_penetration",
        "  + regularization_weight * (L_pose_reg + L_joint_reg)",
        "```",
        "",
        "默认权重：`contact_weight=10.0`，`surface_attraction_weight=80.0`，`normal_weight=0.05`，`fc_weight=0.05`，`penetration_weight=120.0`，`regularization_weight=0.005`。",
        "",
        "random10 批量参数：`position_restarts=2`，`position_max_nfev=60`，`num_robot_samples=900`，`stage2_maxiter=80`。",
        "",
        "## Baseline 3: 接触热力图匹配",
        "",
        "实现路径：`tools/retarget/contact_heatmap.py`。该路线参考 GenDexGrasp 的 object-centric contact map 思路，在物体表面采样点，先由 MANO hand surface 生成目标 contact heatmap，再优化机器人 wrist pose delta 和 action，使机器人 surface 诱导出的物体表面 heatmap 接近目标 heatmap。",
        "",
        "热力图定义：",
        "",
        "```text",
        "aligned_distance(o, H) = nearest_distance(o, H) * exp(gamma * (1 - dot(direction_to_hand, object_normal)))",
        "heatmap(o) = exp(-(aligned_distance(o, H) / sigma)^2)",
        "```",
        "",
        "当前 loss 组成：",
        "",
        "```text",
        "L = heatmap_weight * L_heatmap_mse",
        "  + surface_attraction_weight * L_high_contact_surface",
        "  + penetration_weight * L_penetration",
        "  + fingertip_prior_weight * L_fingertip_prior",
        "  + fc_weight * L_dex_fc",
        "  + regularization_weight * (L_pose_reg + L_joint_reg)",
        "```",
        "",
        "默认权重：`heatmap_weight=30.0`，`high_contact_weight=6.0`，`surface_attraction_weight=75.0`，`penetration_weight=120.0`，`fingertip_prior_weight=8.0`，`fc_weight=0.02`，`regularization_weight=0.005`。",
        "",
        "random10 批量参数：`position_restarts=2`，`position_max_nfev=60`，`num_object_samples=256`，`num_mano_samples=900`，`num_robot_samples=520`，`maxiter=25`，`fingertip_prior_weight=20`。",
        "",
        "## Random10 实验结果",
        "",
        f"结果目录：{md_link(rel(run_root, REPO_ROOT), run_root, base)}",
        "",
        f"4x10 可视化总览：{md_link('visual_index.html', visual_index, base)}",
        "",
        *route_stats(summary),
        "",
        "## 代表性帧",
        "",
        *representative_table(grouped, picks, base),
        "",
        "## 10 帧总览",
        "",
        "下面的表格嵌入原图，并链接到三条 baseline 的交互式 HTML。真正的 4×10 同屏查看请打开上面的 `visual_index.html`。",
        "",
        *frame_table(grouped, base),
        "",
        "## 当前结论",
        "",
        "- `position_only` 是最稳定、最轻量的动作初始化，但只关心五个 fingertip，无法显式约束物体表面接触。",
        "- `position_force` 在 random10 中 fingertip error 最低，并且 6/10 达到 strict force closure；它适合作为当前最强工程 baseline。",
        "- `contact_heatmap` 的优势是优化接触分布，而不是单个指尖/投影点；当前 fingertip error 介于两者之间，后续值得继续调 `fingertip_prior_weight`、高热区采样策略和 penetration/normal 项。",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    summary = load_summary(args.run_root)
    grouped = group_by_frame(summary)
    visual_index = write_visual_index(args.run_root, grouped)
    write_report(args.report, args.run_root, visual_index, summary)
    print(f"Wrote {args.report}")
    print(f"Wrote {visual_index}")


if __name__ == "__main__":
    main()
