"""Score Phase 2 answers and generate the final comparison report."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    EXP,
    HARD_CAP_CNY,
    HARD_CAP_USD,
    PRICE_CACHE_HIT,
    PRICE_CACHE_MISS,
    PRICE_OUTPUT,
    QUERY,
    SCENARIO_DATE_LABEL,
    USD_CNY,
    classify_answer,
    cost_usd,
    read_json,
    write_json,
)

ROOT = Path(r"G:\deepseek_ragproject_workspace")
REPORT_PATH = ROOT / "EXPERIMENT_FINAL_COMPARISON.md"
USAGE_CSV = ROOT / "EXPERIMENT_USAGE.csv"
ARMS = ("A0", "A1", "B")


def latest_run() -> Path:
    runs = sorted(
        (EXP / "runs").glob("phase2-*"),
        key=lambda p: p.stat().st_mtime,
    )
    if not runs:
        raise SystemExit("no phase2 run directory found")
    return runs[-1]


def load_calls(run_dir: Path) -> list[dict[str, Any]]:
    calls = []
    for path in sorted(run_dir.glob("*.json")):
        if path.name in {"summary.json", "manual_labels.json"}:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "arm" in data and "trial" in data:
            calls.append(data)
    return calls


def final_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {"A0": "full", "A1": "reduce", "B": "generate"}
    result = [
        call for call in calls if call["step"] == wanted.get(call["arm"])
    ]
    result.sort(key=lambda c: (ARMS.index(c["arm"]), c["trial"]))
    return result


def aggregate(calls: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        by_arm[call["arm"]].append(call)
    stats: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        arm_calls = by_arm.get(arm, [])
        trials = sorted({c["trial"] for c in arm_calls})
        trial_rows = []
        for trial in trials:
            rows = [c for c in arm_calls if c["trial"] == trial]
            cost = sum((Decimal(str(c["cost_usd"])) for c in rows), Decimal("0"))
            trial_rows.append(
                {
                    "trial": trial,
                    "calls": len(rows),
                    "input_tokens": sum(int(c["usage"].get("prompt_tokens", 0)) for c in rows),
                    "output_tokens": sum(int(c["usage"].get("completion_tokens", 0)) for c in rows),
                    "cache_hit_tokens": sum(
                        int(c["usage"].get("prompt_cache_hit_tokens", 0) or 0)
                        for c in rows
                    ),
                    "cache_miss_tokens": sum(
                        int(c["usage"].get("prompt_cache_miss_tokens", 0) or 0)
                        for c in rows
                    ),
                    "cost_usd": str(cost),
                    "cost_cny": str((cost * USD_CNY).quantize(Decimal("0.0001"))),
                    "latency_ms": round(sum(float(c["latency_ms"]) for c in rows), 2),
                }
            )
        stats[arm] = {
            "trials": trial_rows,
            "calls": len(arm_calls),
            "cost_usd": str(
                sum((Decimal(str(c["cost_usd"])) for c in arm_calls), Decimal("0"))
            ),
            "cost_cny": str(
                (
                    sum((Decimal(str(c["cost_usd"])) for c in arm_calls), Decimal("0"))
                    * USD_CNY
                ).quantize(Decimal("0.0001"))
            ),
            "median_latency_ms": round(
                statistics.median([t["latency_ms"] for t in trial_rows]), 2
            )
            if trial_rows
            else 0.0,
            "median_cost_usd": str(
                statistics.median([Decimal(str(t["cost_usd"])) for t in trial_rows])
            )
            if trial_rows
            else "0",
        }
    return stats


def category_for(
    key: str,
    answer: str,
    manual: dict[str, Any],
    context: str,
) -> tuple[str, str]:
    auto = classify_answer(answer, context)
    if key in manual:
        entry = manual[key]
        if isinstance(entry, str):
            return entry, auto
        return str(entry.get("category", auto)), auto
    return auto, auto


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--phase1", type=str, default="")
    args = parser.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else latest_run()
    calls = load_calls(run_dir)
    finals = final_calls(calls)
    if len(finals) != 9:
        raise SystemExit(f"expected 9 final answers, got {len(finals)}")

    manual_path = run_dir / "manual_labels.json"
    manual = read_json(manual_path) if manual_path.is_file() else {}

    phase0 = read_json(EXP / "phase0_report.json")
    phase1 = read_json(
        Path(args.phase1) if args.phase1 else EXP / "phase1_report.json"
    )
    summary = read_json(run_dir / "summary.json")

    answer_rows = []
    for call in finals:
        key = f"{call['arm']}-{call['trial']}"
        context = call.get("prompt", "")
        category, auto = category_for(key, call["answer"], manual, context)
        answer_rows.append(
            {
                "key": key,
                "arm": call["arm"],
                "trial": call["trial"],
                "category": category,
                "auto_category": auto,
                "answer": call["answer"],
                "cost_usd": call["cost_usd"],
                "latency_ms": call["latency_ms"],
                "usage": call["usage"],
                "finish_reason": call["finish_reason"],
            }
        )

    # Add manual rationale where supplied.
    for row in answer_rows:
        entry = manual.get(row["key"])
        if isinstance(entry, dict):
            row["rationale"] = str(entry.get("rationale", ""))
        else:
            row["rationale"] = ""

    by_arm_category: dict[str, dict[str, int]] = {}
    for arm in ARMS:
        rows = [r for r in answer_rows if r["arm"] == arm]
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            counts[row["category"]] += 1
        by_arm_category[arm] = dict(counts)

    strict = {
        arm: sum(1 for r in answer_rows if r["arm"] == arm and r["category"] == "CORRECT_LATEST")
        for arm in ARMS
    }
    lenient = {
        arm: sum(
            1
            for r in answer_rows
            if r["arm"] == arm and r["category"] in {"CORRECT_LATEST", "PARTIAL_LATEST"}
        )
        for arm in ARMS
    }

    stats = aggregate(calls)
    total_cost = sum(
        (Decimal(str(c["cost_usd"])) for c in calls),
        Decimal("0"),
    )
    total_cny = (total_cost * USD_CNY).quantize(Decimal("0.0001"))
    total_input = sum(int(c["usage"].get("prompt_tokens", 0)) for c in calls)
    total_output = sum(int(c["usage"].get("completion_tokens", 0)) for c in calls)
    total_hit = sum(int(c["usage"].get("prompt_cache_hit_tokens", 0) or 0) for c in calls)
    total_miss = sum(int(c["usage"].get("prompt_cache_miss_tokens", 0) or 0) for c in calls)

    # Request-level CSV.
    with USAGE_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "arm",
                "trial",
                "step",
                "prompt_tokens",
                "completion_tokens",
                "cache_hit_tokens",
                "cache_miss_tokens",
                "cost_usd",
                "cost_cny",
                "latency_ms",
                "finish_reason",
                "response_id",
                "requested_at_cst",
                "answer",
            ],
        )
        writer.writeheader()
        for call in calls:
            writer.writerow(
                {
                    "arm": call["arm"],
                    "trial": call["trial"],
                    "step": call["step"],
                    "prompt_tokens": call["usage"].get("prompt_tokens"),
                    "completion_tokens": call["usage"].get("completion_tokens"),
                    "cache_hit_tokens": call["usage"].get("prompt_cache_hit_tokens"),
                    "cache_miss_tokens": call["usage"].get("prompt_cache_miss_tokens"),
                    "cost_usd": call["cost_usd"],
                    "cost_cny": call.get("cost_cny"),
                    "latency_ms": call["latency_ms"],
                    "finish_reason": call["finish_reason"],
                    "response_id": call.get("response_id"),
                    "requested_at_cst": call.get("requested_at_cst"),
                    "answer": call["answer"].replace("\n", " "),
                }
            )

    write_json(
        run_dir / "scores.json",
        {
            "run_dir": str(run_dir),
            "total_cost_usd": str(total_cost),
            "total_cost_cny": str(total_cny),
            "stats": stats,
            "answers": answer_rows,
            "strict": strict,
            "lenient": lenient,
        },
    )

    # Build report.
    lines: list[str] = []
    add = lines.append
    add("# A/B 实验最终对比报告：100k 记忆检索 vs. 1M 上下文裸跑")
    add("")
    add("> **实验代号**：`AB-100K-KEY-2026`  ")
    add("> **场景日期**：`2026-10-28`（冻结；“上周二”= 2026-10-20）  ")
    add(f"> **模型**：`deepseek-flash`（DeepSeek-V4.1-Flash-1017）  ")
    add(f"> **运行目录**：`{run_dir}`  ")
    add(f"> **总请求数**：{len(calls)}  ")
    add(f"> **总实际花费**：**${total_cost:.4f}**（约 **¥{total_cny:.2f}**）  ")
    add(
        f"> **预算**：${HARD_CAP_USD} / ¥{HARD_CAP_CNY}，"
        f"剩余 ${HARD_CAP_USD - total_cost:.4f}"
    )
    if any(call.get("architecture") == "RRF+Flat" for call in calls):
        add("")
        add(
            "> **架构说明：B组数据已基于去除了硬编码的 RRF + Flat 混合检索架构重新测试，"
            "A0/A1 仍沿用历史数据。**"
        )
    add("")

    add("## 1. 结论摘要")
    add("")
    add("### 1.1 一句话结论")
    add("")
    b_strict = strict["B"]
    a1_strict = strict["A1"]
    a0_strict = strict["A0"]
    if b_strict == 3 and a0_strict == 0:
        add(
            f"> 在 10 万条 / 约 2.58M Token 的设定下，B（RRF+Flat Top-3）"
            f"严格正确 **{b_strict}/3**，A0（1M 截断裸跑）严格正确 **{a0_strict}/3**；"
            f"B 的 Prompt Token 约为 A0 的 **1/{max(1, round(stats['A0']['trials'][0]['input_tokens'] / max(1, stats['B']['trials'][0]['input_tokens'])))}**。"
        )
    else:
        add(
            f"> B 严格正确 **{b_strict}/3**，A1 严格正确 **{a1_strict}/3**，"
            f"A0 严格正确 **{a0_strict}/3**。结论以第 6 节为准。"
        )
    add("")

    add("### 1.2 核心数字")
    add("")
    add("| 指标 | A0 截断裸跑 | A1 分片全量 | B RRF+Flat Top-3 |")
    add("|---|---:|---:|---:|")
    add(
        f"| API 调用数 | {stats['A0']['calls']} | {stats['A1']['calls']} | {stats['B']['calls']} |"
    )
    add(
        f"| Input Token | {sum(t['input_tokens'] for t in stats['A0']['trials']):,} | "
        f"{sum(t['input_tokens'] for t in stats['A1']['trials']):,} | "
        f"{sum(t['input_tokens'] for t in stats['B']['trials']):,} |"
    )
    add(
        f"| Output Token | {sum(t['output_tokens'] for t in stats['A0']['trials']):,} | "
        f"{sum(t['output_tokens'] for t in stats['A1']['trials']):,} | "
        f"{sum(t['output_tokens'] for t in stats['B']['trials']):,} |"
    )
    add(
        f"| 总成本 USD | ${Decimal(stats['A0']['cost_usd']):.6f} | "
        f"${Decimal(stats['A1']['cost_usd']):.6f} | "
        f"${Decimal(stats['B']['cost_usd']):.6f} |"
    )
    add(
        f"| 总成本 CNY | ¥{Decimal(stats['A0']['cost_cny']):.4f} | "
        f"¥{Decimal(stats['A1']['cost_cny']):.4f} | "
        f"¥{Decimal(stats['B']['cost_cny']):.4f} |"
    )
    add(
        f"| 端到端中位耗时 | {stats['A0']['median_latency_ms']:.0f} ms | "
        f"{stats['A1']['median_latency_ms']:.0f} ms | "
        f"{stats['B']['median_latency_ms']:.0f} ms |"
    )
    add(f"| 严格正确率 | {strict['A0']}/3 | {strict['A1']}/3 | {strict['B']}/3 |")
    add(f"| 宽松正确率 | {lenient['A0']}/3 | {lenient['A1']}/3 | {lenient['B']}/3 |")
    add("")

    add("## 2. 实验设置")
    add("")
    add("| 项 | 值 |")
    add("|---|---|")
    add(f"| 语料 | 100,000 条，2,576,079 Token（官方 tokenizer） |")
    add(f"| T1 | 20,000 · 2026-07-15 · 玄关鞋柜的蓝色盒子 |")
    add(f"| T2 | 50,000 · 2026-09-10 · 阳台花盆下面 |")
    add(f"| T3 | 88,888 · 2026-10-20（上周二）· 主卧床头柜第二个抽屉 |")
    add(f"| A0 截断 | 前 {phase0['a0']['cutoff']:,} 条，{phase0['a0']['tokens']:,} Token |")
    add(f"| A1 分片 | 1–33,333 / 33,334–66,666 / 66,667–100,000 |")
    add(
        f"| B 检索 | `search_memory(\"上周二 备用钥匙\", top_k=3)`，"
        f"RRF+Flat（无硬编码实体/槽位） |"
    )
    add(f"| B Prompt Token | {phase1['b_prompt_tokens']} |")
    add(f"| B 门禁 | G1={phase1['gates']['G1']}，G2a={phase1['gates']['G2a']}，G3={phase1['gates']['G3']} |")
    add(f"| B latest-wins | `{phase1['latest_wins_mechanism']}` |")
    add("")

    add("## 3. 成本对比")
    add("")
    add("| Arm | 调用数 | Input | Output | Cache hit | Cache miss | USD | CNY |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        t = stats[arm]["trials"]
        add(
            f"| {arm} | {stats[arm]['calls']} | "
            f"{sum(x['input_tokens'] for x in t):,} | "
            f"{sum(x['output_tokens'] for x in t):,} | "
            f"{sum(x['cache_hit_tokens'] for x in t):,} | "
            f"{sum(x['cache_miss_tokens'] for x in t):,} | "
            f"${Decimal(stats[arm]['cost_usd']):.6f} | "
            f"¥{Decimal(stats[arm]['cost_cny']):.4f} |"
        )
    add(
        f"| **合计** | **{len(calls)}** | **{total_input:,}** | **{total_output:,}** | "
        f"**{total_hit:,}** | **{total_miss:,}** | **${total_cost:.6f}** | **¥{total_cny:.4f}** |"
    )
    add("")
    add("成本倍率：")
    add("")
    b_cost = Decimal(stats["B"]["cost_usd"])
    a0_cost = Decimal(stats["A0"]["cost_usd"])
    a1_cost = Decimal(stats["A1"]["cost_usd"])
    add(f"- A0 / B ≈ **{a0_cost / b_cost:.1f}×**")
    add(f"- A1 / B ≈ **{a1_cost / b_cost:.1f}×**")
    add("")

    add("## 4. 延迟对比")
    add("")
    add("| Arm | Trial 1 | Trial 2 | Trial 3 | 中位 |")
    add("|---|---:|---:|---:|---:|")
    for arm in ARMS:
        values = [t["latency_ms"] for t in stats[arm]["trials"]]
        add(
            f"| {arm} | " + " | ".join(f"{v:.0f} ms" for v in values)
            + f" | **{stats[arm]['median_latency_ms']:.0f} ms** |"
        )
    add("")
    add("> A1 的一次 trial 含 3 次分片抽取 + 1 次汇总；A0/B 为单次生成。")
    add("")

    add("## 5. 召回与答案")
    add("")
    add("### 5.1 B 组检索门禁")
    add("")
    add("```text")
    add(f"T3 top-1:       {phase1['ranks']['t3_top3']}")
    add(f"T1 in top-3:    {phase1['ranks']['t1_in_top3']}")
    add(f"T2 in top-3:    {phase1['ranks']['t2_in_top3']}")
    if phase1.get("architecture") == "RRF+Flat":
        vector_ranks = phase1.get("vector_ranks_top12", {})
        add(
            "Flat vector ranks: "
            f"T1={vector_ranks.get('T1')} T3={vector_ranks.get('T3')} "
            f"T2={vector_ranks.get('T2')}"
        )
        add(
            "RRF fused (before latest-wins): "
            + " > ".join(phase1.get("rrf_fused_before_latest_wins", []))
        )
    else:
        add(f"T3 keyword rank: {phase1['ranks']['t3_keyword']}")
        add(f"T1 keyword rank: {phase1['ranks']['t1_keyword']}")
        add(f"T2 keyword rank: {phase1['ranks']['t2_keyword']}")
    add(f"latest-wins:    {phase1['latest_wins_mechanism']}")
    add(f"B prompt tokens: {phase1['b_prompt_tokens']}")
    add("```")
    add("")
    add("### 5.2 答案分类")
    add("")
    add("| Arm | Trial | 自动分类 | 最终分类 | 耗时 | 成本 |")
    add("|---|---:|---|---|---:|---:|")
    for row in answer_rows:
        add(
            f"| {row['arm']} | {row['trial']} | {row['auto_category']} | "
            f"**{row['category']}** | {row['latency_ms']:.0f} ms | "
            f"${Decimal(str(row['cost_usd'])):.6f} |"
        )
    add("")
    add("分类统计：")
    add("")
    add("| 类别 | A0 | A1 | B |")
    add("|---|---:|---:|---:|")
    for category in (
        "CORRECT_LATEST",
        "PARTIAL_LATEST",
        "STALE_T1",
        "STALE_T2",
        "UNKNOWN",
        "OTHER_HALLUCINATION",
        "MIXED",
    ):
        add(
            f"| {category} | {by_arm_category['A0'].get(category, 0)} | "
            f"{by_arm_category['A1'].get(category, 0)} | "
            f"{by_arm_category['B'].get(category, 0)} |"
        )
    add("")

    add("## 6. 原始回答（全文）")
    add("")
    for arm in ARMS:
        add(f"### {arm}")
        add("")
        for row in [r for r in answer_rows if r["arm"] == arm]:
            add(f"**{row['key']} · {row['category']}**")
            add("")
            add("```text")
            add(row["answer"])
            add("```")
            if row.get("rationale"):
                add("")
                add(f"评分理由：{row['rationale']}")
            add("")

    add("## 7. 结论与局限")
    add("")
    add("### 7.1 结论")
    add("")
    if b_strict == 3 and a0_strict == 0:
        add(
            f"- B 在 3/3 次中给出最新位置（主卧床头柜第二个抽屉），"
            f"A0 在 3/3 次中未能给出最新位置。"
        )
    else:
        add(f"- B 严格正确 {b_strict}/3；A0 {a0_strict}/3；A1 {a1_strict}/3。")
    add(
        f"- B 的总成本为 ${b_cost:.6f}，A0 为 ${a0_cost:.6f}，A1 为 ${a1_cost:.6f}。"
    )
    add(
        f"- B 的端到端中位耗时为 {stats['B']['median_latency_ms']:.0f} ms，"
        f"A0 为 {stats['A0']['median_latency_ms']:.0f} ms，"
        f"A1 为 {stats['A1']['median_latency_ms']:.0f} ms。"
    )
    add("")
    add("### 7.2 局限")
    add("")
    add("- N=3，仅作描述统计，不做显著性检验；单评审分类。")
    add("- A0 是人为 1M 截断，不是“模型注意了全部 2.5M Token”。")
    add("- A1 是 map-reduce 工程基线，不是原生全上下文注意力。")
    add("- B 的实体查询依赖关键词兜底；HNSW 在 100k 上仍有残余召回问题，不能宣称 HNSW 已修复。")
    add("- B Prompt 仍包含 2 条无关噪音；但 T1/T2 已剔除。")
    add("- 场景日期冻结为 2026-10-28；实际运行日期为 2026-10-30，只影响价格窗口，不影响语料/查询。")
    add("")
    add("## 8. 预算与可复现性")
    add("")
    add(f"- 实际总花费：**${total_cost:.6f} / ¥{total_cny:.4f}**")
    add(f"- 硬上限：${HARD_CAP_USD} / ¥{HARD_CAP_CNY}")
    add(f"- 请求级明细：[EXPERIMENT_USAGE.csv](EXPERIMENT_USAGE.csv)")
    add(f"- 运行目录：`{run_dir}`")
    add("")
    add("---")
    add("")
    add("**报告生成时间**：" + __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"))

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT_PATH}")
    print(f"wrote {USAGE_CSV}")
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "total_cost_usd": str(total_cost),
                "total_cost_cny": str(total_cny),
                "strict": strict,
                "lenient": lenient,
                "categories": by_arm_category,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
