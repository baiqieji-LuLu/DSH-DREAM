"""10k trivia v2: diverse corpus, 200-char diaries, centered target windows."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
import time
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    CST,
    EXP,
    MODEL,
    PRICE_CACHE_MISS,
    PRICE_OUTPUT,
    USD_CNY,
    count_tokens,
    cost_usd,
    is_peak,
    load_api_key,
    load_tokenizer,
    make_client,
    now_cst,
    usage_to_dict,
    write_json,
)

ROOT = Path(r"G:\deepseek_ragproject_workspace")
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agent_memory.rag_store import RAGStore  # noqa: E402

INDEX_DIR = EXP / "trivia-10k-v2-flat"
RUN_ID = "trivia-10k-v2-" + datetime.now(tz=CST).strftime("%Y%m%dT%H%M%S")
RUN_DIR = EXP / "runs" / RUN_ID
PRIOR = Decimal("1.612286733")
CAP = Decimal("3.50")
QUERY = "我那个星巴克蓝色马克杯现在怎么样了？请告诉我它的完整历史。"
SUMMARY_PROMPT = "请总结这100条日常记录的核心事件。"

TARGETS = {
    2_500: "今天逛街买了一个星巴克的蓝色马克杯，挺好看的。",
    5_000: "刚才不小心把那个蓝色马克杯摔碎了，有点可惜。",
    7_500: "又在网上买了一个同款的蓝色马克杯补上，希望这次别摔了。",
}

ACTIVITIES = (
    "早上坐地铁去公司上班", "中午和同事去楼下吃面", "下午在工位上改了三版方案",
    "下班后去超市买了点水果", "晚上在家煮了一锅番茄汤", "周末把房间打扫了一遍",
    "早上绕着小区走了两圈", "午后在客厅看了一集纪录片", "傍晚去菜市场买了青菜",
    "晚上把攒了一周的衣服洗了", "今天的工作比平时顺一点", "中午一个人吃了碗牛肉面",
    "下午开了两个小时的会", "下班路上顺便取了快递", "晚上给自己煎了个鸡蛋",
    "早上把绿萝搬到窗边晒太阳", "周末去图书馆还了两本书", "晚上和朋友通了电话",
    "今天把冰箱里的剩菜热了", "下午去楼下咖啡店坐了会儿", "早上出门发现公交卡忘带了",
    "中午在食堂排了十分钟队", "下午把下周的计划重新排了一遍", "晚上去河边慢跑了二十分钟",
    "今天把书桌重新整理了一遍", "早上听了一期播客", "下午修好了松动的椅子",
    "晚上试着做了一道新菜", "今天把阳台的鞋子刷了", "周末去公园坐了一下午",
    "早上把窗户擦了一遍", "中午吃了一份沙拉", "下午整理了电脑里的文件",
    "晚上看了一场球赛", "今天路过一家新开的面包店", "早上把床单换洗了",
    "下午去银行办了业务", "晚上和家人视频了半小时", "今天把冰箱清理了一遍",
    "周末去郊外走了走", "早上喝了一杯热豆浆", "中午在楼下吃了饺子",
)

DETAILS = (
    "味道比上次咸了一点", "路上有点堵但没迟到", "回来的时候天已经黑了",
    "顺手把桌子擦干净了", "发现家里的洗衣液快用完了", "楼下的桂花开了味道很浓",
    "新买的日用品比想象中好用", "把下周要交的材料列了个清单", "地铁上人很多站了一路",
    "回家路上看到一只小狗", "晚饭后沿着河边走了一圈", "把手机里存了很久的照片删了",
    "今天喝了两杯温水", "窗台上的绿萝又长了一片叶子", "同事推荐了一家新的面馆",
    "把明天的闹钟提前了十分钟", "把文件按日期排好了", "晚上看完了那部剧的最后一集",
    "雨伞落在公司了，明天得去拿", "把厨房的台面收拾干净了", "发现冰箱里的牛奶过期了",
    "把旧衣服整理出来准备捐掉", "楼下的快递柜又满了", "今天走了一万多步",
    "把客厅的灯换成了暖色", "早上起得比平时早半小时", "下午喝了一杯不加糖的茶",
    "把书架上的书重新分类了", "发现阳台的多肉长出了新芽", "把地毯拿到窗边晒了晒",
)

FEELINGS = (
    "心情还不错", "觉得挺充实的", "有点累但很踏实", "希望明天也顺利",
    "感觉时间过得很快", "整个人放松了不少", "没什么特别的", "比预想中顺利",
    "稍微有点困", "觉得这样挺好", "打算早点休息", "心情平静了很多",
    "觉得今天没白过", "想周末再出去走走", "对明天有点期待", "觉得日子很普通",
)


def build_records() -> list[dict[str, Any]]:
    rng = random.Random(20261031)
    base = datetime(2026, 9, 1, 12, 0, tzinfo=CST)
    records = []
    used: set[str] = set()
    for position in range(1, 10_001):
        dt = base + timedelta(minutes=position * 3)
        if position in TARGETS:
            content = TARGETS[position]
        else:
            for _ in range(50):
                content = (
                    f"{dt.month}月{dt.day}日，"
                    f"{rng.choice(ACTIVITIES)}，{rng.choice(DETAILS)}，"
                    f"{rng.choice(FEELINGS)}。"
                )
                if content not in used:
                    break
            used.add(content)
        records.append(
            {
                "id": f"trivia10k-{position:05d}",
                "position": position,
                "timestamp_ms": int(dt.timestamp() * 1000),
                "content": content,
                "source_type": "user_message",
            }
        )
    return records


class Tracker:
    def __init__(self) -> None:
        self.spent = PRIOR

    def precheck(self, prompt_tokens: int, max_tokens: int) -> None:
        worst = (
            Decimal(prompt_tokens) / Decimal(1_000_000) * PRICE_CACHE_MISS
            + Decimal(max_tokens) / Decimal(1_000_000) * PRICE_OUTPUT
        )
        if self.spent + worst > CAP:
            raise RuntimeError(f"budget cap: {self.spent} + {worst}")

    def record(self, usage: dict[str, int]) -> Decimal:
        cost = cost_usd(usage)
        self.spent += cost
        if self.spent > CAP:
            raise RuntimeError(f"budget exceeded: {self.spent}")
        return cost


def call(
    client: Any,
    tokenizer: Any,
    tracker: Tracker,
    *,
    label: str,
    system: str,
    user: str,
    max_tokens: int,
) -> dict[str, Any]:
    prompt = f"{system}\n\n{user}"
    prompt_tokens_est = count_tokens(prompt, tokenizer)
    tracker.precheck(prompt_tokens_est, max_tokens)
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        stream=False,
        extra_body={"thinking": {"type": "disabled"}},
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    usage = usage_to_dict(response.usage)
    text = (response.choices[0].message.content or "").strip()
    if not usage or not text:
        raise RuntimeError("bad response")
    cost = tracker.record(usage)
    record = {
        "label": label,
        "answer": text,
        "usage": usage,
        "cost_usd": str(cost),
        "cost_cny": str((cost * USD_CNY).quantize(Decimal("0.0001"))),
        "latency_ms": round(latency_ms, 4),
        "finish_reason": response.choices[0].finish_reason,
        "prompt_tokens_est": prompt_tokens_est,
        "prompt": prompt,
        "requested_at_cst": now_cst().isoformat(timespec="seconds"),
        "response_id": getattr(response, "id", None),
        "response_model": getattr(response, "model", None),
        "response": response.model_dump() if hasattr(response, "model_dump") else {},
    }
    write_json(RUN_DIR / f"{label}.json", record)
    return record


def summarize_group(
    client: Any,
    tokenizer: Any,
    tracker: Tracker,
    records: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    log = "\n".join(f"- {r['content']}" for r in records)
    return call(
        client,
        tokenizer,
        tracker,
        label=label,
        system="你是一个日记整理助手。只输出日记正文，不要解释。200字以内。",
        user=f"下面是100条日常记录：\n{log}\n\n{SUMMARY_PROMPT}",
        max_tokens=300,
    )


def main() -> int:
    load_api_key()
    if is_peak(now_cst()):
        raise SystemExit("STOP: peak hour")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    records = build_records()
    tokenizer = load_tokenizer()
    client = make_client()
    tracker = Tracker()

    print("[1/5] 100 standard diaries (200 chars)")
    standard: dict[int, dict[str, Any]] = {}
    for group in range(1, 101):
        start = (group - 1) * 100
        standard[group] = summarize_group(
            client, tokenizer, tracker, records[start : start + 100],
            label=f"standard-{group:03d}",
        )
        if group % 10 == 0:
            print(f"      {group}/100 spent=${tracker.spent:.6f}", flush=True)

    print("[2/5] 3 centered target windows")
    centered: dict[int, dict[str, Any]] = {}
    for group in (25, 50, 75):
        center = group * 100
        start = center - 50
        centered[group] = summarize_group(
            client, tokenizer, tracker, records[start : start + 100],
            label=f"centered-{group:03d}",
        )

    print("[3/5] inspect diaries")
    standard_matches = []
    for group, result in standard.items():
        if any(term in result["answer"] for term in ("马克杯", "星巴克", "杯子")):
            standard_matches.append((group, result["answer"]))
    centered_matches = [
        (group, result["answer"]) for group, result in centered.items()
        if any(term in result["answer"] for term in ("马克杯", "星巴克", "杯子"))
    ]
    print("standard_matches", [g for g, _ in standard_matches])
    print("centered_matches", [g for g, _ in centered_matches])
    for group, text in centered_matches:
        print(f"--- centered {group} ---\n{text}\n")

    c_context = "\n\n".join(
        f"【日记{group}】\n{centered[group]['answer']}" for group in (25, 50, 75)
    )
    c_query = call(
        client, tokenizer, tracker,
        label="c-query",
        system="你是一个严谨的个人助理。只依据提供的日记回答。材料中没有的信息，明确说不知道，不要编造。",
        user=f"以下是三篇日记：\n<diaries>\n{c_context}\n</diaries>\n\n问题：{QUERY}",
        max_tokens=256,
    )

    print("[4/5] RAG 10k Flat timeline")
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR, ignore_errors=True)
    store = RAGStore(
        persist_directory=INDEX_DIR,
        collection_name="trivia_10k_v2",
        model_name="BAAI/bge-small-zh-v1.5",
        device="cpu",
        flat_max_records=200_000,
    )
    texts = [r["content"] for r in records]
    vectors = store._embedding_function(texts)
    for start in range(0, len(records), 5000):
        stop = min(start + 5000, len(records))
        store._collection.add(
            ids=[r["id"] for r in records[start:stop]],
            documents=texts[start:stop],
            metadatas=[
                {
                    "source_type": "user_message",
                    "created_at": now_cst().isoformat(timespec="seconds"),
                    "timestamp_ms": r["timestamp_ms"],
                    "content_sha256": hashlib.sha256(r["content"].encode()).hexdigest(),
                }
                for r in records[start:stop]
            ],
            embeddings=vectors[start:stop],
        )
    hits = store.search_memory("星巴克 蓝色马克杯", top_k=5, timeline_mode=True)
    store.close()
    print("timeline_hits", [(h.memory_id, h.content) for h in hits])

    rag_context = "\n".join(f"[{i}] {h.content}" for i, h in enumerate(hits, 1))
    rag_query = call(
        client, tokenizer, tracker,
        label="rag-query",
        system="你是一个严谨的个人助理。只依据提供的记忆原文回答。材料中没有的信息，明确说不知道，不要编造。",
        user=f"以下是按时间顺序排列的记忆原文：\n<memory>\n{rag_context}\n</memory>\n\n问题：{QUERY}",
        max_tokens=256,
    )

    total_c = tracker.spent - PRIOR
    report = {
        "run_id": RUN_ID,
        "records": 10_000,
        "targets": TARGETS,
        "standard_match_groups": [g for g, _ in standard_matches],
        "centered_match_groups": [g for g, _ in centered_matches],
        "centered_diaries": {str(g): centered[g]["answer"] for g in (25, 50, 75)},
        "standard_target_diaries": {str(g): standard[g]["answer"] for g in (25, 50, 75)},
        "c_query": c_query["answer"],
        "rag_hits": [
            {"id": h.memory_id, "content": h.content, "timestamp_ms": h.metadata.get("timestamp_ms")}
            for h in hits
        ],
        "rag_query": rag_query["answer"],
        "cost": {
            "standard_100": str(sum(Decimal(standard[g]["cost_usd"]) for g in standard)),
            "centered_3": str(sum(Decimal(centered[g]["cost_usd"]) for g in centered)),
            "c_query": c_query["cost_usd"],
            "rag_query": rag_query["cost_usd"],
            "total_c": str(total_c),
            "total_with_prior": str(tracker.spent),
        },
        "latency": {
            "c_query": c_query["latency_ms"],
            "rag_query": rag_query["latency_ms"],
        },
        "usage": {
            "c_query": c_query["usage"],
            "rag_query": rag_query["usage"],
        },
    }
    write_json(RUN_DIR / "summary.json", report)
    write_json(EXP / "trivia_10k_v2_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
