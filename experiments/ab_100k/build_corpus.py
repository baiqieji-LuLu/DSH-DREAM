"""Build the deterministic 100k corpus and token accounting."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    CORPUS_MANIFEST_PATH,
    CORPUS_PATH,
    CST,
    EXP,
    MANIFEST_PATH,
    SCENARIO_DATE_LABEL,
    TARGETS,
    TOTAL_RECORDS,
    count_many,
    count_tokens,
    iso_utc_from_ms,
    load_tokenizer,
    local_date_from_ms,
    sha256_file,
    target_timestamp_ms,
    write_json,
)

SEED = 20261028

FORBIDDEN = (
    "备用钥匙",
    "钥匙",
    "鞋柜",
    "蓝色盒子",
    "盒子",
    "花盆",
    "阳台",
    "主卧",
    "床头柜",
    "抽屉",
    "玄关",
    "猫",
    "下雨",
    "淋湿",
    "上周二",
    "备用",
    "柜",
)

ACTIVITIES = (
    "早上坐地铁去公司上班",
    "中午和同事去楼下吃面",
    "下午在工位上改了三版方案",
    "下班后去超市买了点水果",
    "晚上在家煮了一锅番茄汤",
    "周末把房间从头到尾打扫了一遍",
    "早上绕着小区走了两圈",
    "午后在客厅看了一集纪录片",
    "傍晚去菜市场买了青菜和豆腐",
    "晚上把攒了一周的衣服洗了",
    "今天的工作比平时顺一点",
    "中午一个人吃了碗牛肉面",
    "下午开了两个小时的会",
    "下班路上顺便取了快递",
    "晚上给自己煎了个鸡蛋",
    "早上把绿萝搬到窗边晒太阳",
    "周末去图书馆还了两本书",
    "晚上和朋友通了半小时电话",
    "今天把冰箱里的剩菜热了",
    "下午去楼下咖啡店坐了会儿",
    "早上出门时发现公交卡忘带了",
    "中午在食堂排了十分钟的队",
    "下午把下周的计划重新排了一遍",
    "晚上去河边慢跑了二十分钟",
)

DETAILS = (
    "味道比上次咸了一点",
    "路上有点堵但没迟到",
    "回来的时候天已经黑了",
    "顺手把桌子擦干净了",
    "发现家里的洗衣液快用完了",
    "楼下的桂花开了味道很浓",
    "新买的杯子比想象中好用",
    "把下周要交的材料列了个清单",
    "地铁上人很多站了一路",
    "回家路上看到一只小狗",
    "晚饭后沿着河边走了一圈",
    "把手机里存了很久的照片删了",
    "今天喝了两杯温水",
    "窗台上的绿萝又长了一片叶子",
    "同事推荐了一家新的面馆",
    "把明天的闹钟提前了十分钟",
    "把桌上的文件按日期排好了",
    "晚上看完了那部剧的最后一集",
    "雨伞落在公司了，明天得去拿",
    "把厨房的台面收拾干净了",
)

FEELINGS = (
    "心情还不错",
    "觉得挺充实的",
    "有点累但很踏实",
    "希望明天也顺利",
    "感觉时间过得很快",
    "整个人放松了不少",
    "没什么特别的",
    "比预想中顺利",
    "稍微有点困",
    "觉得这样挺好",
    "打算早点休息",
    "心情平静了很多",
    "觉得今天没白过",
    "想周末再出去走走",
    "对明天有点期待",
)

FILLERS = (
    "整体还算顺利",
    "和平时差不多",
    "没什么特别的事",
    "感觉一天过得很快",
)

# 1-based anchors; target positions must land exactly on their date.
ANCHORS = (
    (1, datetime(2026, 1, 1, 12, 0, tzinfo=CST)),
    (20_000, datetime(2026, 7, 15, 12, 0, tzinfo=CST)),
    (50_000, datetime(2026, 9, 10, 12, 0, tzinfo=CST)),
    (88_888, datetime(2026, 10, 20, 12, 0, tzinfo=CST)),
    (100_000, datetime(2026, 10, 27, 12, 0, tzinfo=CST)),
)


def timestamp_for_position(position: int) -> int:
    for index in range(len(ANCHORS) - 1):
        p0, t0 = ANCHORS[index]
        p1, t1 = ANCHORS[index + 1]
        if p0 <= position <= p1:
            if position == p0:
                dt = t0
            elif position == p1:
                dt = t1
            else:
                ratio = (position - p0) / (p1 - p0)
                dt = t0 + timedelta(seconds=(t1 - t0).total_seconds() * ratio)
            return int(dt.timestamp() * 1000)
    raise ValueError(f"position out of range: {position}")


def clean_noise(text: str) -> bool:
    return not any(word in text for word in FORBIDDEN)


def make_noise_text(rng: random.Random) -> str:
    for _ in range(200):
        date = None
        activity = rng.choice(ACTIVITIES)
        detail = rng.choice(DETAILS)
        feeling = rng.choice(FEELINGS)
        text = f"{activity}，{detail}，{feeling}。"
        if len(text) < 30:
            text = f"{activity}，{detail}，{feeling}，{rng.choice(FILLERS)}。"
        if len(text) > 42:
            text = f"{activity}，{detail}。"
            if len(text) < 30:
                text = f"{activity}，{detail}，{rng.choice(FILLERS)}。"
        if 30 <= len(text) <= 42 and clean_noise(text):
            return text
    fallback = "今天没有什么特别的事，吃完晚饭在小区里走了一圈，心情还算平静。"
    if not clean_noise(fallback):
        raise AssertionError("fallback noise contains a forbidden term")
    return fallback


def build_noise_records() -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    records: list[dict[str, Any]] = []
    target_positions = set(TARGETS)
    for position in range(1, TOTAL_RECORDS + 1):
        if position in target_positions:
            continue
        ts = timestamp_for_position(position)
        local_date = local_date_from_ms(ts)
        body = make_noise_text(rng)
        content = f"{local_date}，{body}"
        records.append(
            {
                "id": f"exp-{position:06d}",
                "position": position,
                "timestamp_ms": ts,
                "date_label": local_date,
                "content": content,
                "source_type": "user_message",
                "is_target": False,
                "target_label": None,
            }
        )
    return records


def build_all_records() -> list[dict[str, Any]]:
    noise = {r["position"]: r for r in build_noise_records()}
    records: list[dict[str, Any]] = []
    for position in range(1, TOTAL_RECORDS + 1):
        if position in TARGETS:
            target = TARGETS[position]
            ts = target_timestamp_ms(target)
            records.append(
                {
                    "id": target["id"],
                    "position": position,
                    "timestamp_ms": ts,
                    "date_label": local_date_from_ms(ts),
                    "content": target["text"],
                    "source_type": "user_message",
                    "is_target": True,
                    "target_label": target["label"],
                }
            )
        else:
            records.append(noise[position])
    return records


def validate(records: list[dict[str, Any]]) -> None:
    assert len(records) == TOTAL_RECORDS
    assert len({r["id"] for r in records}) == TOTAL_RECORDS
    assert len({r["position"] for r in records}) == TOTAL_RECORDS
    positions = [r["position"] for r in records]
    assert positions == list(range(1, TOTAL_RECORDS + 1))
    timestamps = [r["timestamp_ms"] for r in records]
    assert all(b > a for a, b in zip(timestamps, timestamps[1:])), "timestamps not strictly increasing"
    for position, target in TARGETS.items():
        record = records[position - 1]
        assert record["position"] == position
        assert record["content"] == target["text"], (position, record["content"])
        assert record["target_label"] == target["label"]
        assert record["timestamp_ms"] == target_timestamp_ms(target)
    noise = [r for r in records if not r["is_target"]]
    assert len(noise) == TOTAL_RECORDS - len(TARGETS)
    for record in noise:
        assert clean_noise(record["content"]), record["content"]
        assert "我上周二把家里的备用钥匙放在哪了？" not in record["content"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()

    EXP.mkdir(parents=True, exist_ok=True)
    records = build_all_records()
    validate(records)

    tokenizer = load_tokenizer()
    if args.sample:
        sample = records[: args.sample]
        counts = count_many([r["content"] for r in sample], tokenizer)
        print(
            json.dumps(
                {
                    "sample": len(sample),
                    "total_tokens": sum(counts),
                    "avg_tokens": round(sum(counts) / len(counts), 2),
                    "min": min(counts),
                    "max": max(counts),
                },
                ensure_ascii=False,
            )
        )
        return 0

    counts = count_many([r["content"] for r in records], tokenizer)
    for record, count in zip(records, counts):
        record["token_count"] = count

    with CORPUS_PATH.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    target_tokens = {
        record["target_label"]: record["token_count"]
        for record in records
        if record["is_target"]
    }
    total_tokens = sum(counts)
    corpus_sha = sha256_file(CORPUS_PATH)
    manifest = {
        "experiment": "AB-100K-KEY-2026",
        "generated_at_cst": datetime.now(tz=CST).isoformat(timespec="seconds"),
        "scenario_date": SCENARIO_DATE_LABEL,
        "seed": SEED,
        "records": len(records),
        "noise_records": len(records) - len(TARGETS),
        "target_positions": {str(k): v["label"] for k, v in TARGETS.items()},
        "target_tokens": target_tokens,
        "total_content_tokens": total_tokens,
        "avg_content_tokens": round(total_tokens / len(records), 4),
        "min_content_tokens": min(counts),
        "max_content_tokens": max(counts),
        "corpus_sha256": corpus_sha,
        "corpus_bytes": CORPUS_PATH.stat().st_size,
        "tokenizer_sha256": sha256_file(
            EXP / "tokenizer" / "deepseek_v4_tokenizer" / "tokenizer.json"
        ),
        "anchors": [
            {"position": p, "datetime": t.isoformat()} for p, t in ANCHORS
        ],
        "forbidden_terms": list(FORBIDDEN),
    }
    write_json(CORPUS_MANIFEST_PATH, manifest)
    write_json(
        MANIFEST_PATH,
        {
            "experiment": "AB-100K-KEY-2026",
            "status": "phase0_corpus_built",
            "model": "deepseek-flash",
            "base_url": "https://api.deepseek.com",
            "decisions": {
                "D1": "T3=2026-10-20",
                "D2": "A0+A1+B",
                "D3": "N=3, hard cap USD 2.50 / CNY 18",
                "D4": "strict gates G1/G2a/G3",
                "D5": "deepseek-flash",
                "D6": "Chinese rubric",
                "D7": "off-peak only",
            },
            "prices_usd_per_1m_offpeak": {
                "cache_hit": 0.003,
                "cache_miss": 0.15,
                "output": 0.60,
            },
            "corpus_manifest": manifest,
        },
    )
    print(
        json.dumps(
            {
                "records": len(records),
                "total_content_tokens": total_tokens,
                "avg": round(total_tokens / len(records), 2),
                "target_tokens": target_tokens,
                "corpus_sha256": corpus_sha,
                "corpus_mb": round(CORPUS_PATH.stat().st_size / 1024 / 1024, 2),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
