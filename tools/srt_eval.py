#!/usr/bin/env python3
"""
audio-tools — SRT 断句质量评测（离线，不调引擎/模型）

用法：
    uv run python tools/srt_eval.py 字幕.srt                 # 只跑排版/语速指标
    uv run python tools/srt_eval.py 字幕.srt 人工校对.srt     # 再算断点 F1 与需改动条数

只读文件、只打印，不改动任何产物。指标口径：

- 孤行：非首行宽度 ≤ 6（3 汉字）的行数；断词粗判为"第二行以单字虚词起首"；
- 行宽：按 CJK=2、其余=1 计，与 src/subtitle.py 的排版口径一致；
- 语速：CJK 字数 / 秒（国内字幕惯例 9-17 字/秒），>17 记一条，短于 0.5s 的
  条目不计；
- 断点 F1：把两条字幕的正文（去标点空白）拼成字符流，比较条末位置的集合；
- 需改动条数：预测里起止字符区间与人工版某条完全重合的算"命中"，其余算需要
  人工改动（衡量的是"断条位置"，不评价用词）。
"""

from __future__ import annotations

import difflib
import re
import sys
import unicodedata

_TIME_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})")


def parse_srt(path: str) -> list[tuple[float, float, str]]:
    """解析 SRT：返回 [(起秒, 止秒, 文本)]，文本保留原始换行前的拼接。"""
    with open(path, encoding="utf-8-sig") as f:
        raw = f.read().replace("\r\n", "\n").replace("\r", "\n")
    cues: list[tuple[float, float, str]] = []
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        m = _TIME_RE.search(block)
        if not m:
            continue
        idx = 1 if not lines[0].strip().isdigit() else 2
        num = lambda t: (int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2])
                         + int(t[3].ljust(3, "0")) / 1000)      # noqa: E731
        start = num((m.group(1), m.group(2), m.group(3), m.group(4)))
        end = num((m.group(5), m.group(6), m.group(7), m.group(8)))
        text = " ".join(" ".join(lines[idx:]).split())
        if text:
            cues.append((start, end, text))
    return cues


def strip_text(text: str) -> str:
    """去标点与空白，只留正文（用于跨版本比对）。"""
    return "".join(ch for ch in text
                   if not ch.isspace() and not unicodedata.category(ch).startswith("P"))


def width(text: str) -> int:
    """显示宽度：CJK 与全角符号按 2 计，其余按 1 计。"""
    return sum(2 if _is_wide(ch) else 1 for ch in text)


def _is_wide(ch: str) -> bool:
    code = ord(ch)
    return (0x1100 <= code <= 0x115F or 0x2E80 <= code <= 0xA4CF
            or 0xAC00 <= code <= 0xD7A3 or 0xF900 <= code <= 0xFAFF
            or 0xFE30 <= code <= 0xFE6F or 0xFF00 <= code <= 0xFF60
            or 0xFFE0 <= code <= 0xFFE6 or 0x20000 <= code <= 0x3FFFD)


_NO_LINE_START = "的了和与在是把被对为而及或"
_MAX_WIDTH = 32                       # 与 SRT_MAX_LINE_WIDTH 默认值一致
_CPS_MAX = 17                         # 国内字幕惯例的语速上限（字/秒）


def layout_stats(path: str) -> dict:
    """排版与语速指标（不需要金标准）。"""
    cues = parse_srt(path)
    stats = {
        "条数": len(cues), "行数": 0, "孤行": 0, "行首虚词": 0, "超宽行": 0,
        "语速过快": 0, "平均语速": 0.0, "平均宽度": 0.0, "平均时长": 0.0,
    }
    total_width = total_dur = 0.0
    chars_per_sec = 0.0
    for start, end, text in cues:
        wide = width(text)
        dur = max(end - start, 0.0)
        total_width += wide
        total_dur += dur
        if dur >= 0.5:
            cps = (wide / 2.0) / dur
            chars_per_sec += cps
            if cps > _CPS_MAX:
                stats["语速过快"] += 1
    # 行级统计读原始文件（parse_srt 会把多行拼成一句，这里保留行结构）
    with open(path, encoding="utf-8-sig") as f:
        raw = f.read().replace("\r\n", "\n").replace("\r", "\n")
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not _TIME_RE.search(block):
            continue
        lines = lines[2:] if lines[0].isdigit() else lines[1:]
        stats["行数"] += len(lines)
        for i, line in enumerate(lines):
            if width(line) > _MAX_WIDTH:
                stats["超宽行"] += 1
            if i and width(line) <= 6:
                stats["孤行"] += 1
            if i and line[:1] in _NO_LINE_START:
                stats["行首虚词"] += 1
    if stats["条数"]:
        stats["平均宽度"] = round(total_width / stats["条数"], 1)
        stats["平均时长"] = round(total_dur / stats["条数"], 2)
        stats["平均语速"] = round(chars_per_sec / stats["条数"], 1)
    return stats


def cue_spans(path: str) -> tuple[str, list[tuple[int, int]]]:
    """正文字符流 + 每条字幕的 (起, 止) 字符区间。"""
    stream = ""
    spans: list[tuple[int, int]] = []
    for _start, _end, text in parse_srt(path):
        body = strip_text(text)
        spans.append((len(stream), len(stream) + len(body)))
        stream += body
    return stream, spans


def _map_offsets(pred: str, gold: str) -> list[int | None]:
    """预测字符流下标 → 金标准下标（按相同片段对齐，不同片段置 None）。"""
    mapping: list[int | None] = [None] * (len(pred) + 1)
    sm = difflib.SequenceMatcher(None, pred, gold, autojunk=False)
    for a, b, size in sm.get_matching_blocks():
        for k in range(size + 1):
            if a + k <= len(pred) and b + k <= len(gold):
                mapping[a + k] = b + k
    return mapping


def compare(pred_path: str, gold_path: str) -> dict:
    """与人工校对版比：断点 F1 与"需要改动的条数"。"""
    pred, pred_spans = cue_spans(pred_path)
    gold, gold_spans = cue_spans(gold_path)
    same_text = pred == gold
    mapping = _map_offsets(pred, gold)
    gold_bounds = {end for _s, end in gold_spans}
    gold_span_set = set(gold_spans)
    hit = 0
    pred_bounds: set[int] = set()
    for start, end in pred_spans:
        ms, me = mapping[start], mapping[end]
        if ms is None or me is None:
            continue
        pred_bounds.add(me)
        if (ms, me) in gold_span_set:
            hit += 1
    inter = len(pred_bounds & gold_bounds)
    precision = inter / len(pred_bounds) if pred_bounds else 0.0
    recall = inter / len(gold_bounds) if gold_bounds else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {
        "文本一致": same_text,
        "预测条数": len(pred_spans),
        "金标准条数": len(gold_spans),
        "断点精确率": round(precision, 3),
        "断点召回率": round(recall, 3),
        "断点F1": round(f1, 3),
        "命中条数": hit,
        "需要改动条数": len(pred_spans) - hit,
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip())
        return 2
    for path in argv[1:]:
        stats = layout_stats(path)
        print("== %s ==" % path)
        for key in ("条数", "行数", "平均宽度", "平均时长", "平均语速",
                    "孤行", "行首虚词", "超宽行", "语速过快"):
            print("  %s: %s" % (key, stats[key]))
    if len(argv) >= 3:
        print("== 与 %s 对比 ==" % argv[2])
        for key, value in compare(argv[1], argv[2]).items():
            print("  %s: %s" % (key, value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
