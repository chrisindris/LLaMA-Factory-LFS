"""Deterministic per-prediction diagnostics: tags, length, repetition, GT match."""

from __future__ import annotations

import math
import re
import unicodedata
import zlib
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any


TAG_RE = re.compile(r"</?(?:think|answer)>")
SENTENCE_RE = re.compile(r"[.!?]+|\n+")
WHITESPACE_TOKEN_RE = re.compile(r"\S+")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
PUNCT_RUN_RE = re.compile(r"([^\w\s])\1{2,}")
CHAR_RUN_RE = re.compile(r"(.)\1{4,}", re.DOTALL)
BRACKET_PAIRS = {"(": ")", "[": "]", "{": "}"}
ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"


def simple_tokenize(text: str | None) -> list[str]:
    """Whitespace tokenizer used for repetition / distinct-n / the self-tests."""
    if not text:
        return []
    return WHITESPACE_TOKEN_RE.findall(text)


def word_tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    return WORD_RE.findall(text)


def count_tokens(text: str | None, tokenizer: Any | None = None) -> int:
    if not text:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    return len(simple_tokenize(text))


def count_words(text: str | None) -> int:
    return len(word_tokenize(text))


def count_sentences(text: str | None) -> int:
    if not text or not text.strip():
        return 0
    parts = [part for part in SENTENCE_RE.split(text) if part.strip()]
    return max(len(parts), 1)


def count_lines(text: str | None) -> int:
    if not text:
        return 0
    return text.count("\n") + 1


def _find_tags(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(), match.start(), match.end()) for match in TAG_RE.finditer(text or "")]


def _content_between(text: str, open_tag: str, close_tag: str) -> tuple[str | None, bool]:
    """Return inner text of the first well-ordered pair, plus a malformed flag."""
    tags = _find_tags(text)
    opens = [(start, end) for name, start, end in tags if name == open_tag]
    closes = [(start, end) for name, start, end in tags if name == close_tag]
    if not opens or not closes:
        return None, bool(opens or closes)
    open_start, open_end = opens[0]
    close_start, _close_end = closes[0]
    malformed = len(opens) != 1 or len(closes) != 1 or close_start < open_end
    if close_start < open_end:
        return None, True
    return text[open_end:close_start], malformed


def extract_think(text: str | None) -> tuple[str | None, bool]:
    return _content_between(text or "", THINK_OPEN, THINK_CLOSE)


def extract_answer(text: str | None) -> tuple[str | None, bool]:
    return _content_between(text or "", ANSWER_OPEN, ANSWER_CLOSE)


def analyze_tags(text: str | None) -> dict[str, Any]:
    text = text or ""
    tags = _find_tags(text)
    names = [name for name, _start, _end in tags]

    think_open_count = names.count(THINK_OPEN)
    think_close_count = names.count(THINK_CLOSE)
    answer_open_count = names.count(ANSWER_OPEN)
    answer_close_count = names.count(ANSWER_CLOSE)

    first_pos: dict[str, int] = {}
    for name, start, _end in tags:
        first_pos.setdefault(name, start)

    has_think_open = think_open_count > 0
    has_think_close = think_close_count > 0
    has_answer_open = answer_open_count > 0
    has_answer_close = answer_close_count > 0

    think_pair_ordered = has_think_open and has_think_close and first_pos[THINK_OPEN] < first_pos[THINK_CLOSE]
    answer_pair_ordered = has_answer_open and has_answer_close and first_pos[ANSWER_OPEN] < first_pos[ANSWER_CLOSE]
    has_complete_think_pair = think_pair_ordered
    has_complete_answer_pair = answer_pair_ordered
    has_both_tag_pairs = has_complete_think_pair and has_complete_answer_pair

    think_before_answer = False
    if has_think_open and has_answer_open:
        think_before_answer = first_pos[THINK_OPEN] < first_pos[ANSWER_OPEN]

    proper_tag_order = False
    if has_both_tag_pairs:
        proper_tag_order = (
            first_pos[THINK_OPEN] < first_pos[THINK_CLOSE] < first_pos[ANSWER_OPEN] < first_pos[ANSWER_CLOSE]
        )

    tags_non_overlapping = True
    stack: list[str] = []
    open_to_close = {THINK_OPEN: THINK_CLOSE, ANSWER_OPEN: ANSWER_CLOSE}
    close_to_open = {THINK_CLOSE: THINK_OPEN, ANSWER_CLOSE: ANSWER_OPEN}
    for name in names:
        if name in open_to_close:
            if stack:
                tags_non_overlapping = False
            stack.append(name)
        elif name in close_to_open:
            if not stack or stack[-1] != close_to_open[name]:
                tags_non_overlapping = False
                if stack:
                    stack.pop()
            else:
                stack.pop()
    if stack:
        tags_non_overlapping = False

    think_text, think_malformed = extract_think(text)
    answer_text, answer_malformed = extract_answer(text)
    malformed_tags = (
        think_open_count not in (0, 1)
        or think_close_count not in (0, 1)
        or answer_open_count not in (0, 1)
        or answer_close_count not in (0, 1)
        or think_malformed
        or answer_malformed
        or not tags_non_overlapping
        or (has_both_tag_pairs and not proper_tag_order)
    )

    think_stripped = (think_text or "").strip()
    answer_stripped = (answer_text or "").strip()
    think_is_empty = has_complete_think_pair and think_stripped == ""
    answer_is_empty = has_complete_answer_pair and answer_stripped == ""

    text_before_think = text[: first_pos[THINK_OPEN]] if has_think_open else text
    if has_complete_think_pair and has_answer_open:
        think_close_end = next(end for name, start, end in tags if name == THINK_CLOSE)
        text_between = text[think_close_end : first_pos[ANSWER_OPEN]]
    else:
        text_between = ""
    if has_answer_close:
        last_answer_close_end = max(end for name, start, end in tags if name == ANSWER_CLOSE)
        text_after_answer = text[last_answer_close_end:]
    else:
        text_after_answer = ""

    answer_inside_think = False
    if has_complete_think_pair and has_answer_open:
        think_open_at = first_pos[THINK_OPEN]
        think_close_at = first_pos[THINK_CLOSE]
        if think_open_at < first_pos[ANSWER_OPEN] < think_close_at:
            answer_inside_think = True

    think_after_answer = False
    if has_answer_close and has_think_open:
        last_answer_close = max(start for name, start, end in tags if name == ANSWER_CLOSE)
        # any think open after the last </answer>
        think_after_answer = any(start > last_answer_close for name, start, end in tags if name == THINK_OPEN)

    tag_presence_score = (
        0.25 * float(has_think_open)
        + 0.25 * float(has_think_close)
        + 0.25 * float(has_answer_open)
        + 0.25 * float(has_answer_close)
    )
    tag_pair_score = 0.5 * float(has_complete_think_pair) + 0.5 * float(has_complete_answer_pair)

    exact_sequence = names == [THINK_OPEN, THINK_CLOSE, ANSWER_OPEN, ANSWER_CLOSE]
    canonical_format = (
        exact_sequence
        and tags_non_overlapping
        and proper_tag_order
        and not think_is_empty
        and not answer_is_empty
        and text_before_think.strip() == ""
        and text_between.strip() == ""
        and text_after_answer.strip() == ""
    )
    usable_format = (
        exact_sequence
        and tags_non_overlapping
        and proper_tag_order
        and not think_is_empty
        and not answer_is_empty
        and text_between.strip() == ""
        and THINK_OPEN not in text_before_think
        and ANSWER_OPEN not in text_before_think
        and THINK_OPEN not in text_after_answer
        and ANSWER_OPEN not in text_after_answer
    )

    return {
        "has_think_open": has_think_open,
        "has_think_close": has_think_close,
        "has_answer_open": has_answer_open,
        "has_answer_close": has_answer_close,
        "think_open_count": think_open_count,
        "think_close_count": think_close_count,
        "answer_open_count": answer_open_count,
        "answer_close_count": answer_close_count,
        "has_complete_think_pair": has_complete_think_pair,
        "has_complete_answer_pair": has_complete_answer_pair,
        "has_both_tag_pairs": has_both_tag_pairs,
        "think_before_answer": think_before_answer,
        "proper_tag_order": proper_tag_order,
        "tags_non_overlapping": tags_non_overlapping,
        "think_is_empty": think_is_empty,
        "answer_is_empty": answer_is_empty,
        "text_before_think": text_before_think,
        "text_between_think_and_answer": text_between,
        "text_after_answer": text_after_answer,
        "think_text": think_text,
        "answer_text": answer_text,
        "malformed_tags": malformed_tags,
        "tag_presence_score": tag_presence_score,
        "tag_pair_score": tag_pair_score,
        "canonical_format": canonical_format,
        "usable_format": usable_format,
        "answer_inside_think": answer_inside_think,
        "think_after_answer": think_after_answer,
        "repeated_think_blocks": think_open_count > 1 or think_close_count > 1,
        "repeated_answer_blocks": answer_open_count > 1 or answer_close_count > 1,
        "text_after_answer_nonempty": bool(text_after_answer.strip()),
    }


def length_metrics(text: str | None, tokenizer: Any | None = None, prefix: str = "") -> dict[str, Any]:
    raw = text or ""
    tokens = simple_tokenize(raw)
    data = {
        f"{prefix}char_count": len(raw),
        f"{prefix}word_count": count_words(raw),
        f"{prefix}token_count": count_tokens(raw, tokenizer),
        f"{prefix}simple_token_count": len(tokens),
        f"{prefix}line_count": count_lines(raw),
        f"{prefix}sentence_count": count_sentences(raw),
    }
    return data


def _ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def ngram_stats(tokens: Sequence[str], n: int) -> dict[str, Any]:
    grams = _ngrams(tokens, n)
    total = len(grams)
    unique = len(set(grams))
    counts = Counter(grams)
    repeated = sum(1 for count in counts.values() if count > 1)
    max_freq = max(counts.values()) if counts else 0
    distinct = (unique / total) if total else math.nan
    repeated_frac = (1.0 - unique / total) if total else math.nan
    return {
        f"ngram{n}_total": total,
        f"ngram{n}_unique": unique,
        f"ngram{n}_repeated_count": repeated,
        f"ngram{n}_repeated_fraction": repeated_frac,
        f"ngram{n}_max_frequency": max_freq,
        f"distinct_{n}": distinct,
    }


def consecutive_repetition(tokens: Sequence[str]) -> dict[str, Any]:
    if not tokens:
        return {
            "adjacent_identical_pairs": 0,
            "adjacent_identical_fraction": math.nan,
            "max_identical_token_run": 0,
            "runs_ge3": 0,
            "runs_ge5": 0,
        }
    pairs = 0
    max_run = 1
    run = 1
    runs_ge3 = 0
    runs_ge5 = 0
    for prev, cur in zip(tokens, tokens[1:]):
        if prev == cur:
            pairs += 1
            run += 1
            max_run = max(max_run, run)
        else:
            if run >= 3:
                runs_ge3 += 1
            if run >= 5:
                runs_ge5 += 1
            run = 1
    if run >= 3:
        runs_ge3 += 1
    if run >= 5:
        runs_ge5 += 1
    return {
        "adjacent_identical_pairs": pairs,
        "adjacent_identical_fraction": pairs / max(len(tokens) - 1, 1),
        "max_identical_token_run": max_run if tokens else 0,
        "runs_ge3": runs_ge3,
        "runs_ge5": runs_ge5,
    }


def token_concentration(tokens: Sequence[str], exclude_punct: bool = False) -> dict[str, Any]:
    filtered = [tok for tok in tokens if tok.strip()]
    if exclude_punct:
        filtered = [tok for tok in filtered if any(ch.isalnum() for ch in tok)]
    n = len(filtered)
    if n == 0:
        return {
            "most_common_token_fraction": math.nan,
            "top_3_token_fraction": math.nan,
            "top_5_token_fraction": math.nan,
        }
    counts = Counter(filtered)
    top = counts.most_common(5)
    top1 = top[0][1] / n
    top3 = sum(count for _tok, count in top[:3]) / n
    top5 = sum(count for _tok, count in top[:5]) / n
    return {
        "most_common_token_fraction": top1,
        "top_3_token_fraction": top3,
        "top_5_token_fraction": top5,
    }


def repeated_spans(tokens: Sequence[str], max_n: int = 32) -> dict[str, Any]:
    n = len(tokens)
    longest_span = ""
    longest_n = 0
    longest_count = 0
    most_span = ""
    most_count = 0
    limit = min(max_n, n // 2) if n else 0
    for length in range(2, limit + 1):
        counts: dict[tuple[str, ...], int] = {}
        for i in range(n - length + 1):
            gram = tuple(tokens[i : i + length])
            counts[gram] = counts.get(gram, 0) + 1
        if not counts:
            continue
        gram, count = max(counts.items(), key=lambda item: item[1])
        if count >= 2 and count > most_count:
            most_count = count
            most_span = " ".join(gram)
        if count >= 2 and length > longest_n:
            longest_n = length
            longest_span = " ".join(gram)
            longest_count = count
    return {
        "longest_repeated_token_span": longest_n,
        "longest_repeated_span": longest_span,
        "longest_repeated_span_count": longest_count,
        "most_repeated_span": most_span,
        "most_repeated_span_count": most_count,
    }


def compression_ratio(text: str | None) -> float:
    raw = (text or "").encode("utf-8")
    if not raw:
        return math.nan
    compressed = zlib.compress(raw, level=6)
    return len(compressed) / len(raw)


def repetition_score_from_parts(
    *,
    adjacent_identical_fraction: float,
    distinct_3: float,
    compression: float,
    most_common_token_fraction: float,
    weights: tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1),
) -> float:
    """Interpretable composite in roughly [0, 1]. Not a calibrated quality metric.

    ``0.4 * adjacent_identical_fraction
      + 0.3 * (1 - distinct_3)
      + 0.2 * (1 - compression_ratio)
      + 0.1 * most_common_token_fraction``
    Missing components are dropped and remaining weights renormalized.
    """
    parts = [
        (weights[0], adjacent_identical_fraction),
        (weights[1], (1.0 - distinct_3) if distinct_3 == distinct_3 else math.nan),
        (weights[2], (1.0 - compression) if compression == compression else math.nan),
        (weights[3], most_common_token_fraction),
    ]
    usable = [(weight, value) for weight, value in parts if value == value]
    if not usable:
        return math.nan
    total_w = sum(weight for weight, _value in usable)
    return sum(weight * value for weight, value in usable) / total_w


def analyze_repetition(text: str | None, prefix: str = "") -> dict[str, Any]:
    tokens = simple_tokenize(text)
    out: dict[str, Any] = {}
    for key, value in consecutive_repetition(tokens).items():
        out[f"{prefix}{key}"] = value
    for n in (1, 2, 3, 4, 5):
        stats = ngram_stats(tokens, n)
        # distinct_1 lives on n=1; ngram1_* also recorded.
        for key, value in stats.items():
            if n == 1 and key.startswith("ngram"):
                continue
            if key.startswith("distinct_"):
                out[f"{prefix}{key}"] = value
            else:
                out[f"{prefix}{key}"] = value
    for key, value in token_concentration(tokens).items():
        out[f"{prefix}{key}"] = value
    for key, value in repeated_spans(tokens).items():
        out[f"{prefix}{key}"] = value
    compression = compression_ratio(text)
    out[f"{prefix}compression_ratio"] = compression
    out[f"{prefix}repetition_score"] = repetition_score_from_parts(
        adjacent_identical_fraction=out.get(f"{prefix}adjacent_identical_fraction", math.nan),
        distinct_3=out.get(f"{prefix}distinct_3", math.nan),
        compression=compression,
        most_common_token_fraction=out.get(f"{prefix}most_common_token_fraction", math.nan),
    )
    return out


def vocab_metrics(text: str | None, prefix: str = "") -> dict[str, Any]:
    tokens = simple_tokenize(text)
    n = len(tokens)
    unique = len(set(tokens))
    counts = Counter(tokens)
    hapax = sum(1 for count in counts.values() if count == 1)
    mean_len = (sum(len(tok) for tok in tokens) / n) if n else math.nan
    return {
        f"{prefix}unique_token_count": unique,
        f"{prefix}type_token_ratio": (unique / n) if n else math.nan,
        f"{prefix}hapax_count": hapax,
        f"{prefix}hapax_fraction": (hapax / n) if n else math.nan,
        f"{prefix}mean_token_length": mean_len,
    }


def unclosed_brackets(text: str | None) -> bool:
    stack: list[str] = []
    for char in text or "":
        if char in BRACKET_PAIRS:
            stack.append(char)
        elif char in BRACKET_PAIRS.values():
            if not stack or BRACKET_PAIRS[stack[-1]] != char:
                return True
            stack.pop()
    return bool(stack)


def punctuation_diagnostics(text: str | None) -> dict[str, Any]:
    raw = text or ""
    punct_runs = PUNCT_RUN_RE.findall(raw)
    char_runs = CHAR_RUN_RE.findall(raw)
    max_char_run = 1
    if raw:
        run = 1
        prev = raw[0]
        for char in raw[1:]:
            if char == prev:
                run += 1
                max_char_run = max(max_char_run, run)
            else:
                prev = char
                run = 1
    extra_xml = len(re.findall(r"</?(?!think\b)(?!answer\b)[A-Za-z][\w:-]*>", raw))
    return {
        "max_char_run": max_char_run,
        "repeated_punctuation_runs": len(punct_runs),
        "long_char_runs": len(char_runs),
        "unclosed_brackets": unclosed_brackets(raw),
        "extra_xml_tag_count": extra_xml,
    }


def unicode_normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def normalize_whitespace(text: str) -> str:
    return " ".join(unicode_normalize(text).split())


def normalize_casefold(text: str) -> str:
    return normalize_whitespace(text).casefold()


def normalize_punct_articles(text: str) -> str:
    folded = normalize_casefold(text)
    folded = re.sub(r"[^\w\s]", " ", folded)
    folded = ARTICLES_RE.sub(" ", folded)
    return " ".join(folded.split())


def token_overlap_metrics(pred_tokens: Sequence[str], ref_tokens: Sequence[str]) -> dict[str, float]:
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    n_pred = sum(pred_counts.values())
    n_ref = sum(ref_counts.values())
    precision = overlap / n_pred if n_pred else math.nan
    recall = overlap / n_ref if n_ref else math.nan
    if precision == precision and recall == recall and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = math.nan
    pred_set, ref_set = set(pred_tokens), set(ref_tokens)
    union = pred_set | ref_set
    jaccard = (len(pred_set & ref_set) / len(union)) if union else math.nan
    return {
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
        "jaccard": jaccard,
    }


def _levenshtein_ratio(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    try:
        from rapidfuzz.distance import Levenshtein

        return 1.0 - Levenshtein.normalized_distance(left, right)
    except Exception:
        pass
    # Small DP for short strings; skip very long pairs.
    if len(left) * len(right) > 400_000:
        return math.nan
    prev = list(range(len(right) + 1))
    for i, ch_l in enumerate(left, start=1):
        cur = [i]
        for j, ch_r in enumerate(right, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ch_l != ch_r)
            cur.append(min(ins, delete, sub))
        prev = cur
    dist = prev[-1]
    return 1.0 - dist / max(len(left), len(right))


def compare_to_references(prediction: str | None, references: Iterable[str | None]) -> dict[str, Any]:
    refs = [ref for ref in references if isinstance(ref, str) and ref != ""]
    pred = prediction or ""
    result = {
        "n_references": len(refs),
        "exact_match": False,
        "case_insensitive_exact_match": False,
        "normalized_exact_match": False,
        "punct_article_exact_match": False,
        "token_precision": math.nan,
        "token_recall": math.nan,
        "token_f1": math.nan,
        "jaccard": math.nan,
        "normalized_levenshtein": math.nan,
        "best_reference": None,
    }
    if not refs:
        return result

    best: dict[str, Any] | None = None
    best_key = (-1.0, -1.0, -1.0)
    pred_tokens = simple_tokenize(normalize_casefold(pred))
    for ref in refs:
        overlap = token_overlap_metrics(pred_tokens, simple_tokenize(normalize_casefold(ref)))
        exact = pred == ref
        ci = pred.casefold() == ref.casefold()
        norm = normalize_casefold(pred) == normalize_casefold(ref)
        punct = normalize_punct_articles(pred) == normalize_punct_articles(ref)
        lev = _levenshtein_ratio(normalize_casefold(pred), normalize_casefold(ref))
        key = (
            float(norm),
            overlap["token_f1"] if overlap["token_f1"] == overlap["token_f1"] else -1.0,
            lev if lev == lev else -1.0,
        )
        candidate = {
            "exact_match": exact,
            "case_insensitive_exact_match": ci,
            "normalized_exact_match": norm,
            "punct_article_exact_match": punct,
            **overlap,
            "normalized_levenshtein": lev,
            "best_reference": ref,
        }
        if best is None or key > best_key:
            best = candidate
            best_key = key
    assert best is not None
    result.update(best)
    result["n_references"] = len(refs)
    return result


def ratio(numer: float, denom: float) -> float:
    if denom is None or denom != denom or denom == 0:
        return math.nan
    if numer != numer:
        return math.nan
    return numer / denom


def per_prediction_metrics(
    prediction: str | None,
    *,
    tokenizer: Any | None = None,
    ground_truth: str | None = None,
    gt_think: str | None = None,
    gt_answer: str | None = None,
    extra_references: Sequence[str] | None = None,
    short_think_tokens: int = 8,
    long_think_tokens: int = 4000,
    short_answer_tokens: int = 1,
    long_answer_tokens: int = 2000,
    high_repetition: float = 0.35,
    high_ngram_repetition: float = 0.5,
    high_identical_run: int = 8,
    length_explosion_tokens: int = 6000,
) -> dict[str, Any]:
    pred = prediction or ""
    tags = analyze_tags(pred)
    think_text = tags["think_text"]
    answer_text = tags["answer_text"]

    out: dict[str, Any] = dict(tags)
    out.update(length_metrics(pred, tokenizer, prefix=""))
    out.update(length_metrics(think_text, tokenizer, prefix="think_"))
    out.update(length_metrics(answer_text, tokenizer, prefix="answer_"))
    total_tokens = out["token_count"]
    think_tokens = out["think_token_count"]
    answer_tokens = out["answer_token_count"]
    out["think_token_fraction"] = ratio(think_tokens, total_tokens)
    out["answer_token_fraction"] = ratio(answer_tokens, total_tokens)
    out["think_over_answer_tokens"] = ratio(think_tokens, max(answer_tokens, 1))

    out.update(analyze_repetition(pred, prefix=""))
    out.update(analyze_repetition(think_text, prefix="think_"))
    out.update(analyze_repetition(answer_text, prefix="answer_"))
    out.update(vocab_metrics(think_text, prefix="think_"))
    out.update(punctuation_diagnostics(pred))

    out["prediction_empty"] = pred.strip() == ""
    out["extremely_short_reasoning"] = bool(
        think_text is not None and think_tokens <= short_think_tokens and not out["think_is_empty"]
    )
    out["extremely_long_reasoning"] = think_tokens >= long_think_tokens
    out["empty_final_answer"] = bool(
        tags["answer_is_empty"] or (tags["has_complete_answer_pair"] is False and answer_text in (None, ""))
    )
    out["length_explosion"] = total_tokens >= length_explosion_tokens
    out["length_collapse"] = (not out["prediction_empty"]) and total_tokens <= 3
    out["very_high_token_repetition"] = (
        out.get("adjacent_identical_fraction") == out.get("adjacent_identical_fraction")
        and out["adjacent_identical_fraction"] >= high_repetition
    ) or out.get("max_identical_token_run", 0) >= high_identical_run
    ngram_rep = out.get("ngram3_repeated_fraction")
    out["very_high_ngram_repetition"] = ngram_rep == ngram_rep and ngram_rep >= high_ngram_repetition

    gt_tags = analyze_tags(ground_truth or "")
    out["gt_think_text"] = gt_think if gt_think is not None else gt_tags["think_text"]
    out["gt_answer_text"] = gt_answer if gt_answer is not None else gt_tags["answer_text"]
    out["gt_canonical_format"] = gt_tags["canonical_format"]
    out["gt_usable_format"] = gt_tags["usable_format"]
    out.update(length_metrics(ground_truth, tokenizer, prefix="gt_"))
    out.update(length_metrics(out["gt_think_text"], tokenizer, prefix="gt_think_"))
    out.update(length_metrics(out["gt_answer_text"], tokenizer, prefix="gt_answer_"))
    out["pred_minus_gt_think_tokens"] = think_tokens - out["gt_think_token_count"]
    out["pred_minus_gt_answer_tokens"] = answer_tokens - out["gt_answer_token_count"]
    out["pred_minus_gt_total_tokens"] = total_tokens - out["gt_token_count"]
    out["pred_over_gt_think_tokens"] = ratio(think_tokens, max(out["gt_think_token_count"], 1))
    out["pred_over_gt_answer_tokens"] = ratio(answer_tokens, max(out["gt_answer_token_count"], 1))
    out.update(analyze_repetition(ground_truth, prefix="gt_"))

    pred_answer_for_match = answer_text if answer_text is not None else pred
    refs: list[str] = []
    if out["gt_answer_text"]:
        refs.append(out["gt_answer_text"])
    if ground_truth and ground_truth not in refs:
        # Use full GT only if no answer span was recovered.
        if out["gt_answer_text"] is None:
            refs.append(ground_truth)
    if extra_references:
        refs.extend(extra_references)
    # Unique refs, preserve order.
    unique_refs: list[str] = []
    seen_refs: set[str] = set()
    for ref in refs:
        if ref not in seen_refs:
            unique_refs.append(ref)
            seen_refs.add(ref)
    match = compare_to_references(pred_answer_for_match, unique_refs)
    out.update(match)
    return out
