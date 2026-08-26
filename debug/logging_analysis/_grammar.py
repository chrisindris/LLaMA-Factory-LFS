"""Optional LanguageTool grammar diagnostics. Not a correctness metric."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Any

from _metrics import count_tokens, count_words


logger = logging.getLogger(__name__)

TAG_STRIP_RE = re.compile(r"</?(?:think|answer)>")

# LanguageTool category grouping (best-effort; LT category names vary).
CATEGORY_GROUPS = {
    "grammar": ("GRAMMAR", "COMPOUNDING"),
    "spelling": ("TYPOS", "TYPOGRAPHY", "CONFUSED_WORDS"),
    "punctuation": ("PUNCTUATION", "TYPOGRAPHY"),
    "style": ("STYLE", "REDUNDANCY", "PLAIN_ENGLISH"),
    "capitalization": ("CASING", "CAPITALIZATION"),
}


def strip_xml_like_tags(text: str | None) -> str:
    return TAG_STRIP_RE.sub(" ", text or "")


class GrammarAnalyzer:
    """Reuse one LanguageTool instance; cache by SHA-256 of the exact text."""

    def __init__(self, language: str = "en-US") -> None:
        self.language = language
        self._tool = None
        self._cache: dict[str, dict[str, Any]] = {}
        self.unavailable_reason: str | None = None
        self._init_tool()

    def _init_tool(self) -> None:
        try:
            import language_tool_python
        except ImportError as exc:
            self.unavailable_reason = (
                "language_tool_python is not installed. Install it to use --grammar, and ensure a JRE is available."
            )
            logger.error(self.unavailable_reason)
            raise RuntimeError(self.unavailable_reason) from exc
        try:
            self._tool = language_tool_python.LanguageTool(self.language)
        except Exception as exc:
            self.unavailable_reason = (
                f"Failed to start LanguageTool ({type(exc).__name__}: {exc}). "
                "Java may be missing or the language code is invalid."
            )
            logger.error(self.unavailable_reason)
            raise RuntimeError(self.unavailable_reason) from exc

    def close(self) -> None:
        if self._tool is not None:
            try:
                self._tool.close()
            except Exception:
                pass
            self._tool = None

    def analyze(self, text: str | None, tokenizer: Any | None = None) -> dict[str, Any]:
        raw = text or ""
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if digest in self._cache:
            return dict(self._cache[digest])
        matches = self._tool.check(raw) if raw.strip() else []
        n_issues = len(matches)
        n_words = count_words(raw)
        n_tokens = count_tokens(raw, tokenizer)
        by_category: dict[str, int] = {}
        by_rule: dict[str, int] = {}
        grouped = dict.fromkeys(CATEGORY_GROUPS, 0)
        for match in matches:
            category = getattr(match, "category", None) or getattr(match, "ruleIssueType", None) or "unknown"
            category_s = str(category)
            by_category[category_s] = by_category.get(category_s, 0) + 1
            rule_id = getattr(match, "ruleId", None) or "unknown"
            by_rule[str(rule_id)] = by_rule.get(str(rule_id), 0) + 1
            cat_upper = category_s.upper()
            for group, names in CATEGORY_GROUPS.items():
                if any(name in cat_upper for name in names):
                    grouped[group] += 1
        result = {
            "grammar_issue_count": n_issues,
            "grammar_issues_per_100_words": (100.0 * n_issues / n_words) if n_words else math.nan,
            "grammar_issues_per_100_tokens": (100.0 * n_issues / n_tokens) if n_tokens else math.nan,
            "grammar_category_counts": by_category,
            "grammar_rule_counts": by_rule,
            "grammar_grammar_count": grouped["grammar"],
            "grammar_spelling_count": grouped["spelling"],
            "grammar_punctuation_count": grouped["punctuation"],
            "grammar_style_count": grouped["style"],
            "grammar_capitalization_count": grouped["capitalization"],
        }
        self._cache[digest] = result
        return dict(result)


def grammar_columns_for_text(
    analyzer: GrammarAnalyzer | None,
    text: str | None,
    prefix: str,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    empty = {
        f"{prefix}grammar_issue_count": math.nan,
        f"{prefix}grammar_issues_per_100_words": math.nan,
        f"{prefix}grammar_issues_per_100_tokens": math.nan,
        f"{prefix}grammar_grammar_count": math.nan,
        f"{prefix}grammar_spelling_count": math.nan,
        f"{prefix}grammar_punctuation_count": math.nan,
        f"{prefix}grammar_style_count": math.nan,
        f"{prefix}grammar_capitalization_count": math.nan,
    }
    if analyzer is None:
        return empty
    stats = analyzer.analyze(text, tokenizer=tokenizer)
    return {
        f"{prefix}grammar_issue_count": stats["grammar_issue_count"],
        f"{prefix}grammar_issues_per_100_words": stats["grammar_issues_per_100_words"],
        f"{prefix}grammar_issues_per_100_tokens": stats["grammar_issues_per_100_tokens"],
        f"{prefix}grammar_grammar_count": stats["grammar_grammar_count"],
        f"{prefix}grammar_spelling_count": stats["grammar_spelling_count"],
        f"{prefix}grammar_punctuation_count": stats["grammar_punctuation_count"],
        f"{prefix}grammar_style_count": stats["grammar_style_count"],
        f"{prefix}grammar_capitalization_count": stats["grammar_capitalization_count"],
        f"{prefix}grammar_rule_counts": stats["grammar_rule_counts"],
    }
