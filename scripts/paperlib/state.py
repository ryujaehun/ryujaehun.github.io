"""무엇을 이미 다뤘는지 추적한다.

이미 리뷰한 논문은 별도 DB 없이 `content/posts/` 를 매 실행 스캔해서
알아낸다. 글을 지우거나 옮겨도 다음 실행에 자동으로 반영된다.
materialize 이력만 state.json 에 남긴다.
"""

import json
import os
import re
from datetime import date
from pathlib import Path

FILENAME_ID_RE = re.compile(r"paper-(\d{4}\.\d{4,5})")
BODY_LINK_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})")


def reviewed_ids(posts_dir):
    """이미 글로 쓴 arXiv ID 집합. 버전 접미사는 무시한다."""
    found = set()
    for path in Path(posts_dir).glob("*.md"):
        found.update(FILENAME_ID_RE.findall(path.name))
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        found.update(BODY_LINK_RE.findall(text))
    return found


class State:
    """materialize 이력. 실패 횟수를 세어 재시도 폭주를 막는다."""

    def __init__(self, path, entries=None):
        self.path = Path(path)
        self.entries = entries or {}

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            return cls(path)
        with path.open(encoding="utf-8") as f:
            return cls(path, json.load(f))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=1, sort_keys=True)
            f.write("\n")
        os.replace(tmp, self.path)

    def _entry(self, paper_id):
        return self.entries.setdefault(
            paper_id,
            {"first_seen": date.today().isoformat(), "status": "seen", "failures": 0},
        )

    def mark_materialized(self, paper_id, score, draft_path):
        entry = self._entry(paper_id)
        entry.update(
            {
                "materialized_at": date.today().isoformat(),
                "score": score,
                "draft_path": str(draft_path),
                "status": "materialized",
            }
        )

    def mark_failed(self, paper_id, reason):
        entry = self._entry(paper_id)
        entry["failures"] = entry.get("failures", 0) + 1
        entry["status"] = "failed"
        entry["last_error"] = reason

    def excluded_ids(self, max_failures=3):
        return {
            paper_id
            for paper_id, entry in self.entries.items()
            if entry.get("status") == "materialized"
            or entry.get("failures", 0) >= max_failures
        }
