from paperlib.state import State, reviewed_ids


def test_reviewed_ids_reads_both_filenames_and_body_links(tmp_path):
    (tmp_path / "2025-12-15-paper-2506.19852v1.md").write_text(
        "---\ntitle: x\n---\nno link here\n", encoding="utf-8"
    )
    (tmp_path / "2024-01-01-some-post.md").write_text(
        "see https://arxiv.org/abs/2411.02820v4 and https://arxiv.org/pdf/2405.21060\n",
        encoding="utf-8",
    )
    (tmp_path / "2021-2-11-zsh.md").write_text("nothing relevant\n", encoding="utf-8")

    assert reviewed_ids(tmp_path) == {"2506.19852", "2411.02820", "2405.21060"}


def test_reviewed_ids_ignores_version_suffixes(tmp_path):
    (tmp_path / "a-paper-2506.19852v1.md").write_text("x", encoding="utf-8")
    (tmp_path / "b-paper-2506.19852v3.md").write_text("x", encoding="utf-8")

    assert reviewed_ids(tmp_path) == {"2506.19852"}


def test_state_round_trips_through_disk(tmp_path):
    path = tmp_path / "state.json"
    state = State.load(path)
    state.mark_materialized("2609.03430", score=0.71, draft_path="content/posts/x.md")
    state.save()

    reloaded = State.load(path)

    assert reloaded.entries["2609.03430"]["status"] == "materialized"
    assert reloaded.entries["2609.03430"]["score"] == 0.71


def test_state_excludes_materialized_and_repeatedly_failed_ids(tmp_path):
    state = State.load(tmp_path / "state.json")
    state.mark_materialized("done", score=0.9, draft_path="x.md")
    for _ in range(3):
        state.mark_failed("broken", reason="pdf 404")
    state.mark_failed("flaky", reason="timeout")

    excluded = state.excluded_ids(max_failures=3)

    assert "done" in excluded
    assert "broken" in excluded
    assert "flaky" not in excluded
