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


def test_repo_relative_keeps_committed_state_portable(tmp_path):
    from paperlib.state import repo_relative

    root = tmp_path / "blog"
    (root / "content" / "posts").mkdir(parents=True)
    draft = root / "content" / "posts" / "x.md"

    # state.json 은 커밋되므로 절대 경로가 들어가면 홈 디렉터리가 새어 나간다
    assert repo_relative(draft, root) == "content/posts/x.md"
    # 리포 밖 경로는 그대로 둔다
    assert repo_relative(tmp_path / "elsewhere.md", root) == str(tmp_path / "elsewhere.md")


def test_mark_materialized_keeps_the_metadata_the_task_backend_needs(tmp_path):
    state = State.load(tmp_path / "state.json")

    state.mark_materialized(
        "2609.03430",
        score=0.89,
        draft_path="content/posts/x.md",
        title="Random Attention: Rethinking KV Cache Eviction",
        version="v1",
        matched={"core": ["kv cache", "cache eviction"]},
    )
    state.save()
    entry = State.load(tmp_path / "state.json").entries["2609.03430"]

    # 반자동 경로에서 에이전트가 논문을 식별하려면 제목과 근거가 있어야 한다
    assert entry["title"].startswith("Random Attention")
    assert entry["version"] == "v1"
    assert entry["matched"]["core"] == ["kv cache", "cache eviction"]
