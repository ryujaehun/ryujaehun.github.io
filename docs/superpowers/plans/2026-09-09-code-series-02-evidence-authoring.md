# Code Series Evidence and Authoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consume a registered guide and produce an unpublished, bilingual, evidence-validated series that passes an independent whole-series review.

**Architecture:** A persistent file queue leases one small task per worker run. Generic opencode execution moves into `core`, while task-specific prompts and validators remain in `code_series`; every generated claim, brief, article, and review is schema-checked and atomically stored.

**Tech Stack:** Python 3.10+, opencode CLI JSONL mode, PyYAML, jsonschema, pytest, existing Hugo-aware lint/translation utilities

**Spec:** `docs/superpowers/specs/2026-09-09-code-series-automation-design.md`

## Global Constraints

- Complete `2026-09-09-code-series-01-guide-foundation.md` first.
- One worker invocation performs at most one queue task.
- Every opencode invocation uses a fresh session and a dedicated work directory.
- Generated output is accepted from the promised file, never from a short status message.
- Claims distinguish `verified`, `documented`, `inferred`, `conflicted`, and `unknown`.
- Article tasks see guide, glossary, their own brief/evidence, and prior chapter summaries only.
- No chapter is publishable until Korean, English, and whole-series review pass.
- Preserve the existing papers behavior and tests while extracting shared code.

---

### Task 1: Extract the generic opencode runner

**Files:**
- Create: `automation/core/agent_runner.py`
- Modify: `automation/tasks/papers/summarize.py`
- Create: `automation/tests/test_agent_runner.py`
- Modify: `automation/tests/papers/test_summarize.py`

**Interfaces:**
- Produces: `AgentRunError`, `AgentResult(body, model, cost)`, `build_command(prompt, model, workdir, attachments=(), variant=None)`, `run_agent(prompt, model, workdir, output_path, attachments=(), variant=None, timeout=900, min_chars=1, runner=subprocess.run)`, `run_model_chain(models, **kwargs)`
- Preserves: `tasks.papers.summarize.SummarizeError`, `Summary`, `summarize_with_opencode`, `summarize_with_models`

- [ ] **Step 1: Copy behavior tests to the generic interface before moving code**

```python
def test_run_agent_prefers_the_promised_output_file(tmp_path):
    out = tmp_path / "result.md"
    runner = agent_writing(out, "# Article\n\n" + "body " * 500)
    result = run_agent("write", "m", tmp_path, out, runner=runner, min_chars=100)
    assert result.body.startswith("# Article")
```

Cover JSONL text joining, cost extraction, attachment ordering, timeout, nonzero exit, stale
output removal, short-status rejection, absolute paths, variant, and model fallback.

- [ ] **Step 2: Verify the generic tests fail and papers tests pass**

Run: `cd automation && uv run pytest tests/test_agent_runner.py tests/papers/test_summarize.py -q`

Expected: generic tests FAIL on missing module; existing papers tests PASS.

- [ ] **Step 3: Move generic behavior and leave compatibility wrappers**

```python
def summarize_with_opencode(pdf_path, prompt, model, workdir, output_path, **kwargs):
    try:
        result = run_agent(
            prompt=prompt,
            model=model,
            workdir=workdir,
            output_path=output_path,
            attachments=(() if pdf_path is None else (pdf_path,)),
            **kwargs,
        )
    except AgentRunError as exc:
        raise SummarizeError(str(exc)) from exc
    return Summary(result.body, result.model, result.cost)
```

- [ ] **Step 4: Run both suites and lint**

Run: `cd automation && uv run pytest tests/test_agent_runner.py tests/papers/test_summarize.py tests/papers/test_translate.py -q && uv run ruff check core/agent_runner.py tasks/papers/summarize.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add automation/core/agent_runner.py automation/tasks/papers/summarize.py automation/tests/test_agent_runner.py automation/tests/papers/test_summarize.py
git commit -m "refactor: share the opencode task runner"
```

### Task 2: Durable task queue and lease recovery

**Files:**
- Create: `automation/tasks/code_series/queue.py`
- Extend: `automation/tasks/code_series/state.py`
- Create: `automation/tests/code_series/test_queue.py`

**Interfaces:**
- Produces: `TaskKind`, `TaskRecord`, `enqueue_series(project_dir)`, `lease_next(project_dir, now, lease_seconds)`, `complete_task(project_dir, task_id, outcome)`, `fail_task(project_dir, task_id, reason, retryable)`, `recover_expired_leases(project_dir, now)`

- [ ] **Step 1: Write ordering, lease, and retry tests**

```python
def test_worker_leases_only_one_ready_dependency(queue):
    first = lease_next(queue.project_dir, now=NOW, lease_seconds=900)
    second = lease_next(queue.project_dir, now=NOW, lease_seconds=900)
    assert first.kind == TaskKind.RECON
    assert second is None

def test_same_failure_stops_after_three_attempts(queue):
    task = queue.task("recon-purpose")
    for _ in range(3):
        fail_task(queue.project_dir, task.id, "provider-timeout", retryable=True)
    assert queue.reload().task(task.id).status == "blocked"
```

- [ ] **Step 2: Confirm tests fail**

Run: `cd automation && uv run pytest tests/code_series/test_queue.py -q`

Expected: FAIL on missing queue types.

- [ ] **Step 3: Implement atomic queue mutation with explicit dependencies**

```python
class TaskKind(str, Enum):
    RECON = "recon"
    TRACE = "trace"
    EVIDENCE = "evidence"
    BRIEF = "brief"
    WRITE_KO = "write-ko"
    REVIEW_KO = "review-ko"
    TRANSLATE_EN = "translate-en"
    REVIEW_EN = "review-en"
    SERIES_REVIEW = "series-review"
```

Use an exclusive lock file plus temporary-file replacement. A task is ready only when every
dependency is `complete`; expired leases return to `pending` without increasing attempts.

- [ ] **Step 4: Run queue/state tests**

Run: `cd automation && uv run pytest tests/code_series/test_queue.py tests/code_series/test_state.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add automation/tasks/code_series/queue.py automation/tasks/code_series/state.py automation/tests/code_series/test_queue.py
git commit -m "feat: queue resumable code analysis tasks"
```

### Task 3: Reconnaissance and execution-trace workers

**Files:**
- Create: `automation/tasks/code_series/runner.py`
- Create: `automation/tasks/code_series/defaults/prompts/recon.md`
- Create: `automation/tasks/code_series/defaults/prompts/trace.md`
- Create: `automation/tests/code_series/test_runner_recon.py`

**Interfaces:**
- Consumes: `TaskRecord`, inventory JSONL, `core.agent_runner.run_model_chain`
- Produces: `run_recon(task, context)`, `run_trace(task, context)`, `TaskOutcome(paths, cost, model)`

- [ ] **Step 1: Write input-scope and output-contract tests**

```python
def test_recon_prompt_contains_only_declared_inventory_files(recon_context):
    invocation = build_recon_invocation(recon_context.task, recon_context)
    assert set(invocation.attachments) == set(recon_context.task.input_paths)
    assert str(recon_context.snapshot.root) not in invocation.prompt

def test_trace_rejects_symbols_outside_pinned_inventory(trace_context, fake_agent):
    fake_agent.body = "uses missing.py::invented"
    with pytest.raises(OutputInvalid, match="missing.py"):
        run_trace(trace_context.task, trace_context)
```

- [ ] **Step 2: Run focused tests and observe failure**

Run: `cd automation && uv run pytest tests/code_series/test_runner_recon.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement bounded context builders and Markdown validators**

```python
@dataclass(frozen=True)
class TaskOutcome:
    paths: tuple[str, ...]
    cost: float | None
    model: str
```

Recon prompts must label observations as code, docs, tests, run, or inference and list source
paths. Trace output must include entrypoint, ordered steps, state changes, branching, exit,
and unresolved gaps.

- [ ] **Step 4: Run tests and lint**

Run: `cd automation && uv run pytest tests/code_series/test_runner_recon.py tests/test_agent_runner.py -q && uv run ruff check tasks/code_series/runner.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add automation/tasks/code_series/runner.py automation/tasks/code_series/defaults/prompts automation/tests/code_series/test_runner_recon.py
git commit -m "feat: research code repositories in bounded tasks"
```

### Task 4: Evidence and chapter brief contracts

**Files:**
- Create: `automation/tasks/code_series/schemas/claim.schema.json`
- Create: `automation/tasks/code_series/evidence.py`
- Create: `automation/tasks/code_series/defaults/prompts/evidence.md`
- Create: `automation/tasks/code_series/defaults/prompts/brief.md`
- Create: `automation/tests/code_series/test_evidence.py`

**Interfaces:**
- Produces: `Claim`, `EvidenceRef`, `validate_claims(rows, snapshot, inventory)`, `run_evidence(task, context)`, `run_brief(task, context)`

- [ ] **Step 1: Write provenance and confidence tests**

```python
def test_claim_permalink_must_pin_the_project_commit(claim, snapshot, inventory):
    claim["evidence"][0]["permalink"] = "https://github.com/o/r/blob/main/a.py#L1"
    errors = validate_claims([claim], snapshot, inventory)
    assert any("pinned commit" in error for error in errors)

def test_inferred_claim_requires_a_limitation(claim):
    claim["confidence"] = "inferred"
    claim["limitations"] = []
    assert validate_claims([claim], SNAPSHOT, INVENTORY)
```

Also reject nonexistent paths, out-of-range lines, unknown symbols, invalid confidence, and
strong claims without code evidence.

- [ ] **Step 2: Verify failures**

Run: `cd automation && uv run pytest tests/code_series/test_evidence.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement schema-backed records and brief coverage checks**

```python
CONFIDENCE = {"verified", "documented", "inferred", "conflicted", "unknown"}

def github_permalink(snapshot: Snapshot, path: str, start: int, end: int) -> str:
    quoted = quote(path)
    return f"{snapshot.ref.url}/blob/{snapshot.commit}/{quoted}#L{start}-L{end}"
```

A brief must map every chapter question and requested visual to claim IDs and must list the
chapter's explicit exclusions to prevent duplication.

- [ ] **Step 4: Run evidence, schema, and guide tests**

Run: `cd automation && uv run pytest tests/code_series/test_evidence.py tests/code_series/test_guide.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add automation/tasks/code_series/schemas/claim.schema.json automation/tasks/code_series/evidence.py automation/tasks/code_series/defaults/prompts automation/tests/code_series/test_evidence.py
git commit -m "feat: ground chapter briefs in code evidence"
```

### Task 5: Korean article rendering and independent review

**Files:**
- Create: `automation/tasks/code_series/article.py`
- Create: `automation/tasks/code_series/review.py`
- Create: `automation/tasks/code_series/defaults/prompts/write-ko.md`
- Create: `automation/tasks/code_series/defaults/prompts/review-ko.md`
- Create: `automation/tests/code_series/test_article.py`
- Create: `automation/tests/code_series/test_review.py`

**Interfaces:**
- Produces: `build_article_context`, `run_write_ko`, `validate_article`, `run_review_ko`, `ReviewReport`
- Reuses: `tasks.papers.lint.check`, `tasks.papers.blogconfig.load`

- [ ] **Step 1: Write tests for evidence coverage and visual purpose**

```python
def test_article_rejects_unreferenced_required_claim(article_fixture):
    text = article_fixture.text.replace("scheduler-overlap-001", "")
    errors = validate_article(text, article_fixture.contract)
    assert any("scheduler-overlap-001" in error for error in errors)

def test_mermaid_visual_declares_supporting_claims(article_fixture):
    assert validate_visuals(article_fixture.text, article_fixture.contract) == []
```

Test missing limitations, mutable GitHub links, broken fences/tables/Mermaid, duplicate scope,
and a short status report in place of an article.

- [ ] **Step 2: Confirm tests fail**

Run: `cd automation && uv run pytest tests/code_series/test_article.py tests/code_series/test_review.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement article context, front matter, lint, and review report**

```python
@dataclass(frozen=True)
class ReviewReport:
    passed: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    checked_claim_ids: tuple[str, ...]
```

The writer emits body plus machine-readable claim comments; `article.py` adds Hugo front
matter with `draft: true`, series ID, chapter order, repository, and commit. The reviewer
gets the article and evidence but not the writer's model transcript.

- [ ] **Step 4: Run article, lint, and blog config tests**

Run: `cd automation && uv run pytest tests/code_series/test_article.py tests/code_series/test_review.py tests/papers/test_lint.py tests/papers/test_blogconfig.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add automation/tasks/code_series/article.py automation/tasks/code_series/review.py automation/tasks/code_series/defaults/prompts automation/tests/code_series
git commit -m "feat: write and review grounded Korean chapters"
```

### Task 6: English translation and structure preservation

**Files:**
- Create: `automation/tasks/code_series/translate.py`
- Create: `automation/tasks/code_series/defaults/prompts/translate-en.md`
- Create: `automation/tasks/code_series/defaults/prompts/review-en.md`
- Create: `automation/tests/code_series/test_translate.py`

**Interfaces:**
- Produces: `run_translate_en`, `validate_translation`, `run_review_en`
- Reuses: `tasks.papers.translate.structure_of`, `check_structure`, `looks_korean`, `english_path`

- [ ] **Step 1: Write tests for code-series-specific invariants**

```python
def test_translation_preserves_claim_ids_and_pinned_links(korean, english):
    errors = validate_translation(korean, english)
    assert errors == []

def test_translation_rejects_changed_code_or_mermaid_ids(korean, english):
    changed = english.replace("Scheduler.overlap_loop", "Scheduler.loop")
    assert any("code token" in e for e in validate_translation(korean, changed))
```

- [ ] **Step 2: Verify tests fail**

Run: `cd automation && uv run pytest tests/code_series/test_translate.py -q`

Expected: FAIL.

- [ ] **Step 3: Wrap existing structure checks and add exact technical-token checks**

```python
def validate_translation(korean: str, english: str) -> list[str]:
    problems = list(check_structure(korean, english))
    problems.extend(compare_claim_ids(korean, english))
    problems.extend(compare_code_tokens(korean, english))
    return problems
```

Preserve front matter pairing, numbers, equations, table shape, paths, symbols, URLs, code
blocks, Mermaid node IDs, limitations, and confidence labels.

- [ ] **Step 4: Run both translation suites**

Run: `cd automation && uv run pytest tests/code_series/test_translate.py tests/papers/test_translate.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add automation/tasks/code_series/translate.py automation/tasks/code_series/defaults/prompts automation/tests/code_series/test_translate.py
git commit -m "feat: translate code analysis chapters safely"
```

### Task 7: Whole-series review, schedule eligibility, and worker CLI

**Files:**
- Extend: `automation/tasks/code_series/review.py`
- Extend: `automation/tasks/code_series/runner.py`
- Modify: `automation/tasks/code_series/cli.py`
- Create: `automation/tests/code_series/test_series_review.py`
- Create: `automation/tests/code_series/test_worker_cli.py`

**Interfaces:**
- Produces: `review_series(project_dir)`, `mark_series_verified(project_dir)`, CLI commands `work --once` and `finalize PROJECT_SLUG`

- [ ] **Step 1: Write cross-chapter and one-task-per-run tests**

```python
def test_series_review_finds_conflicting_glossary_definitions(series_fixture):
    series_fixture.chapter(2).write("Radix cache means FIFO eviction")
    report = review_series(series_fixture.path)
    assert not report.passed
    assert any("Radix cache" in error for error in report.errors)

def test_work_once_completes_exactly_one_task(worker_cli, queued_project):
    before = queued_project.completed_count
    assert worker_cli("work", "--once").returncode == 0
    assert queued_project.reload().completed_count == before + 1
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `cd automation && uv run pytest tests/code_series/test_series_review.py tests/code_series/test_worker_cli.py -q`

Expected: FAIL.

- [ ] **Step 3: Dispatch task kinds and validate global coverage**

```python
TASK_HANDLERS = {
    TaskKind.RECON: run_recon,
    TaskKind.TRACE: run_trace,
    TaskKind.EVIDENCE: run_evidence,
    TaskKind.BRIEF: run_brief,
    TaskKind.WRITE_KO: run_write_ko,
    TaskKind.REVIEW_KO: run_review_ko,
    TaskKind.TRANSLATE_EN: run_translate_en,
    TaskKind.REVIEW_EN: run_review_en,
    TaskKind.SERIES_REVIEW: run_series_review,
}
```

`finalize` requires every chapter review and translation review, then runs independent
coverage, ordering, terminology, duplication, contradiction, SHA, and scope checks. Success
sets `series-verified`; it does not write to `content/posts`.

- [ ] **Step 4: Run the entire unpublished-series suite**

Run: `cd automation && uv run pytest tests/code_series tests/test_agent_runner.py tests/papers/test_summarize.py tests/papers/test_translate.py tests/papers/test_lint.py -q && uv run ruff check core tasks tests`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add automation/tasks/code_series automation/tests/code_series
git commit -m "feat: complete unpublished code analysis series"
```
