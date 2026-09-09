# Code Series Guide Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a GitHub URL into a pinned, inventoried project workspace that an interactive Claude/Codex skill can use to create and register a valid series guide.

**Architecture:** Deterministic Python modules own repository acquisition, inventory, compatibility, schemas, and state. A single Agent Skills package orchestrates research and user questions, then calls the internal CLI to register its Markdown and YAML outputs. No model call, article generation, or publishing belongs in this slice.

**Tech Stack:** Python 3.10+, standard library, PyYAML, jsonschema, pytest, git CLI, Agent Skills (`SKILL.md`)

**Spec:** `docs/superpowers/specs/2026-09-09-code-series-automation-design.md`

## Global Constraints

- The user-facing interface is a natural-language request containing a GitHub URL.
- Every analysis is pinned to one full 40-character commit SHA.
- Guide creation must ask about consequential ambiguity and store answers in `decision-log.md`.
- A guide with unresolved questions must not enter the work queue.
- Repository code is untrusted: do not execute hooks, setup scripts, or submodules.
- Outputs live below `data/code-series/<project-slug>/`; clones live below `.cache/code-series/`.
- Workspace paths must go through `core.workspace.Workspace` and `tasks.code_series.layout`.
- Use atomic writes for committed state and manifests.
- Do not add AST frameworks or support non-GitHub for this slice.

---

### Task 1: Package layout and configuration contract

**Files:**
- Create: `automation/tasks/code_series/__init__.py`
- Create: `automation/tasks/code_series/layout.py`
- Create: `automation/tasks/code_series/config.py`
- Create: `automation/tasks/code_series/defaults/code-series.yaml`
- Create: `automation/tests/code_series/test_config.py`
- Modify: `automation/requirements.txt`

**Interfaces:**
- Consumes: `core.workspace.Workspace.path(relative)`
- Produces: `CodeSeriesConfig`, `load_config(workspace, explicit=None)`, and layout constants `ROOT`, `REPO_CACHE`, `WORK_CACHE`, `POSTS`, `DEFAULT_CONFIG`

- [ ] **Step 1: Write the failing configuration tests**

```python
def test_default_config_has_bounded_repository_limits(blog_workspace):
    cfg = load_config(Workspace(blog_workspace))
    assert cfg.repository.max_files == 20_000
    assert cfg.repository.max_bytes == 250_000_000
    assert cfg.worker.max_attempts == 3

def test_workspace_config_replaces_the_packaged_default(blog_workspace):
    path = blog_workspace / "data/code-series-config.yaml"
    path.write_text("repository: {max_files: 99, max_bytes: 1000}\nworker: {max_attempts: 2}\n")
    assert load_config(Workspace(blog_workspace)).repository.max_files == 99
```

- [ ] **Step 2: Run the tests and confirm the missing-module failure**

Run: `cd automation && uv run pytest tests/code_series/test_config.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'tasks.code_series'`.

- [ ] **Step 3: Add minimal dataclasses, YAML validation, and paths**

```python
@dataclass(frozen=True)
class RepositoryLimits:
    max_files: int
    max_bytes: int

@dataclass(frozen=True)
class WorkerLimits:
    max_attempts: int

@dataclass(frozen=True)
class CodeSeriesConfig:
    repository: RepositoryLimits
    worker: WorkerLimits
```

Add `jsonschema>=4.23,<5` to `requirements.txt`; later tasks use the same dependency for
`series.yaml` and claim validation. Reject missing, boolean, zero, or negative numeric values
with `ConfigError` that names the YAML key.

- [ ] **Step 4: Run focused and core tests**

Run: `cd automation && uv run pytest tests/code_series/test_config.py tests/test_workspace.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add automation/tasks/code_series automation/tests/code_series/test_config.py automation/requirements.txt
git commit -m "feat: add code series configuration"
```

### Task 2: Safe GitHub snapshot acquisition

**Files:**
- Create: `automation/tasks/code_series/repository.py`
- Create: `automation/tests/code_series/test_repository.py`

**Interfaces:**
- Consumes: `layout.REPO_CACHE`, `RepositoryLimits`
- Produces: `RepoRef`, `Snapshot`, `parse_github_url(url)`, `resolve_default_commit(ref, runner=subprocess.run)`, `checkout_snapshot(ref, commit, cache_root, runner=subprocess.run)`

- [ ] **Step 1: Write URL, SHA, and command-construction tests**

```python
def test_parse_github_url_normalizes_https_and_git_suffix():
    ref = parse_github_url("https://github.com/OpenXLA/shardy.git")
    assert (ref.owner, ref.name, ref.slug) == ("OpenXLA", "shardy", "openxla--shardy")

def test_checkout_disables_hooks_and_submodules(tmp_path, recording_runner):
    checkout_snapshot(RepoRef("o", "r"), "a" * 40, tmp_path, runner=recording_runner)
    joined = [call.args for call in recording_runner.calls]
    assert any("core.hooksPath=/dev/null" in args for args in joined)
    assert all("--recurse-submodules" not in args for args in joined)
```

Also test rejection of SSH, non-GitHub hosts, extra path components, malformed names, and a
non-40-character SHA.

- [ ] **Step 2: Verify the tests fail**

Run: `cd automation && uv run pytest tests/code_series/test_repository.py -q`

Expected: FAIL because `tasks.code_series.repository` does not exist.

- [ ] **Step 3: Implement immutable repository records and injected git calls**

```python
@dataclass(frozen=True)
class RepoRef:
    owner: str
    name: str

    @property
    def url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}"

    @property
    def slug(self) -> str:
        return f"{self.owner.lower()}--{self.name.lower()}"

@dataclass(frozen=True)
class Snapshot:
    ref: RepoRef
    commit: str
    root: Path
```

Use argument arrays, `check=False`, captured text, explicit timeouts, `git -c
core.hooksPath=/dev/null`, `clone --no-checkout --filter=blob:none`, fetch the pinned commit,
and detached checkout. Convert nonzero results to `RepositoryError` with redacted stderr.

- [ ] **Step 4: Run tests and static checks**

Run: `cd automation && uv run pytest tests/code_series/test_repository.py -q && uv run ruff check tasks/code_series/repository.py tests/code_series/test_repository.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add automation/tasks/code_series/repository.py automation/tests/code_series/test_repository.py
git commit -m "feat: pin safe GitHub snapshots"
```

### Task 3: Deterministic repository inventory

**Files:**
- Create: `automation/tasks/code_series/inventory.py`
- Create: `automation/tests/code_series/test_inventory.py`
- Create: `automation/tests/code_series/fixtures/mixed-repo/README.md`
- Create: `automation/tests/code_series/fixtures/mixed-repo/src/main.py`
- Create: `automation/tests/code_series/fixtures/mixed-repo/include/engine.h`
- Create: `automation/tests/code_series/fixtures/mixed-repo/vendor/generated.cc`

**Interfaces:**
- Consumes: `Snapshot.root`, `Snapshot.commit`, `core.jsonl.write_jsonl`
- Produces: `FileRecord`, `SymbolRecord`, `DependencyRecord`, `Inventory`, `build_inventory(snapshot)`, `write_inventory(inventory, output_dir)`

- [ ] **Step 1: Write tests for classification and stable output**

```python
def test_inventory_marks_vendor_and_extracts_python_symbols(snapshot_fixture):
    inv = build_inventory(snapshot_fixture)
    records = {item.path: item for item in inv.files}
    assert records["vendor/generated.cc"].excluded_reason == "vendor"
    assert any(s.path == "src/main.py" and s.name == "main" for s in inv.symbols)

def test_inventory_is_sorted_and_never_follows_external_symlink(snapshot_fixture):
    inv = build_inventory(snapshot_fixture)
    assert [f.path for f in inv.files] == sorted(f.path for f in inv.files)
    assert "outside.txt" not in {f.path for f in inv.files}
```

- [ ] **Step 2: Verify failure before implementation**

Run: `cd automation && uv run pytest tests/code_series/test_inventory.py -q`

Expected: FAIL on the missing `inventory` module.

- [ ] **Step 3: Implement one-pass file inventory and small extractors**

```python
@dataclass(frozen=True)
class FileRecord:
    path: str
    size: int
    lines: int | None
    language: str | None
    sha256: str
    role: str
    excluded_reason: str | None
```

Read tracked paths through `git ls-files -z` so `.gitignore` and symlinks are handled
consistently. Add conservative regex extractors for Python definitions/imports, C/C++/CUDA
definitions/includes, MLIR/TableGen declarations, Bazel labels, docs, tests, build files, and
entrypoints. Store unknown syntax as files without inventing symbols.

- [ ] **Step 4: Run focused tests and JSONL regressions**

Run: `cd automation && uv run pytest tests/code_series/test_inventory.py tests/test_jsonl.py -q`

Expected: PASS with byte-identical output across two runs.

- [ ] **Step 5: Commit**

```bash
git add automation/tasks/code_series/inventory.py automation/tests/code_series
git commit -m "feat: inventory code analysis repositories"
```

### Task 4: Compatibility assessment and report

**Files:**
- Create: `automation/tasks/code_series/compatibility.py`
- Create: `automation/tests/code_series/test_compatibility.py`

**Interfaces:**
- Consumes: `Inventory`, `CodeSeriesConfig`
- Produces: `CompatibilityStatus`, `CompatibilityFinding`, `CompatibilityReport`, `assess(inventory, config)`, `render_report(report)`

- [ ] **Step 1: Write threshold and atypical-shape tests**

```python
def test_large_repository_requires_review(config, inventory_factory):
    report = assess(inventory_factory(file_count=20_001), config)
    assert report.status == CompatibilityStatus.NEEDS_REVIEW
    assert "file-count" in {f.code for f in report.findings}

def test_unsupported_core_language_is_not_silently_supported(config, inventory_factory):
    report = assess(inventory_factory(core_languages={"solidity": 0.8}), config)
    assert report.status == CompatibilityStatus.NEEDS_REVIEW
```

Cover monorepo roots, vendor ratio, submodules, sparse docs, execution prerequisites, and a
small supported Python project.

- [ ] **Step 2: Verify missing-module failures**

Run: `cd automation && uv run pytest tests/code_series/test_compatibility.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement deterministic rules and actionable Markdown**

```python
class CompatibilityStatus(str, Enum):
    SUPPORTED = "supported"
    ADAPTED = "adapted"
    NEEDS_REVIEW = "needs-review"
    BLOCKED = "blocked"
```

Every finding must include `code`, observed value, threshold, consequence, and recommended
guide question. `render_report` must state analyzable scope, omissions, extension proposal,
and expected cost/series-size impact.

- [ ] **Step 4: Run compatibility and inventory tests**

Run: `cd automation && uv run pytest tests/code_series/test_compatibility.py tests/code_series/test_inventory.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add automation/tasks/code_series/compatibility.py automation/tests/code_series/test_compatibility.py
git commit -m "feat: report atypical code repositories"
```

### Task 5: Guide manifests, decision log, and registration state

**Files:**
- Create: `automation/tasks/code_series/schemas/series.schema.json`
- Create: `automation/tasks/code_series/guide.py`
- Create: `automation/tasks/code_series/state.py`
- Create: `automation/tests/code_series/test_guide.py`
- Create: `automation/tests/code_series/test_state.py`

**Interfaces:**
- Consumes: `jsonschema.validate`, `Workspace`, `Snapshot`, `CompatibilityReport`
- Produces: `GuideValidation`, `validate_guide(project_dir)`, `register_guide(project_dir)`, `ProjectState.load(path)`, `ProjectState.save()`, `ProjectState.transition(next_status)`

- [ ] **Step 1: Write tests that reject an unresolved or ungrounded guide**

```python
def test_unresolved_questions_block_registration(valid_project):
    valid_project.series["open_questions"] = ["Which runtime path is primary?"]
    result = validate_guide(valid_project.path)
    assert "open_questions must be empty" in result.errors

def test_registration_pins_the_manifest_commit(valid_project):
    state = register_guide(valid_project.path)
    assert state.status == "queued"
    assert state.commit == "a" * 40
```

Test missing guide headings, invalid chapter order, unknown dependencies, unscoped chapters,
missing decision-log entries, and atomic state round trips.

- [ ] **Step 2: Run tests and confirm failure**

Run: `cd automation && uv run pytest tests/code_series/test_guide.py tests/code_series/test_state.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement schema validation and an explicit transition table**

```python
ALLOWED_TRANSITIONS = {
    "preflight": {"needs-guidance", "queued", "blocked"},
    "needs-guidance": {"preflight"},
    "queued": {"researching", "blocked"},
}
```

Registration writes `project.yaml`, validates `guide.md`, `decision-log.md`,
`compatibility-report.md`, and `series.yaml`, then atomically creates `state.json`. It never
registers a `needs-review` project unless the decision log records the user's resolution.

- [ ] **Step 4: Run the new tests**

Run: `cd automation && uv run pytest tests/code_series/test_guide.py tests/code_series/test_state.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add automation/tasks/code_series/schemas automation/tasks/code_series/guide.py automation/tasks/code_series/state.py automation/tests/code_series
git commit -m "feat: validate and register series guides"
```

### Task 6: Internal prepare/register/status CLI

**Files:**
- Create: `automation/tasks/code_series/cli.py`
- Create: `automation/tests/code_series/test_cli.py`

**Interfaces:**
- Consumes: `Workspace.resolve`, `parse_github_url`, `checkout_snapshot`, `build_inventory`, `assess`, `validate_guide`, `register_guide`, `core.log.emit`
- Produces: `PreparedProject`, `prepare_project(workspace, url)`, `read_status(workspace, project_slug=None)`, and commands `prepare URL`, `register PROJECT_SLUG`, `status [PROJECT_SLUG]`; all accept `--workspace` and `--json`

- [ ] **Step 1: Write CLI orchestration tests with injected fakes**

```python
def test_prepare_emits_paths_the_agent_needs(cli_runner, fake_repository):
    result = cli_runner("prepare", "https://github.com/o/r", "--json")
    payload = json.loads(result.stdout)
    assert payload["project_slug"] == "o--r"
    assert payload["commit"] == "a" * 40
    assert payload["next_action"] == "write-guide"

def test_register_refuses_open_questions(cli_runner, prepared_project):
    result = cli_runner("register", prepared_project.slug)
    assert result.returncode == EXIT_CONFIG
```

- [ ] **Step 2: Verify the CLI tests fail**

Run: `cd automation && uv run pytest tests/code_series/test_cli.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement thin command handlers and JSON stdout discipline**

```python
def cmd_prepare(args) -> int:
    prepared = prepare_project(args.ws, args.url)
    emit(prepared.as_dict(), args.json)
    return EXIT_OK

def cmd_register(args) -> int:
    state = register_guide(args.ws.path(layout.ROOT) / args.project_slug)
    emit(state.as_dict(), args.json)
    return EXIT_OK

def cmd_status(args) -> int:
    emit(read_status(args.ws, args.project_slug), args.json)
    return EXIT_OK

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="code-series")
    parser.add_argument("--workspace")
    parser.add_argument("--json", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("url")
    prepare.set_defaults(func=cmd_prepare)
    register = sub.add_parser("register")
    register.add_argument("project_slug")
    register.set_defaults(func=cmd_register)
    status = sub.add_parser("status")
    status.add_argument("project_slug", nargs="?")
    status.set_defaults(func=cmd_status)
    return parser

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.ws = Workspace.resolve(args.workspace)
    return args.func(args)
```

Human logs go to stderr via `core.log.log`; JSON is the only stdout content under `--json`.
Return existing `EXIT_OK`, `EXIT_NETWORK`, and `EXIT_CONFIG` values rather than inventing an
uncoordinated exit-code set.

- [ ] **Step 4: Run CLI and foundation tests**

Run: `cd automation && uv run pytest tests/code_series tests/test_workspace.py tests/test_jsonl.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add automation/tasks/code_series/cli.py automation/tests/code_series/test_cli.py
git commit -m "feat: add guide preparation CLI"
```

### Task 7: Shared Claude/Codex skill and guide documentation

**Files:**
- Create: `automation/skills/code-series-guide/SKILL.md`
- Create: `automation/skills/code-series-guide/references/guide-contract.md`
- Create: `automation/skills/code-series-guide/references/questioning-policy.md`
- Create: `automation/skills/code-series-guide/references/compatibility-policy.md`
- Create: `automation/skills/code-series-guide/references/evidence-policy.md`
- Create: `automation/skills/code-series-guide/assets/guide-template.md`
- Create: `automation/skills/code-series-guide/assets/series-template.yaml`
- Create: `automation/skills/code-series-guide/assets/decision-log-template.md`
- Create: `.agents/skills/code-series-guide` (symlink)
- Create: `.claude/skills/code-series-guide` (symlink)
- Create: `automation/tasks/code_series/README.md`
- Modify: `automation/README.md`
- Create: `automation/tests/code_series/test_skill.py`

**Interfaces:**
- Consumes: internal CLI `prepare`, `register`, `status`
- Produces: implicitly triggered `code-series-guide` Agent Skill and a complete operator README

- [ ] **Step 1: Use the skill-creator workflow and write structural tests first**

```python
def test_skill_is_shared_by_claude_and_codex(repo_root):
    canonical = repo_root / "automation/skills/code-series-guide/SKILL.md"
    assert (repo_root / ".agents/skills/code-series-guide/SKILL.md").samefile(canonical)
    assert (repo_root / ".claude/skills/code-series-guide/SKILL.md").samefile(canonical)

def test_skill_requires_questions_before_registration(skill_text):
    assert "decision-log.md" in skill_text
    assert "open_questions" in skill_text
    assert "register" in skill_text
```

- [ ] **Step 2: Run tests and verify missing artifacts fail**

Run: `cd automation && uv run pytest tests/code_series/test_skill.py -q`

Expected: FAIL.

- [ ] **Step 3: Write the canonical skill, references, assets, and relative symlinks**

```yaml
---
name: code-series-guide
description: Create a detailed, evidence-driven blog series guide from a GitHub repository URL. Use when the user asks to analyze an open-source codebase, study project structure, plan a code-reading series, or make a repository guide. Ask the user about consequential ambiguity before registering the guide; do not use for a single-file code review.
---
```

The skill must run `prepare`, read every emitted inventory/report path, ask one grounded
question at a time, update `decision-log.md`, render guide assets, run `register`, and report
the pinned SHA and queue state. It must stop on `blocked` and must not edit the shared
template silently for `needs-review`.

- [ ] **Step 4: Write README examples and run the whole foundation suite**

Run: `cd automation && uv run pytest tests/code_series tests/test_workspace.py tests/test_jsonl.py -q && uv run ruff check core tasks tests`

Expected: PASS. Manually start one fresh Codex and one fresh Claude Code session and confirm
that a natural-language prompt mentioning a GitHub URL discovers the skill before running
any repository command.

- [ ] **Step 5: Commit**

```bash
git add .agents .claude automation/skills automation/tasks/code_series/README.md automation/README.md automation/tests/code_series/test_skill.py
git commit -m "feat: add interactive code series guide skill"
```

### Task 8: Foundation end-to-end test

**Files:**
- Create: `automation/tests/code_series/test_foundation_e2e.py`
- Modify: `automation/tasks/code_series/README.md`

**Interfaces:**
- Consumes: all plan interfaces
- Produces: a network-free proof that prepare, guide authoring fixture, register, and status compose correctly

- [ ] **Step 1: Write an end-to-end test around a local Git fixture**

```python
def test_prepare_then_register_reaches_queued(tmp_blog, local_remote, guide_writer):
    prepared = prepare_project(tmp_blog, local_remote.url)
    guide_writer.write_valid(prepared.project_dir)
    state = register_guide(prepared.project_dir)
    assert state.status == "queued"
    assert state.commit == local_remote.head
```

- [ ] **Step 2: Run it and fix only integration defects**

Run: `cd automation && uv run pytest tests/code_series/test_foundation_e2e.py -q`

Expected before fixes: FAIL only where independently tested modules do not compose.

- [ ] **Step 3: Apply minimal wiring corrections**

```python
def prepare_project(workspace: Workspace, url: str) -> PreparedProject:
    """Create snapshot, inventory, compatibility report, and guide input paths."""
```

Do not add article, model, publisher, or Telegram behavior to make this test pass.

- [ ] **Step 4: Run full automation verification**

Run: `cd automation && uv run pytest -q && uv run ruff check core tasks tests`

Expected: PASS with no network or opencode credentials.

- [ ] **Step 5: Commit**

```bash
git add automation/tasks/code_series automation/tests/code_series automation/tasks/code_series/README.md
git commit -m "test: verify code series guide foundation"
```
