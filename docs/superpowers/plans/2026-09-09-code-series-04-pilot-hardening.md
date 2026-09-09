# Code Series Pilot Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the template on tiny-vLLM, Mini-SGLang, and Shardy, convert discovered structural differences into explicit profiles or compatibility questions, and leave reproducible golden tests and complete documentation.

**Architecture:** Run the same guide workflow from smallest to largest repository at pinned revisions. Preserve sanitized inventories and expected guide shapes as fixtures, add narrowly scoped profiles only when evidence demands them, and never weaken global quality gates to make a pilot pass.

**Tech Stack:** Python 3.10+, pytest, git, Agent Skills, opencode dry-run/fake backends, existing code-series CLI

**Spec:** `docs/superpowers/specs/2026-09-09-code-series-automation-design.md`

## Global Constraints

- Complete plans 01, 02, 02b, and 03 first.
- Pin every pilot to the SHA resolved when its guide session starts.
- Do not commit cloned repositories, model outputs containing secrets, or benchmark artifacts.
- Do not automatically execute pilot repository code on the host.
- A project-specific profile may classify/extract; it may not override evidence or publication gates.
- Normalize only timestamps, temporary paths, and intentionally refreshed SHA fields in goldens.
- If a pilot is `needs-review`, exercise the real guide-question path instead of forcing support.
- Do not publish pilot posts until their entire bilingual series passes the approved pipeline.

---

### Task 1: Golden fixture format and refresh command

**Files:**
- Create: `automation/tasks/code_series/golden.py`
- Create: `automation/tests/code_series/test_golden.py`
- Create: `automation/tests/code_series/fixtures/pilots/README.md`
- Modify: `automation/tasks/code_series/cli.py`

**Interfaces:**
- Produces: `normalize_pilot_output(project_dir)`, `compare_golden(actual, expected)`, CLI command `golden PROJECT_SLUG --output PATH`

- [ ] **Step 1: Write normalization tests that retain semantic changes**

```python
def test_golden_normalization_removes_time_but_keeps_findings(pilot_output):
    normalized = normalize_pilot_output(pilot_output)
    assert "generated_at" not in normalized["project"]
    assert normalized["compatibility"]["findings"] == pilot_output["compatibility"]["findings"]

def test_compare_reports_chapter_and_status_changes(expected, actual):
    actual["chapters"][0]["id"] = "different"
    assert "chapters[0].id" in compare_golden(actual, expected)
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `cd automation && uv run pytest tests/code_series/test_golden.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement a reviewable compact golden projection**

```python
GOLDEN_KEYS = {
    "project": ("slug", "repository", "commit", "languages", "build_systems"),
    "compatibility": ("status", "findings", "questions"),
    "chapters": ("id", "order", "code_scopes", "depends_on", "visuals"),
}
```

The command writes only summaries, paths, counts, finding codes, and chapter contracts; it
must not copy source text or whole generated articles into fixtures.

- [ ] **Step 4: Run golden and CLI tests**

Run: `cd automation && uv run pytest tests/code_series/test_golden.py tests/code_series/test_cli.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add automation/tasks/code_series/golden.py automation/tasks/code_series/cli.py automation/tests/code_series
git commit -m "test: add code series pilot goldens"
```

### Task 2: tiny-vLLM small-monolith pilot

**Files:**
- Create: `automation/tasks/code_series/profiles/small_monolith.py`
- Create: `automation/tests/code_series/fixtures/pilots/tiny-vllm.json`
- Create: `automation/tests/code_series/test_pilot_tiny_vllm.py`
- Modify: `automation/tasks/code_series/compatibility.py`
- Modify: `automation/tasks/code_series/README.md`

**Interfaces:**
- Produces: optional profile `small-monolith`, pinned tiny-vLLM guide and golden summary

- [ ] **Step 1: Run `prepare` on tiny-vLLM without executing its code**

Run: `cd automation && uv run python tasks/code_series/cli.py prepare https://github.com/jmaczan/tiny-vllm --workspace .. --json`

Expected: a full SHA, inventory paths, compatibility status, and no host build/run command.
Record the command output path, not its transient timestamp, in the guide decision log.

- [ ] **Step 2: Use the shared skill to resolve guide questions and register**

Run in a fresh Claude or Codex session:

```text
Use code-series-guide to create a detailed series guide for
https://github.com/jmaczan/tiny-vllm. Do not publish or execute repository code.
```

Expected: the agent asks only consequential questions, writes a guide centered on model
loading, CUDA forward path, prefill/decode, batching, KV cache, and paged attention, then
registers a pinned series.

- [ ] **Step 3: Add the narrow profile only if the inventory proves it is needed**

```python
class SmallMonolithProfile(ProjectProfile):
    name = "small-monolith"

    def applies(self, inventory: Inventory) -> bool:
        return inventory.metrics.source_files <= 40 and inventory.metrics.largest_share >= 0.45
```

The profile recommends function/region-based chapters when one source file dominates; it
does not split files by arbitrary line counts.

- [ ] **Step 4: Generate the golden and assert required guide shape**

```python
def test_tiny_vllm_guide_follows_runtime_not_directories(golden):
    ids = [chapter["id"] for chapter in golden["chapters"]]
    assert "prefill-decode" in ids
    assert "paged-kv-cache" in ids
    assert golden["compatibility"]["status"] in {"supported", "adapted"}
```

Run: `cd automation && uv run pytest tests/code_series/test_pilot_tiny_vllm.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add automation/tasks/code_series/profiles automation/tasks/code_series/compatibility.py automation/tests/code_series/fixtures/pilots/tiny-vllm.json automation/tests/code_series/test_pilot_tiny_vllm.py automation/tasks/code_series/README.md
git commit -m "test: validate the tiny-vllm guide pilot"
```

### Task 3: Mini-SGLang modular-runtime pilot

**Files:**
- Create: `automation/tasks/code_series/profiles/modular_runtime.py`
- Create: `automation/tests/code_series/fixtures/pilots/mini-sglang.json`
- Create: `automation/tests/code_series/test_pilot_mini_sglang.py`
- Modify: `automation/tasks/code_series/compatibility.py`
- Modify: `automation/tasks/code_series/README.md`

**Interfaces:**
- Produces: optional `modular-runtime` profile, pinned Mini-SGLang guide and golden

- [ ] **Step 1: Prepare and inspect deterministic process/module signals**

Run: `cd automation && uv run python tasks/code_series/cli.py prepare https://github.com/sgl-project/mini-sglang --workspace .. --json`

Expected: Python/CUDA languages, package/module boundaries, docs, tests, and multiple runtime
entrypoints in inventory; no GPU execution.

- [ ] **Step 2: Use the shared skill and record user decisions**

```text
Use code-series-guide to create a detailed series guide for
https://github.com/sgl-project/mini-sglang. Ask me where competing runtime flows make the
series direction ambiguous. Do not publish or run GPU code.
```

Expected guide axes: process/message architecture, request lifecycle, scheduler,
prefill/decode, cache/radix behavior, engine/model, attention backends, tensor parallelism,
CUDA graph/overlap, and benchmark/test evidence.

- [ ] **Step 3: Add a profile only for reusable modular-runtime signals**

```python
class ModularRuntimeProfile(ProjectProfile):
    name = "modular-runtime"

    def applies(self, inventory: Inventory) -> bool:
        roles = inventory.metrics.roles
        return roles.get("server", 0) > 0 and roles.get("scheduler", 0) > 0
```

Do not key the profile on the repository name or `minisgl` package string.

- [ ] **Step 4: Generate and test the golden**

```python
def test_mini_sglang_guide_connects_process_and_gpu_paths(golden):
    scopes = "\n".join(path for c in golden["chapters"] for path in c["code_scopes"])
    assert "scheduler" in scopes
    assert "engine" in scopes
    assert "message" in scopes
```

Run: `cd automation && uv run pytest tests/code_series/test_pilot_mini_sglang.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add automation/tasks/code_series/profiles automation/tasks/code_series/compatibility.py automation/tests/code_series/fixtures/pilots/mini-sglang.json automation/tests/code_series/test_pilot_mini_sglang.py automation/tasks/code_series/README.md
git commit -m "test: validate the mini-sglang guide pilot"
```

### Task 4: Shardy compiler/large-repository pilot

**Files:**
- Create: `automation/tasks/code_series/profiles/compiler_pipeline.py`
- Create: `automation/tests/code_series/fixtures/pilots/shardy.json`
- Create: `automation/tests/code_series/test_pilot_shardy.py`
- Modify: `automation/tasks/code_series/inventory.py`
- Modify: `automation/tasks/code_series/compatibility.py`
- Modify: `automation/tasks/code_series/README.md`

**Interfaces:**
- Produces: reusable `compiler-pipeline` profile, MLIR/TableGen/Bazel extraction adjustments,
  pinned Shardy guide or a correct `needs-review` decision path

- [ ] **Step 1: Prepare Shardy and preserve the initial compatibility result**

Run: `cd automation && uv run python tasks/code_series/cli.py prepare https://github.com/openxla/shardy --workspace .. --json`

Expected: inventory reports MLIR, C++, TableGen, Bazel, docs/RFCs, dialect/pass/test groups,
and either `adapted` or `needs-review`. A large input must not be mislabeled `supported` by
silently dropping files.

- [ ] **Step 2: Exercise the guide question loop for scope and seasons**

```text
Use code-series-guide to create a detailed guide for https://github.com/openxla/shardy.
Before registering, ask me about SDY versus MPMD scope, expected prerequisite depth, and
whether the series should be split into seasons. Do not build Shardy.
```

Expected: all decisions appear in `decision-log.md`; `series.yaml.open_questions` is empty
before registration.

- [ ] **Step 3: Implement only evidenced compiler extraction/profile gaps**

```python
class CompilerPipelineProfile(ProjectProfile):
    name = "compiler-pipeline"

    def applies(self, inventory: Inventory) -> bool:
        languages = inventory.metrics.languages
        return "MLIR" in languages and inventory.metrics.roles.get("pass", 0) > 0
```

Extend extraction for observed `Passes.td`, dialect ops/attrs, Bazel targets, pass test files,
and RFC/docs links. Do not parse arbitrary MLIR semantics.

- [ ] **Step 4: Generate and test the golden decision/guide shape**

```python
def test_shardy_guide_surfaces_sdy_mpmd_scope(golden):
    questions = "\n".join(golden["compatibility"]["questions"])
    chapter_text = json.dumps(golden["chapters"])
    assert "SDY" in questions + chapter_text
    assert "MPMD" in questions + chapter_text
    assert golden["compatibility"]["status"] in {"adapted", "needs-review"}
```

Run: `cd automation && uv run pytest tests/code_series/test_pilot_shardy.py -q`

Expected: PASS without reducing file/byte thresholds just for Shardy.

- [ ] **Step 5: Commit**

```bash
git add automation/tasks/code_series/profiles automation/tasks/code_series/inventory.py automation/tasks/code_series/compatibility.py automation/tests/code_series/fixtures/pilots/shardy.json automation/tests/code_series/test_pilot_shardy.py automation/tasks/code_series/README.md
git commit -m "test: validate the Shardy guide pilot"
```

### Task 5: Cross-pilot regression, docs, and release verification

**Files:**
- Create: `automation/tests/code_series/test_pilot_matrix.py`
- Modify: `automation/tasks/code_series/README.md`
- Modify: `automation/README.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: all three goldens and profiles
- Produces: a documented, reproducible support matrix and final acceptance report

- [ ] **Step 1: Write matrix tests that require distinct guides**

```python
def test_pilots_do_not_share_a_forced_chapter_template(all_goldens):
    chapter_sets = [tuple(c["id"] for c in g["chapters"]) for g in all_goldens]
    assert len(set(chapter_sets)) == 3

def test_every_pilot_is_pinned_and_has_visual_evidence(all_goldens):
    for golden in all_goldens:
        assert re.fullmatch(r"[0-9a-f]{40}", golden["project"]["commit"])
        assert any(c["visuals"] for c in golden["chapters"])
```

- [ ] **Step 2: Run the pilot matrix and all code-series tests**

Run: `cd automation && uv run pytest tests/code_series -q`

Expected: PASS; failures must show semantic field paths rather than dumping entire goldens.

- [ ] **Step 3: Complete the support/extension documentation**

Document the three observed shapes, how profiles are selected, what remains unsupported,
how to refresh a pinned pilot intentionally, how to inspect compatibility reports, how guide
questions block registration, and how to add a language/build extractor without weakening
generic rules.

- [ ] **Step 4: Run final repository verification**

Run: `cd automation && uv run pytest -q && uv run ruff check core tasks tests`

Run from the blog root: `hugo --gc --minify --printPathWarnings`

Expected: all tests PASS, ruff reports no findings, Hugo exits 0, and no new duplicate output
path warning names a code-series post.

- [ ] **Step 5: Inspect security and dirty-tree boundaries**

Run: `git status --short && git ls-files '.cache/code-series/**' 'data/code-series/**'`

Expected: no cache clone is tracked; only intended guide/evidence/state data is tracked; no
Telegram token, absolute home path, cloned source, or opencode credential appears in the diff.

- [ ] **Step 6: Commit**

```bash
git add automation/tests/code_series/test_pilot_matrix.py automation/tasks/code_series/README.md automation/README.md README.md
git commit -m "docs: complete code series pilot guidance"
```
