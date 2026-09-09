# Code Series Execution Evidence and Assets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe, reproducible runtime evidence and licensed visual assets to the unpublished series pipeline without executing untrusted repositories on the host.

**Architecture:** Guide-approved execution recipes run through an injected sandbox backend with read-only source, resource limits, no secrets, and network disabled by default. Every result records environment provenance; unavailable execution becomes an explicit limitation. Asset handling accepts upstream images only when a deterministic license policy permits them.

**Tech Stack:** Python 3.10+, container CLI abstraction, pytest fakes, PyYAML, Pillow only for already-allowed raster conversion

**Spec:** `docs/superpowers/specs/2026-09-09-code-series-automation-design.md`

## Global Constraints

- Complete plans 01 and 02 first.
- Never run cloned repository commands directly on the host.
- Default sandbox policy is read-only source, writable scratch, no secrets, network off,
  non-root user, explicit CPU/memory/process/time limits.
- Network, GPU, model downloads, and private data require an explicit guide decision.
- A failed or unavailable runtime check is a limitation, not evidence that the code is wrong.
- Never present upstream benchmark data as a local measurement.
- Reuse an upstream image only when its applicable license and attribution are known.

---

### Task 1: Execution recipe and provenance schemas

**Files:**
- Create: `automation/tasks/code_series/schemas/execution.schema.json`
- Create: `automation/tasks/code_series/execution.py`
- Create: `automation/tests/code_series/test_execution_schema.py`

**Interfaces:**
- Produces: `ExecutionRecipe`, `ResourceLimits`, `ExecutionProvenance`, `ExecutionResult`, `validate_recipe(recipe, guide)`, `validate_provenance(result)`

- [ ] **Step 1: Write tests for denied capabilities and required provenance**

```python
def test_network_recipe_requires_a_matching_guide_decision(base_recipe, guide):
    base_recipe["network"] = True
    assert "network was not approved" in validate_recipe(base_recipe, guide)

def test_successful_measurement_requires_hardware_and_repetitions(result):
    result["measurement"] = {"metric": "tokens/s", "value": 10}
    errors = validate_provenance(result)
    assert any("repetitions" in error for error in errors)
    assert any("hardware" in error for error in errors)
```

Cover commit, command array, image digest, OS, architecture, toolchain, GPU/CPU/memory,
configuration, input description, exit code, stdout/stderr paths, duration, repetitions, and
summary method.

- [ ] **Step 2: Run tests and confirm failure**

Run: `cd automation && uv run pytest tests/code_series/test_execution_schema.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement immutable records and schema validation**

```python
@dataclass(frozen=True)
class ResourceLimits:
    cpus: float = 2.0
    memory_bytes: int = 4 * 1024**3
    pids: int = 256
    timeout_seconds: int = 900

@dataclass(frozen=True)
class ExecutionRecipe:
    command: tuple[str, ...]
    image: str
    network: bool
    gpu: bool
    limits: ResourceLimits
```

Reject shell strings; commands are argument arrays. Require immutable container image
digests for results used as measured evidence.

- [ ] **Step 4: Run schema tests and lint**

Run: `cd automation && uv run pytest tests/code_series/test_execution_schema.py -q && uv run ruff check tasks/code_series/execution.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add automation/tasks/code_series/schemas/execution.schema.json automation/tasks/code_series/execution.py automation/tests/code_series/test_execution_schema.py
git commit -m "feat: define reproducible execution evidence"
```

### Task 2: Sandboxed execution backend

**Files:**
- Extend: `automation/tasks/code_series/execution.py`
- Create: `automation/tests/code_series/test_execution_sandbox.py`

**Interfaces:**
- Produces: `SandboxBackend.run(recipe, snapshot, scratch)`, `ContainerBackend`, `UnavailableBackend`, `run_execution_task(task, context, backend)`

- [ ] **Step 1: Write exact container-policy tests**

```python
def test_container_is_read_only_unprivileged_and_offline(snapshot, recipe, recorder):
    ContainerBackend(runner=recorder).run(recipe, snapshot, snapshot.root.parent / "scratch")
    command = recorder.calls[0]
    assert "--read-only" in command
    assert "--network=none" in command
    assert "--cap-drop=ALL" in command
    assert any(arg.startswith("--memory=") for arg in command)
    assert f"{snapshot.root}:/src:ro" in command
```

Also test timeout termination, output clipping with full logs on disk, nonzero exit, missing
container runtime, guide-approved network/GPU flags, and no environment pass-through.

- [ ] **Step 2: Run tests and confirm failure**

Run: `cd automation && uv run pytest tests/code_series/test_execution_sandbox.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement an injected backend and explicit unavailable result**

```python
class SandboxBackend(Protocol):
    def run(self, recipe: ExecutionRecipe, snapshot: Snapshot, scratch: Path) -> ExecutionResult:
        raise NotImplementedError
```

Construct argument arrays for the configured container CLI. Mount source at `/src:ro` and
scratch at `/work:rw`; set the working directory to `/work`; copy only declared inputs.
`UnavailableBackend` returns `status="unavailable"` with a limitation and never fabricates an
exit code or performance value.

- [ ] **Step 4: Run sandbox and repository safety tests**

Run: `cd automation && uv run pytest tests/code_series/test_execution_sandbox.py tests/code_series/test_repository.py -q`

Expected: PASS without starting a real container.

- [ ] **Step 5: Commit**

```bash
git add automation/tasks/code_series/execution.py automation/tests/code_series/test_execution_sandbox.py
git commit -m "feat: run guide-approved checks in a sandbox"
```

### Task 3: Benchmark and runtime-claim validation

**Files:**
- Extend: `automation/tasks/code_series/evidence.py`
- Create: `automation/tests/code_series/test_runtime_evidence.py`
- Create: `automation/tasks/code_series/defaults/prompts/runtime-evidence.md`

**Interfaces:**
- Produces: `validate_runtime_claim(claim, executions)`, `render_execution_limitations(results)`, evidence type `run`

- [ ] **Step 1: Write tests that prevent benchmark laundering**

```python
def test_upstream_number_cannot_be_labeled_as_measured(upstream_claim):
    upstream_claim["wording"] = "직접 측정한 처리량은 120 tokens/s다."
    assert validate_runtime_claim(upstream_claim, [])

def test_local_measurement_must_reference_successful_execution(local_claim, execution):
    local_claim["evidence"][0]["execution_id"] = "missing"
    assert "missing" in " ".join(validate_runtime_claim(local_claim, [execution]))
```

- [ ] **Step 2: Verify failures**

Run: `cd automation && uv run pytest tests/code_series/test_runtime_evidence.py -q`

Expected: FAIL.

- [ ] **Step 3: Add source labels and limitation rendering**

```python
MEASUREMENT_SOURCES = {"local-measurement", "upstream-benchmark", "not-measured"}
```

Require local measurements to reference successful execution provenance. Upstream values
must link to a pinned upstream file or release and use wording that attributes the result.
Unavailable results generate a standard limitation paragraph for the chapter brief.

- [ ] **Step 4: Run evidence and runtime suites**

Run: `cd automation && uv run pytest tests/code_series/test_runtime_evidence.py tests/code_series/test_evidence.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add automation/tasks/code_series/evidence.py automation/tasks/code_series/defaults/prompts/runtime-evidence.md automation/tests/code_series/test_runtime_evidence.py
git commit -m "feat: distinguish measured and upstream performance"
```

### Task 4: Licensed images and evidence-linked diagrams

**Files:**
- Create: `automation/tasks/code_series/assets.py`
- Create: `automation/tests/code_series/test_assets.py`
- Extend: `automation/tasks/code_series/article.py`
- Extend: `automation/tasks/code_series/schemas/series.schema.json`

**Interfaces:**
- Produces: `AssetCandidate`, `AssetDecision`, `decide_asset(candidate, project_license)`, `validate_visual_contract(article, series, claims)`

- [ ] **Step 1: Write license and diagram-contract tests**

```python
def test_unknown_image_license_is_rejected():
    decision = decide_asset(AssetCandidate(path="docs/x.png", license=None), "Apache-2.0")
    assert not decision.allowed
    assert decision.reason == "asset-license-unknown"

def test_mermaid_must_name_supporting_claims(article, contract, claims):
    article = article.replace("supports: [request-001]", "supports: []")
    assert validate_visual_contract(article, contract, claims)
```

- [ ] **Step 2: Verify failures**

Run: `cd automation && uv run pytest tests/code_series/test_assets.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement deny-by-default upstream assets and claim-linked visuals**

```python
ALLOWED_ASSET_LICENSES = {"CC0-1.0", "CC-BY-4.0", "Apache-2.0", "MIT"}
```

An allowed copied asset records source path, pinned permalink, SPDX identifier, attribution,
content hash, and any conversion. A repository license does not automatically license every
docs image; an image without an applicable license is rejected. Mermaid/table/block diagrams
must list the claim IDs they explain.

- [ ] **Step 4: Run asset, article, and lint tests**

Run: `cd automation && uv run pytest tests/code_series/test_assets.py tests/code_series/test_article.py tests/papers/test_lint.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add automation/tasks/code_series/assets.py automation/tasks/code_series/article.py automation/tasks/code_series/schemas/series.schema.json automation/tests/code_series/test_assets.py
git commit -m "feat: validate code series visual assets"
```

### Task 5: Queue integration and safe execution end-to-end

**Files:**
- Extend: `automation/tasks/code_series/queue.py`
- Extend: `automation/tasks/code_series/runner.py`
- Extend: `automation/tasks/code_series/review.py`
- Create: `automation/tests/code_series/test_execution_e2e.py`
- Modify: `automation/tasks/code_series/README.md`

**Interfaces:**
- Consumes: execution, evidence, asset, article, and queue modules
- Produces: task kind `execute`, runtime/asset evidence in chapter briefs, documented sandbox recovery

- [ ] **Step 1: Write end-to-end unavailable and successful paths**

```python
def test_unavailable_gpu_check_becomes_a_limitation(series, unavailable_backend):
    outcome = run_execution_task(series.execution_task, series.context, unavailable_backend)
    assert outcome.status == "unavailable"
    assert "GPU" in series.chapter_brief(outcome).read_text()

def test_successful_run_provenance_reaches_claim(series, fake_sandbox):
    outcome = run_execution_task(series.execution_task, series.context, fake_sandbox)
    claim = series.runtime_claim(outcome)
    assert claim["evidence"][0]["execution_id"] == outcome.execution_id
```

- [ ] **Step 2: Run integration tests and confirm boundary failures**

Run: `cd automation && uv run pytest tests/code_series/test_execution_e2e.py -q`

Expected before wiring: FAIL where the queue lacks `execute` dispatch.

- [ ] **Step 3: Add `execute` between trace and evidence only when guide requests it**

```python
class TaskKind(str, Enum):
    EXECUTE = "execute"
```

Projects without execution recipes receive no execute task. Execution failure does not block
static evidence unless the guide marks the run as a required claim gate.

- [ ] **Step 4: Run the complete unpublished-series suite**

Run: `cd automation && uv run pytest tests/code_series tests/test_agent_runner.py tests/papers/test_lint.py -q && uv run ruff check core tasks tests`

Expected: PASS without Docker, GPU, network, or model credentials.

- [ ] **Step 5: Commit**

```bash
git add automation/tasks/code_series automation/tests/code_series automation/tasks/code_series/README.md
git commit -m "feat: integrate safe runtime and visual evidence"
```
