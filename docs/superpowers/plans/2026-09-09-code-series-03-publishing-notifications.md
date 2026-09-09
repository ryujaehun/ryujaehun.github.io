# Code Series Publishing and Notifications Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish one verified bilingual chapter per day and notify Telegram only after both public links are live, while adding the same reliable notification path to papers.

**Architecture:** Scheduling, Hugo materialization, deployment probing, and notification are separate idempotent stages. Shared Telegram transport lives in `core`; each content task owns its message format and ledger transition. Existing private GitHub Actions plus local systemd dispatch remain the execution topology.

**Tech Stack:** Python 3.10+, httpx, PyYAML, pytest, Hugo, GitHub Actions, systemd user timers, Telegram Bot API

**Spec:** `docs/superpowers/specs/2026-09-09-code-series-automation-design.md`

## Global Constraints

- Complete plans 01, 02, and 02b first.
- Never publish any chapter before the entire bilingual series is verified.
- Publish at most one code-series chapter per configured local date.
- A content commit is not a successful deployment; probe the expected title at the public URL.
- Notification failure never unpublishes content and remains retryable.
- A stable notification ID prevents duplicate Telegram messages across reruns.
- Locally, read only `notify.telegram` from `BLOG_TELEGRAM_CONFIG`; never read booking credentials.
- In GitHub Actions, credentials come only from repository secrets/environment variables.
- Redact Bot tokens from every logging handler.

---

### Task 1: Series schedule and atomic Hugo materialization

**Files:**
- Create: `automation/tasks/code_series/publish.py`
- Create: `automation/tests/code_series/test_publish.py`
- Extend: `automation/tasks/code_series/layout.py`
- Extend: `automation/tasks/code_series/state.py`
- Extend: `automation/tasks/code_series/queue.py`

**Interfaces:**
- Produces: task kind `schedule`, `build_schedule(project_dir, start_date, timezone)`, `due_chapter(project_dir, now)`, `materialize_due(workspace, project_dir, now, publish=False)`

- [ ] **Step 1: Write schedule and guard tests**

```python
def test_schedule_assigns_one_chapter_per_day(verified_series):
    schedule = build_schedule(verified_series.path, date(2026, 9, 10), "Asia/Seoul")
    assert [item.date.isoformat() for item in schedule] == ["2026-09-10", "2026-09-11"]

def test_unverified_series_cannot_materialize(unverified_series, workspace):
    with pytest.raises(PublishError, match="series-verified"):
        materialize_due(workspace, unverified_series.path, NOW, publish=True)
```

Test reruns, existing published file preservation, bilingual filenames, one-per-day, future
dates, slug collisions, and atomic replacement.

- [ ] **Step 2: Run tests and confirm failure**

Run: `cd automation && uv run pytest tests/code_series/test_publish.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement schedule records and Hugo front matter**

```python
@dataclass(frozen=True)
class ScheduleItem:
    chapter_id: str
    order: int
    date: date
    ko_path: str
    en_path: str
```

Write `draft: false` only when `publish=True` and the series is verified. Include series ID,
chapter order/count, repository URL, pinned commit, categories, and stable slug in both
languages. Preserve already published files byte-for-byte on rerun.

- [ ] **Step 4: Run publish and existing rendering tests**

Run: `cd automation && uv run pytest tests/code_series/test_publish.py tests/papers/test_render.py tests/papers/test_collisions.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add automation/tasks/code_series/publish.py automation/tasks/code_series/layout.py automation/tasks/code_series/state.py automation/tests/code_series/test_publish.py
git commit -m "feat: schedule verified code series posts"
```

### Task 2: Shared Telegram transport and safe configuration

**Files:**
- Create: `automation/core/notify.py`
- Create: `automation/tests/test_notify.py`
- Modify: `automation/requirements.txt`

**Interfaces:**
- Produces: `TelegramConfig`, `load_telegram_config(env, config_path=None)`, `install_log_redaction()`, `send_telegram(client, config, message)`, `notify(config, message)`

- [ ] **Step 1: Port behavior tests without importing the macro package**

```python
def test_environment_credentials_win_over_yaml(tmp_path):
    cfg = load_telegram_config(
        {"TELEGRAM_BOT_TOKEN": "12345678:" + "A" * 32, "TELEGRAM_CHAT_ID": "42"},
        tmp_path / "macro.yaml",
    )
    assert cfg.chat_id == "42"

def test_redaction_hides_token_embedded_in_httpx_url(caplog):
    token = "12345678:" + "A" * 32
    install_log_redaction()
    logging.getLogger("httpx").warning("POST https://api.telegram.org/bot%s/sendMessage", token)
    assert token not in caplog.text
```

Also test 3900-character clipping, disabled configuration, YAML reading of only
`notify.telegram`, timeout, HTTP failure, and missing config.

- [ ] **Step 2: Verify failure**

Run: `cd automation && uv run pytest tests/test_notify.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement the small shared adapter**

```python
@dataclass(frozen=True)
class TelegramConfig:
    enabled: bool = False
    token: str = ""
    chat_id: str = ""
```

Add `httpx>=0.27,<1` to runtime requirements. `notify` logs the message and catches transport
errors into a `NotificationResult(sent=False, error=str(exc))`; the caller, not `core`, persists
retry state.

- [ ] **Step 4: Run notification tests and compare macro behavior**

Run: `cd automation && uv run pytest tests/test_notify.py -q && uv run ruff check core/notify.py tests/test_notify.py`

Expected: PASS, including token redaction for direct strings and logging arguments.

- [ ] **Step 5: Commit**

```bash
git add automation/core/notify.py automation/tests/test_notify.py automation/requirements.txt
git commit -m "feat: add shared Telegram notifications"
```

### Task 3: Live deployment probe and notification ledger

**Files:**
- Create: `automation/tasks/code_series/notify.py`
- Create: `automation/tests/code_series/test_notify.py`
- Extend: `automation/tasks/code_series/state.py`

**Interfaces:**
- Produces: `PostNotification`, `probe_page(client, url, expected_title)`, `notification_id(item)`, `notify_due(project_dir, config, client, now)`

- [ ] **Step 1: Write deployment and idempotency tests**

```python
def test_notify_waits_until_both_languages_are_live(published_item, fake_http, telegram):
    fake_http.respond(published_item.ko_url, 200, published_item.title_ko)
    fake_http.respond(published_item.en_url, 404, "")
    result = notify_due(published_item.project_dir, telegram, fake_http, NOW)
    assert result.status == "notify-pending"
    assert telegram.messages == []

def test_sent_notification_is_not_sent_twice(sent_item, fake_http, telegram):
    notify_due(sent_item.project_dir, telegram, fake_http, NOW)
    assert telegram.messages == []
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `cd automation && uv run pytest tests/code_series/test_notify.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement strict live checks and atomic ledger updates**

```python
def notification_id(series: str, chapter: str) -> str:
    return f"code-series:{series}:{chapter}:ko-en"
```

Require HTTP 200, final URL on `jaehun.me`, and the escaped expected title in each HTML body.
Record attempt count, error category, sent timestamp, post commit, URLs, and SHA-256 message
hash. A timeout leaves `notify-pending`; `sent` is terminal.

- [ ] **Step 4: Run notify and state tests**

Run: `cd automation && uv run pytest tests/code_series/test_notify.py tests/code_series/test_state.py tests/test_notify.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add automation/tasks/code_series/notify.py automation/tasks/code_series/state.py automation/tests/code_series/test_notify.py
git commit -m "feat: notify when code series links are live"
```

### Task 4: Papers publication notification

**Files:**
- Create: `automation/tasks/papers/notify.py`
- Extend: `automation/tasks/papers/state.py`
- Modify: `automation/tasks/papers/cli.py`
- Create: `automation/tests/papers/test_notify.py`
- Extend: `automation/tests/papers/test_state.py`

**Interfaces:**
- Produces: `paper_notification_id`, `paper_urls`, `notify_published_papers`, papers CLI command `notify`
- Consumes: `core.notify`, existing paper title/version/draft path state

- [ ] **Step 1: Write paper link and rerun tests**

```python
def test_paper_message_contains_blog_languages_and_arxiv(published_paper):
    message = build_paper_message(published_paper)
    assert published_paper.ko_url in message
    assert published_paper.en_url in message
    assert f"https://arxiv.org/abs/{published_paper.arxiv_id}" in message

def test_paper_state_round_trips_notification_status(tmp_path):
    state = State.load(tmp_path / "state.json")
    state.mark_notified("2609.03430", "paper:2609.03430v1:ko-en", NOW)
    state.save()
    assert State.load(state.path).entries["2609.03430"]["notification"]["status"] == "sent"
```

- [ ] **Step 2: Run tests and verify failure**

Run: `cd automation && uv run pytest tests/papers/test_notify.py tests/papers/test_state.py -q`

Expected: FAIL.

- [ ] **Step 3: Add notification metadata without changing paper selection state**

```python
def cmd_notify(args) -> int:
    state = State.load(args.ws.path(layout.STATE))
    result = notify_published_papers(args.ws, state, args.telegram_config)
    state.save()
    emit(result.as_dict(), args.json)
    return EXIT_OK
```

Only entries with a published Korean draft and a validated English sibling are eligible.
Notification metadata must not make `materialized_today` or `excluded_ids` change behavior.

- [ ] **Step 4: Run all paper state, CLI, and notification tests**

Run: `cd automation && uv run pytest tests/papers/test_notify.py tests/papers/test_state.py tests/papers/test_render.py tests/papers/test_translate.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add automation/tasks/papers/notify.py automation/tasks/papers/state.py automation/tasks/papers/cli.py automation/tests/papers
git commit -m "feat: notify after paper reviews deploy"
```

### Task 5: Worker, publisher, and notifier workflows/timers

**Files:**
- Create: `automation/.github/workflows/code-series.yaml`
- Modify: `automation/.github/workflows/daily.yaml`
- Create: `automation/systemd/blog-code-series-worker.service`
- Create: `automation/systemd/blog-code-series-worker.timer`
- Create: `automation/systemd/blog-code-series-publisher.service`
- Create: `automation/systemd/blog-code-series-publisher.timer`
- Create: `automation/systemd/blog-code-series-notifier.service`
- Create: `automation/systemd/blog-code-series-notifier.timer`
- Create: `automation/tests/code_series/test_operations_config.py`

**Interfaces:**
- Produces: manually dispatchable private workflow jobs `work`, `publish`, `notify`; local timers dispatch those jobs

- [ ] **Step 1: Write static workflow and unit tests**

```python
def test_workflow_never_exposes_telegram_secret(workflow_text):
    assert "TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}" in workflow_text
    assert "echo $TELEGRAM" not in workflow_text

def test_publisher_runs_at_most_one_due_chapter(workflow_text):
    assert "publish --due --limit 1" in workflow_text
```

Parse YAML after replacing GitHub expression markers, and verify `Persistent=true`, explicit
working directories, `--skip-if-done`, concurrency groups, and notifier `continue-on-error`.

- [ ] **Step 2: Run tests and verify failure**

Run: `cd automation && uv run pytest tests/code_series/test_operations_config.py -q`

Expected: FAIL.

- [ ] **Step 3: Add workflow jobs and user services**

```yaml
concurrency:
  group: code-series-${{ github.ref }}
  cancel-in-progress: false
```

The code-series workflow checks out automation and the blog workspace using the existing
deploy-key pattern, installs requirements/opencode, executes exactly one requested mode,
commits only declared data/content paths, and pushes. Add the papers `notify` command after
the existing publish push and public-page probe window.

- [ ] **Step 4: Validate YAML, shell, tests, and workflow diffs**

Run: `cd automation && uv run pytest tests/code_series/test_operations_config.py -q && systemd-analyze --user verify systemd/blog-code-series-worker.service systemd/blog-code-series-worker.timer systemd/blog-code-series-publisher.service systemd/blog-code-series-publisher.timer systemd/blog-code-series-notifier.service systemd/blog-code-series-notifier.timer`

Expected: pytest PASS and `systemd-analyze` exits 0 without unknown directive, dependency,
calendar, or executable-path errors.

- [ ] **Step 5: Commit**

```bash
git add automation/.github/workflows automation/systemd automation/tests/code_series/test_operations_config.py
git commit -m "ops: schedule code series publication and alerts"
```

### Task 6: Publication-to-notification integration and operator docs

**Files:**
- Create: `automation/tests/code_series/test_publication_e2e.py`
- Modify: `automation/tasks/code_series/README.md`
- Modify: `automation/tasks/papers/README.md`
- Modify: `automation/README.md`

**Interfaces:**
- Consumes: verified series, publisher, fake live site, fake Telegram, papers notifier
- Produces: documented recovery procedure and network-free end-to-end proof

- [ ] **Step 1: Write the final operations integration test**

```python
def test_verified_series_publishes_daily_then_notifies_once(operations_fixture):
    first = operations_fixture.publish(day=0)
    assert first.chapter_order == 1
    assert operations_fixture.notify_before_deploy().sent is False
    operations_fixture.deploy(first)
    assert operations_fixture.notify_after_deploy().sent is True
    assert operations_fixture.notify_after_deploy().skipped_duplicate is True
```

- [ ] **Step 2: Run the integration test and fix only composition failures**

Run: `cd automation && uv run pytest tests/code_series/test_publication_e2e.py -q`

Expected before wiring fixes: FAIL only at boundaries already covered by unit tests.

- [ ] **Step 3: Document exact local and Actions configuration**

```text
BLOG_TELEGRAM_CONFIG=/home/jaehun/workspace/macro/config.yaml
TELEGRAM_BOT_TOKEN=<GitHub Actions secret name; never put the value in this file>
TELEGRAM_CHAT_ID=<GitHub Actions secret name; never put the value in this file>
```

Document timer installation, `status`, safe manual `notify`, duplicate handling, live-probe
failures, secret rotation, and how papers and code-series messages differ.

- [ ] **Step 4: Run the complete automation suite**

Run: `cd automation && uv run pytest -q && uv run ruff check core tasks tests`

Expected: PASS without live Telegram, GitHub, opencode, or public network access.

- [ ] **Step 5: Commit**

```bash
git add automation/tests/code_series/test_publication_e2e.py automation/tasks/code_series/README.md automation/tasks/papers/README.md automation/README.md
git commit -m "docs: explain code series publishing operations"
```
