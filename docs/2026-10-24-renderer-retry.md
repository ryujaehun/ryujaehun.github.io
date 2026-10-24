# Renderer startup resilience and unsupported-command validation

## Observed production failure

[Pages run 40086998884](https://github.com/ryujaehun/ryujaehun.github.io/actions/runs/40086998884) failed on attempt 1 while importing the Mermaid CDN module (`TypeError: Failed to fetch dynamically imported module`). Attempt 2 of the identical f63a8c5 commit passed. The failure occurred before diagram validation, so manually rerunning the entire build was unnecessary work.

## Bounded retry

Both runtime checkers now distinguish known CDN fetch failures during initialization with exit status 75. `scripts/renderer-retry.mjs` reruns only that checker, at most three times with 1s/2s waits. Each attempt creates a fresh Chrome process/profile, avoiding the document module-map cache of failed imports.

Only Mermaid/KaTeX CDN fetch messages during loading qualify. Invalid diagrams/math, code-copy mismatches, browser startup errors, CDP timeouts, configuration errors and other failures remain non-retryable. Persistent CDN failures still fail the deployment. Retries and the underlying errors remain visible in logs.

## Additional gate bug found by negative testing

A real Chrome negative fixture using `\undefinedCommand` unexpectedly passed the previous math checker. KaTeX with `throwOnError:false` renders unsupported commands as red text without a `.katex-error` node; see the [official options documentation](https://katex.org/docs/options.html). The checker now validates with `throwOnError:true` and records auto-render error callbacks. Actual website rendering settings remain unchanged; other page options still come from the build.

The stricter gate identified four unsupported-command occurrences across three legacy posts:

- Sharing and Throughput-Oriented Token Batching: malformed `N` subscripts and an undefined `\tokens` command. Restored the processed/original token labels.
- Inference-Time Hyper-Scaling (KO): `\[` used inside an existing math expression instead of a literal vector bracket.
- SageAttention3 (KO): the same bracket misuse in two vectors, with missing closing brackets. Restored both vectors without changing their values.

These are syntax corrections, not a source-level reassessment of those articles' experimental claims.

## Verification

- Policy tests failed in the no-retry implementation; the actual unsupported-command fixture also failed before the stricter gate was enabled.
- Eight Node tests passed, including real Chrome/KaTeX negative validation and existing reading-path tests.
- Controlled CDP Fetch interception exercised both actual checkers with HTTP 503 responses. For each renderer: healthy succeeds on attempt 1; temporary failure succeeds on attempt 2; persistent failure exits 75 after attempt 3; invalid content exits 1 on attempt 1. All eight scenarios passed.
- Whole minified site: 162 Mermaid diagrams, 30,460 math nodes and 2,895 copy sources passed. The strict scan no longer finds unsupported commands.
- Article lint passed all 686 posts. All 2,158 HTML files had zero broken local links/images; eight validator unit tests passed.
- The three corrected article URLs passed real-page Chrome checks at 390px and 1440px, including expected token labels/vector annotations, no KaTeX errors, and no horizontal overflow.
