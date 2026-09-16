# Compatibility report

## Status

`needs-review` for commit `e25bf1994efa90bc98b721ba7c527402f86fbeaf`.

## Analyzable scope

12 files / 1154497 bytes; core languages: c++, cuda, python.

## Omissions and risks

19 omitted files / 71132 bytes are outside analysis.
- `.gitattributes` — no deterministic extractor for this file
- `.gitignore` — no deterministic extractor for this file
- `.vscode/c_cpp_properties.json` — no deterministic extractor for this file
- `.vscode/launch.json` — no deterministic extractor for this file
- `.vscode/settings.json` — no deterministic extractor for this file
- 14 additional omitted files are recorded in the inventory.

### `dominant-source-file`

- Observed: include/json.hpp is 92% of analyzable source bytes
- Threshold: under 60% of analyzable source bytes in any one file
- Consequence: size and chapter estimates describe that file rather than the product
- Recommended guide question: Is this file a vendored dependency to exclude, or genuinely in scope?

### `execution-environment`

- Observed: CUDA core source files
- Threshold: a documented CPU-only validation path
- Consequence: execution validation needs a declared hardware and sandbox profile
- Recommended guide question: Which GPU-free fixture or approved GPU profile validates the representative flow?

### `monorepo-roots`

- Observed: 4 top-level source roots: include, python, src, tests
- Threshold: at most 2 top-level source roots
- Consequence: a single guide could mix independent products and dilute its series narrative
- Recommended guide question: Which source root is the intended product for this series?

### `unclassified-core-file`

- Observed: 1 core files without a language: full_test.sh
- Threshold: 0 unclassified core files
- Consequence: the guide cannot prove that its analyzable scope covers the core implementation
- Recommended guide question: Which unclassified files are essential to the project design?

### `unclassified-unanalyzable-file`

- Observed: 6 unclassified files: build.sh, check.sh, ncu.sh, nsys.sh, run.sh, test.sh
- Threshold: 0 unclassified non-core files without deterministic classification
- Consequence: the guide may omit source, build, or project-boundary evidence
- Recommended guide question: Which unclassified files are required to explain the project?

### `unsupported-build-system`

- Observed: CMakeLists.txt
- Threshold: build files must have a deterministic dependency extractor
- Consequence: the guide may miss build targets and execution boundaries
- Recommended guide question: Should this build system receive an extractor or be scoped out of the guide?

## Extension proposal

- `dominant-source-file`: Exclude the file as a vendored dependency, or declare it in scope with an explicit profile.
- `execution-environment`: Add an approved hardware or CPU-fixture validation profile.
- `monorepo-roots`: Create a project profile for the selected source root.
- `unclassified-core-file`: Classify the core file with a deterministic extractor or explicit profile.
- `unclassified-unanalyzable-file`: Classify the affected files with a deterministic extractor or explicit profile.
- `unsupported-build-system`: Add a deterministic build-manifest extractor before expanding the template.

## Expected cost and series-size impact

- `dominant-source-file`: Scope is unestimable while one file carries the budget; excluding it usually shrinks the series.
- `execution-environment`: Environment setup adds validation work but need not add chapters.
- `monorepo-roots`: Choosing one product narrows the first series; covering all roots adds seasons.
- `unclassified-core-file`: Classification work is required before series scope can be estimated.
- `unclassified-unanalyzable-file`: A classification pass adds discovery work and may add a component-focused chapter.
- `unsupported-build-system`: Build extraction adds discovery work and may add a build-boundary chapter.
