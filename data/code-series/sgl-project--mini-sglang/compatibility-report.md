# Compatibility report

## Status

`needs-review` for commit `9a91cfafe754aa85daee49998176275667eb58f2`.

## Analyzable scope

114 files / 383025 bytes; core languages: c++, cuda, python.

## Omissions and risks

6 omitted files / 521812 bytes are outside analysis.
- `.gitignore` — no deterministic extractor for this file
- `.pre-commit-config.yaml` — no deterministic extractor for this file
- `Dockerfile` — unclassified file without a deterministic extractor
- `LICENSE` — no deterministic extractor for this file
- `assets/logo.png` — no deterministic extractor for this file
- 1 additional omitted files are recorded in the inventory.

### `execution-environment`

- Observed: CUDA core source files
- Threshold: a documented CPU-only validation path
- Consequence: execution validation needs a declared hardware and sandbox profile
- Recommended guide question: Which GPU-free fixture or approved GPU profile validates the representative flow?

### `monorepo-roots`

- Observed: 3 top-level source roots: benchmark, python, tests
- Threshold: at most 2 top-level source roots
- Consequence: a single guide could mix independent products and dilute its series narrative
- Recommended guide question: Which source root is the intended product for this series?

### `unclassified-unanalyzable-file`

- Observed: 1 unclassified files: Dockerfile
- Threshold: 0 unclassified non-core files without deterministic classification
- Consequence: the guide may omit source, build, or project-boundary evidence
- Recommended guide question: Which unclassified files are required to explain the project?

### `unsupported-build-system`

- Observed: pyproject.toml
- Threshold: build files must have a deterministic dependency extractor
- Consequence: the guide may miss build targets and execution boundaries
- Recommended guide question: Should this build system receive an extractor or be scoped out of the guide?

## Extension proposal

- `execution-environment`: Add an approved hardware or CPU-fixture validation profile.
- `monorepo-roots`: Create a project profile for the selected source root.
- `unclassified-unanalyzable-file`: Classify the affected files with a deterministic extractor or explicit profile.
- `unsupported-build-system`: Add a deterministic build-manifest extractor before expanding the template.

## Expected cost and series-size impact

- `execution-environment`: Environment setup adds validation work but need not add chapters.
- `monorepo-roots`: Choosing one product narrows the first series; covering all roots adds seasons.
- `unclassified-unanalyzable-file`: A classification pass adds discovery work and may add a component-focused chapter.
- `unsupported-build-system`: Build extraction adds discovery work and may add a build-boundary chapter.
