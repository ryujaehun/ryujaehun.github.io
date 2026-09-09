# Code Series Automation Implementation Roadmap

**Spec:** `docs/superpowers/specs/2026-09-09-code-series-automation-design.md`

The approved design contains five independently reviewable subsystems. Implement them in
this order; do not start a later plan until the preceding plan's full verification command
passes.

1. `2026-09-09-code-series-01-guide-foundation.md` — repository snapshot, inventory,
   compatibility report, guide registration, and the shared Claude/Codex skill.
2. `2026-09-09-code-series-02-evidence-authoring.md` — opencode task runner, evidence,
   chapter briefs, Korean/English drafts, and series review.
3. `2026-09-09-code-series-02b-execution-assets.md` — sandboxed execution provenance,
   benchmark boundaries, licensed images, and evidence-linked diagrams.
4. `2026-09-09-code-series-03-publishing-notifications.md` — daily publishing, live-page
   verification, Telegram, papers integration, workflows, and timers.
5. `2026-09-09-code-series-04-pilot-hardening.md` — tiny-vLLM, Mini-SGLang, and Shardy
   pilots, golden fixtures, compatibility tuning, and final documentation.

Each plan leaves a working, tested vertical slice. The first plan never calls opencode or
publishes. The second can create a fully verified unpublished series. The third adds safe runtime and visual evidence. The fourth operates publication and
notification. The fifth demonstrates that the template generalizes.


## Spec coverage

| Approved requirement | Owning plan |
|---|---|
| Natural-language Claude/Codex guide entrypoint | 01 |
| Pinned repository, inventory, compatibility and active user questions | 01 |
| Hierarchical recon, evidence, brief and article work | 02 |
| Korean/English validation and whole-series gate | 02 |
| Sandboxed execution, benchmark provenance and licensed visuals | 02b |
| Daily publication only after complete verification | 03 |
| Live-link Telegram alerts for code series and papers | 03 |
| systemd/GitHub Actions operation and recovery docs | 03 |
| tiny-vLLM, Mini-SGLang and Shardy proof | 04 |
