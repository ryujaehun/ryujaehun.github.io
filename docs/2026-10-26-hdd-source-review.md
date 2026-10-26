# HDD failure prediction source review — 2026-10-26

Reviewed the pending correction to `content/posts/2024-11-03-paper-HardDiskDrive.md` against Meta’s DSN 2023 PDF, sections III–V, Figure 6 and Table I:
https://research.facebook.com/file/3641873812712342/Hard-Disk-Drive-Failure-Analysis-and-Prediction---An-Industry-View.pdf

- Verified the 53,000+ failed / 54,000 healthy drive sample, A/B/C feature configurations and all twelve precision/recall values.
- Distinguished same-window evaluation from a different-window production population. Model C precision of 1.11% means most warnings are false positives despite recall of 97.76%.
- Distinguished split-count F-score from F1 and causal attribution. Removed unsupported XGBoost selection motives and the claim that HDD health prediction is inherently static.
- Identified the UCB discussion and illustrative 10,000-warning calculation as interpretation. Restored the authors’ conclusion that the model was not ready for production.
- Retained the original route, title and publication date. The correction date records the pending October 25 review.

Validation after incorporating October 26 automated posts: publication lint passed for 690 posts; Hugo 0.168.0 production build passed without path warnings; 2,174 HTML files had no broken local links/images. Rendering passed for 30,695 formulas, 2,925 code sources and 172 Mermaid diagrams. Existing browser checks temporarily extended to the HDD route passed at 390/1440 px (26 screens). Eight link-checker tests and twelve Node tests passed.

The user explicitly authorized committing, pushing and production deployment on October 26. The Pages workflow with pinned Hugo 0.165.0 and the live article are verified after push.
