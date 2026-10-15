# Reading design follow-up

Approved scope: remove dock overlap, reduce card metadata, and refine article typography and quotations while preserving the existing palette.

- Home pages omit the fixed dock. An explicit Search button beside the introduction opens the existing localized search dialog.
- Other pages use the theme's scroll dock: hidden initially and when scrolling down, available when scrolling up. Smaller controls and immediate hidden-state visibility prevent lingering content overlap and hidden keyboard focus targets.
- Cards show the first two existing tags; full taxonomy information remains on article pages. Dates and reading times retain their existing layout and translations.
- Article content has a default 46rem reading width, 17px desktop / 16px mobile body text, quieter headings, and upright quotations without decorative quotation marks or hover motion. Korean titles prefer breaks between words.
- The page-width control keeps its existing meaning: the text retains a 5rem inset on each side of the configured page width. Slider steps resize text by 1rem and reset returns to the default; mobile remains constrained to the viewport.
- HLA-WM's author/institution metadata is a small source paragraph in both languages, rather than a quotation.
- Site overrides preserve theme files, source URLs, series navigation, alerts, and existing theme controls.

Validation: production Hugo 0.165.0 full build; existing paper lint on both edited posts; 28 Chrome viewport/page checks covering Korean and English home pages, five recent reviews, and an existing code-series episode. Checks include home Search/Escape, dock scroll direction, initial hidden state, card tag counts, default text width, slider step/reset, math rendering, HTML integrity, and keyboard appendix expansion.
