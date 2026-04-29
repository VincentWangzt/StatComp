# Reference PDF Conversion Check

The three reference papers were downloaded from arXiv and converted to Markdown-style text files with page markers.  Conversion used `pdftotext` flow extraction rather than strict layout extraction, because the KSIVI PDF's two-column layout produced broken vertical text under `pdftotext -layout`.

## Files

- `2506.05088-kpg-sivi.pdf` -> `2506.05088-kpg-sivi.md`
- `2506.03839v1-revisiting-uivi-aisivi.pdf` -> `2506.03839v1-revisiting-uivi-aisivi.md`
- `2405.18997-ksivi.pdf` -> `2405.18997-ksivi.md`

## Page Count Check

- `2506.05088-kpg-sivi.pdf`: 16 PDF pages, 16 Markdown page markers.
- `2506.03839v1-revisiting-uivi-aisivi.pdf`: 13 PDF pages, 13 Markdown page markers.
- `2405.18997-ksivi.pdf`: 22 PDF pages, 22 Markdown page markers.

## Visual Spot Check

Rendered PDF pages were compared against the corresponding Markdown page sections:

- KPG-SIVI: pages 7 and 14.
- AISIVI/UIVI revisiting: pages 6 and 13.
- KSIVI: pages 6, 18, and 19.

The section order, page markers, main prose, captions, and table text are consistent with the rendered PDFs on these checked pages.  The Markdown files are suitable for careful reading, search, and experiment-section style comparison.

## Known Limits

- Figures are not embedded in the Markdown files; captions and nearby prose are extracted.
- Complex equations and tables are text-linearized, so the PDF remains the source of truth for exact mathematical layout.
- The files are UTF-8.  Some terminals may display math symbols or accented names incorrectly depending on console encoding, but the Markdown files themselves are encoded as UTF-8.
