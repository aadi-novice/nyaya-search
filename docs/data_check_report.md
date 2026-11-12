# Data check report

Generated 2026-11-11 23:28 by `scripts/data_check.py` (seed 42). Covers all metadata and the 77 English year tars on disk at run time.

## 1. Canonical judgment ID

Metadata rows: 43578. Distinct `path` values: 38397.

| Field | Blank | Unique (non-blank) | Rows sharing a value |
| --- | --- | --- | --- |
| `path` | 0 | 38397 | 10362 |
| `cnr` | 0 | 38397 | 10362 |
| `nc_display` | 162 | 38139 | 10529 |
| `case_id` | 162 | 38139 | 10529 |
| `citation` | 0 | 38397 | 10362 |

- Same `path` in more than one year folder: 0
- `path` values with conflicting CNRs: 0; with conflicting `nc_display`: 0
- Rows whose year folder differs from the decision-date year: 10358
- Rows with non-English versions available: 42374

## 2. Tar access

English tars scanned: 77; errors: 0. Total members: 43572. Header scan time, all tars: 24.9 s.
Mismatches with index.json: 0 files. `S_`-prefixed members: 7901.
Metadata rows in downloaded years: 43578; matched to `<path>_EN.pdf`: 43571 (100.0%).

`index.json` lists file names only, not byte offsets. One header scan per tar gives the offsets; each PDF is then read with a single seek, without unpacking.

## 3. Text layer vs scanned pages

Sample: 200 PDFs (seeded), 0 unreadable. Pages: 2744.
Pages with under 200 characters of text: 5 (0.2%). Judgments with at least one such page: 2 (1.0%).
Extra byte-identical copies within the sample: 0.

## 4. Size estimate

Mean pages per judgment: 13.7. Words: mean 5348, median 4265.
Mean chunks per judgment (paragraph-aware, ~400 words): 19.0.
Estimated corpus: about 526,807 pages and 730,887 chunks for 38,397 judgments.

## Caveats

- The sample covers only years downloaded at run time; recent years may differ (longer judgments).
- A page under 200 characters may be a short last page, not a scan. Treat the low-text share as an upper bound on OCR need.
- Chunk counts use a simple paragraph split, not the final chunker.

## 5. Follow-up checks (run by hand, same day)

- **Duplicates.** 5,181 judgments appear twice in the metadata, in two adjacent year folders (for example `1951_1_1_51` under both `year=1950` and `year=1951`). The two rows come from scrapes about four months apart (June and October 2025). In 5,180 cases the PDF also sits in both tars. All 300 duplicate pairs checked were byte-identical. Keeping one copy per `path` loses nothing.
- **Year folder.** The `year` column matches the folder in 88.1% of rows. Do not use either as the decision year; parse `decision_date`.
- **Text quality by decade.** The share of word-like tokens in 15 random judgments per decade (1950s to 2020s) stays between 0.82 and 0.85. The 1950s are no worse than the 2020s. The older PDFs already carry a usable text layer.
- **Unexplained.** 7,901 PDFs have an `S_` file-name prefix. 7 metadata rows have no matching PDF. Neither has been investigated.
