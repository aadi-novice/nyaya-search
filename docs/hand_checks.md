# The two hand checks

Two gates need a person rather than a script: the **G6 hand check** (is the
parsed text free of page furniture?) and **judge validation** (does the G2
judge agree with you?). This guide walks through both. Run every command from
the repository root.

| Check | Command | Your time | Passes when |
|---|---|---|---|
| G6 hand check | `uv run nyaya-eval g6-holdout` | about 2.5–4 hours (50 PDFs at 3–5 minutes each), plus a few minutes to generate | zero margin letters inside sentences and zero running heads, over all 50 PDFs |
| Judge validation | `uv run nyaya-eval judge-validate` | about 1–1.5 hours (40 pairs at 1–2 minutes each), plus a few minutes for the judge | Cohen's kappa ≥ 0.6 on at least 30 scored pairs |

You can split either check over several sittings; nothing is lost between them.

---

## 1. G6 hand check

### What it is for

The build already measures G6's automatic metrics (paragraph integrity, size
compliance). What a script cannot fully judge is whether **page furniture**
leaked into the text the index uses:

*   **Margin letters**: the line markers `A` to `H` printed in the margin of
    S.C.R. pages. A leak looks like a stray capital in the middle of a
    sentence: `the appellant was C entitled to bail`.
*   **Running heads**: the header repeated at the top of each page, such as
    "SUPREME COURT REPORTS" or "Digital Supreme Court Reports" with a page
    number or a short case title, appearing as a line inside the text.

### Step 1: generate the sample

```sh
uv run nyaya-eval g6-holdout
```

It needs the manifest and the corpus tars, and the Era C bands from
`nyaya-build calibrate` (it stops with a setup error if those are missing). It
never touches the database, the GPU or an LLM.

It picks `g6.holdout_size` (50) PDFs from 2010–2026 with seed
`g6.holdout_seed` (4242), leaving out the calibration sample, and writes them
to **`eval/runs/g6_holdout/`**:

*   `index.html`: start here; it links one review page per PDF.
*   `<path>/review.html`: each PDF page image (left) beside the clean text of
    that page (right), with automatic hints at the top.
*   `<path>/clean_text.txt`: the full text the build indexes.
*   `handcheck.csv`: one row per PDF, where you record what you find.
*   `sample.json`: the seed, years, excluded calibration paths and any skipped
    PDFs.

Options: `--years 2010-2026`, `--size N`, `--out DIR`, `--dpi N` (page image
resolution, default 110). It exits with code 2 if it found fewer in-scope PDFs
than asked for; check `sample.json` for the skipped ones.

### Step 2: read each PDF

Open `eval/runs/g6_holdout/index.html` in a browser. For each PDF:

1.  Look at the **automatic hints** first. They are only a starting point:
    the margin hint (`a lowercase letter, space, A–H, space, lowercase
    letter`) misses some leaks and flags some real text, such as "Schedule A
    of".
2.  Skim the clean text on the right against the page image on the left.
    Look for capitals `A`–`H` standing alone mid-sentence, and for header lines
    (report name, page number, short title) inside the text.
3.  A capital that the judgment itself uses ("Exhibit A", "Annexure B") is
    not a leak; check the page image if unsure.

### Step 3: record results

Fill three columns of `eval/runs/g6_holdout/handcheck.csv` for each row:

| Column | What to write |
|---|---|
| `margin_letters_found` | the number of margin letters left inside sentences (`0` if none) |
| `running_heads_found` | the number of running-head lines left in the text (`0` if none) |
| `notes` | anything worth keeping: a quoted example, the PDF page, a doubt |

Fill every row, including the zeros, so a blank always means "not checked
yet". Keep the auto-hint columns as they are.

`eval/runs/` is not committed, so once you finish, note the totals and the
date (and any PDFs with leaks) in the G6 results of the design document.

### What passes

G6 passes when **both columns sum to zero** across all 50 PDFs. A single leak
fails the gate; note the PDF and page in `notes`, since the fix usually means
recalibrating the margin bands or adding a running-head pattern
(`running_head_patterns` in `config/settings.toml`), and then rebuilding and
re-running this check.

---

## 2. Judge validation

### What it is for

G2 counts an answer sentence as grounded only if an LLM judge rates it
`supported` against a passage it cites. The judge's scores count only if it
agrees with your own labels at **Cohen's kappa ≥ 0.6** (`judge.min_kappa`).

### Step 1: collect (sentence, passage) pairs

Label **40 pairs** (the README asks for 30 to 50; `judge-validate` refuses to
pass on fewer than `judge.min_hand_labels` = 30 *scored* pairs, so a margin
covers pairs the judge cannot label). Take them from an `answers` run, which
writes `eval/runs/answers/<run_id>/results.jsonl`. If you have none yet:

```sh
uv run nyaya-eval answers --sets dev --limit 20
```

This pulls up to 40 pairs from the latest run into a draft file (the
`<<'EOF'` blocks here are for Git Bash). It reads only
the answer sentences and the passages, **not** the judge's verdicts, so you
label blind:

```sh
uv run python - <<'EOF'
import json, pathlib
runs = sorted(pathlib.Path("eval/runs/answers").glob("*/results.jsonl"),
              key=lambda p: p.stat().st_mtime)
pairs, seen = [], set()
for line in runs[-1].read_text(encoding="utf-8").splitlines():
    rec = json.loads(line) if line.strip() else {}
    ids, texts = rec.get("retrieved", []), rec.get("contexts", [])
    passages = dict(zip(ids, texts)) if len(ids) == len(texts) else {}
    for s in rec.get("answer", []):
        for c in s["citations"][:1]:  # one pair per sentence
            if c["chunk_id"] in passages and (s["sentence"], c["chunk_id"]) not in seen:
                seen.add((s["sentence"], c["chunk_id"]))
                pairs.append({"sentence": s["sentence"], "chunk_id": c["chunk_id"],
                              "passage": passages[c["chunk_id"]]})
out = pathlib.Path("eval/runs/hand_label_draft.jsonl")
with out.open("w", encoding="utf-8") as f:
    for i, p in enumerate(pairs[:40], 1):
        f.write(json.dumps({"id": f"hl-{i:03d}", **p, "label": "TODO",
                            "labeller": "author"}, ensure_ascii=False) + "\n")
print(f"{min(len(pairs), 40)} pairs from {runs[-1]} -> {out}")
EOF
```

Try to include some sentences you expect to be `partial` or `unsupported`;
kappa is unreliable when nearly every label is `supported`.

### Step 2: label each pair

For each line, read the `passage` and the `sentence` and set `label` to one of
the judge's three labels (the same definitions as `prompts/judge_v1.md`):

| Label | Meaning |
|---|---|
| `supported` | everything the sentence states is in, or follows directly from, the passage. Paraphrase and minor wording differences are fine. |
| `partial` | the passage supports part of the sentence, but the sentence adds something, overstates it, or drops a condition that changes its meaning. |
| `unsupported` | the passage does not support the sentence, contradicts it, or is about something else. |

Judge only against the passage, not your knowledge of the law. If the sentence
gives the Court's view but the passage shows only a party's argument, a
dissent, or a quoted case the Court did not adopt, the label is at most
`partial`.

### Step 3: save the labels

Write the labelled pairs to **`eval/datasets/hand_labels_v1.jsonl`**, one
object per line with exactly these keys (drop the `passage` field):

```json
{"id": "hl-001", "sentence": "...", "chunk_id": "2025_1_230_238#judgment#014", "label": "supported", "labeller": "author"}
```

```sh
uv run python - <<'EOF'
import json, pathlib
src = pathlib.Path("eval/runs/hand_label_draft.jsonl")
dst = pathlib.Path("eval/datasets/hand_labels_v1.jsonl")
keep = ("id", "sentence", "chunk_id", "label", "labeller")
rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
todo = [r["id"] for r in rows if r["label"] == "TODO"]
assert not todo, f"still unlabelled: {todo}"
dst.write_text("".join(json.dumps({k: r[k] for k in keep}, ensure_ascii=False) + "\n"
                       for r in rows), encoding="utf-8")
print(f"{len(rows)} labels -> {dst}")
EOF
```

A wrong label value or an extra key fails validation when the file is read.

### Step 4: run the judge

```sh
uv run nyaya-eval judge-validate
```

It needs Postgres with the active index (it reads each passage by `chunk_id`)
and the judge model settings `JUDGE_BASE_URL`, `JUDGE_MODEL` and
`JUDGE_API_KEY` in `.env`. It is paced to `judge.rpm` (20 a minute), so 40
pairs take two to three minutes. Use `--labels PATH` to point at another file.

It prints `run_id`, `file`, `labels`, `scored`, `judge_errors`, `kappa`,
`passed` and `missing_chunks`, and saves the full run (with the agreement
rate, the confusion matrix and each pair's verdict) to
`eval/runs/<run_id>.json` and the `eval_runs` table.

### What passes

**`passed: true`**, with exit code 0: kappa ≥ 0.6 over at least 30 scored
pairs. If it fails:

*   **`missing_chunks` is not empty**: those chunk IDs are not in the active
    index (it was rebuilt since the answers run); relabel from a newer run.
*   **Too few scored pairs**: add labels, or re-run if `judge_errors` came from
    rate limits.
*   **Kappa below 0.6**: read the disagreeing rows in the run file. If your
    labels hold up, the judge prompt or model needs changing (a new
    `judge.prompt` version, not an edit to `judge_v1.md`), and G2 does not
    count until a later run passes.
