# Embedding on a Kaggle GPU

`kaggle/embed_corpus.py` embeds the chunk files of a build on a free Kaggle
GPU and writes the files `nyaya-build embed` writes, so `nyaya-build load`
reads them unchanged. Use it when a run is too large for the local GPU, or
when the local GPU is busy (design file 28).

For each model, the output folder `<model-dir>` (the model name with `/`
replaced by `__`, e.g. `Qwen__Qwen3-Embedding-0.6B`) holds:

| File | What it is |
| --- | --- |
| `vectors/<year>.npy` | float32 (n, dim), normalised, rows in chunks-file order |
| `vectors/<year>.ids.txt` | the n chunk IDs, one per line |
| `vectors/<year>.sha1.txt` | SHA-1 of each embedded text (lets `nyaya-build embed` reuse rows) |
| `state/embed_<year>.json` | the embed stamp: model, revision, dim, n, rows hash |
| `vectors/reference/passages.json`, `vectors.npy` | 5 chunks and their vectors, for `startup_check` |
| `embed_report.json` | GPU, package versions, seconds, chunks/s, peak VRAM |

Documents are embedded exactly as locally: `header`, a blank line, then the
text, with no prefix (`nyaya.models.encoders.document_text`), fp16, normalised.
The script keeps its own copy of that function and of the batching helpers so
that it runs without the `nyaya` package; `tests/test_kaggle.py` checks each
copy against the original.

The models and revisions come from `config/settings.toml`: `[embed]` and
`[embed.alternatives."BAAI/bge-m3"]`.

## 1. Credentials

Put the Kaggle API token in `.env` (it is git-ignored). Either form works:

```
KAGGLE_API_TOKEN=KGAT_...                 # a token from kaggle.com/settings/api
# or the legacy pair
KAGGLE_USERNAME=<user>
KAGGLE_KEY=<32 hex digits, or a KGAT_ token>
```

`kaggle/push.py` reads them from the environment or `--env-file` (default:
the repository's `.env`) and never prints them. Running kernels with internet
access needs a phone-verified Kaggle account.

```
uv sync --extra dev --extra eval --extra kaggle
```

## 2. Upload the chunk files (a private dataset)

```
uv run python kaggle/push.py dataset --years 2025
```

This copies `build/chunks/2025.jsonl` into a temporary folder and creates the
private dataset `<user>/nyaya-search-chunks-2025` (license CC BY 4.0, as the
source data), or uploads a new version if it exists. `--build DIR` reads
another build directory; `--slug` names the dataset.

Check first that the year's parse-chunk stage finished (`build/state/parse_<year>.json`
exists and the chunk count in it matches `wc -l build/chunks/<year>.jsonl`).

## 3. Push and run the kernel (private, T4 GPU)

```
uv run python kaggle/push.py kernel --dataset <user>/nyaya-search-chunks-2025 --years 2025
```

This pushes `embed_corpus.py` as the private script kernel
`<user>/nyaya-search-embed`, with:

* the dataset attached (it appears under `/kaggle/input/`);
* accelerator `NvidiaTeslaT4` (free tier; `--machine` changes it);
* internet on (model weights from Hugging Face; `sentence-transformers` and
  `transformers` pinned to the versions in `uv.lock`, without touching
  Kaggle's torch);
* one run per model: every configured model by default, or `--model NAME`
  (repeatable). `--batch` and `--token-budget` tune the batching
  (default 32 chunks, 32k tokens per forward pass); `--timeout` caps the
  session in seconds.

Pushing again with the same slug makes and runs a new version.

**By hand in the Kaggle UI instead:** create a notebook, Settings >
Accelerator > GPU T4 x1, Internet on, Add Input > your dataset, upload
`embed_corpus.py`, and run

```
!pip install -q sentence-transformers==6.3.0 transformers==5.22.2
!python embed_corpus.py --model BAAI/bge-m3 --revision 5617a9f61b028005a4858fdac845db406aefb181 --dim 1024 --years 2025
```

Outputs go to `/kaggle/working/<model-dir>/`.

## 4. Wait for it

```
uv run python kaggle/push.py status --kernel <user>/nyaya-search-embed --wait
```

States: `queued`, `running`, then `complete` or `error` (exit code 1).

## 5. Download the output

```
uv run python kaggle/push.py download --kernel <user>/nyaya-search-embed --out kaggle_out
```

`kaggle_out/` is git-ignored. It holds one `<model-dir>` per model and the
kernel log. Read `embed_report.json` for time, chunks/s and VRAM.

## 6. Use the vectors

**Load an index version.** Copy the model's files into the build directory
the chunks came from, then load. The embed stamp matches, so
`nyaya-build embed` treats the year as done; if `embed.model` in config is a
different model, the stamp differs and the year is embedded again locally.

```
cp -r kaggle_out/Qwen__Qwen3-Embedding-0.6B/vectors kaggle_out/Qwen__Qwen3-Embedding-0.6B/state build/
uv run nyaya-build load --version <name> --years 2025
```

To load the bge-m3 vectors, first set `[embed]` to the bge-m3 values from
`[embed.alternatives."BAAI/bge-m3"]` (the loader records `embed.model`, the API
embeds queries with it, and `startup_check` compares the reference vectors
against it).

**Compare models without loading** (`scripts/select_models.py`, in memory,
local GPU for queries and the reranker):

```
uv run python scripts/select_models.py --year 2025 \
    --vectors Qwen/Qwen3-Embedding-0.6B=kaggle_out/Qwen__Qwen3-Embedding-0.6B \
    --vectors BAAI/bge-m3=kaggle_out/BAAI__bge-m3
```

Before scoring, each model's reference vectors are checked against the local
embedder (cosine at least `encoder_check.min_cosine`).
