# Music Similarity System — Implementation Outline

## Context

The goal is a system that, given a song, retrieves other songs that *sound* similar —
similar genre, mood, instrumentation, and production style — using the **audio itself**,
not artist/genre metadata. The approach is well-established: segment each track, embed the
segments with a pretrained music encoder, pool to a song-level vector, and rank by cosine
similarity. The repo is currently empty, so this is a greenfield build.

This outline defines milestone 1: **a working retrieval baseline** on a small slice of
MTG-Jamendo, built so the encoder is swappable and evaluation is quantitative from day one.

### Confirmed decisions (from planning)
- **Environment:** local **Mac (Apple Silicon)** — favors the PyTorch stack; Essentia/TensorFlow is deferred.
- **First encoder:** **CLAP** (`laion/larger_clap_music`) — clean PyTorch/MPS install, 512-dim, and can embed *text* so queries like "mellow jazz guitar" work. Discogs-EffNet is wired in later behind the same interface.
- **Data scale:** **~1–2k tracks** from MTG-Jamendo (laptop-friendly, iterate in minutes–hours).
- **Evaluation:** **tag-overlap metrics** (precision@k, Jaccard, tag-weighted nDCG) **+ spot listening**.

### Key technical facts (validated)
- **CLAP** — input **48 kHz mono**, 512-dim. `transformers`: `ClapModel`/`ClapProcessor`, `model.get_audio_features(**inputs)`. Trained on ~10 s clips → 10 s segments are natural.
- **Discogs-EffNet** (later) — input **16 kHz**, 1280-dim, EfficientNet-B0 via `essentia-tensorflow`; CC BY-NC-SA 4.0 (research OK). TF on Apple Silicon is the install risk → deferred, likely via conda/Docker.
- **MTG-Jamendo** — 55k tracks, 320 kbps MP3, genre/mood/instrument tags. Full audio = **508 GB**; audio ships as ~1k-track tar shards, so grabbing **1–2 low-quality-audio shards** yields the subset cheaply. Finland mirror available.

## Objective / Scope

**In scope (milestone 1):** subset download, preprocessing, CLAP embeddings, mean-pooled
song vectors, cosine top-k retrieval, tag-overlap evaluation, a query CLI.
**Out of scope (later milestones):** Discogs-EffNet, FAISS at scale, segment+song fusion,
custom/fine-tuned models, a UI.

## Proposed Repository Structure

```
audio-similarity-cds/
  README.md
  pyproject.toml            # deps: torch, transformers, torchaudio, soundfile,
                            #       librosa, numpy, pandas, scikit-learn, pyyaml, tqdm
  config.yaml               # paths, sample rates, segment len/overlap, model id, k
  .gitignore                # data/, embeddings/, *.mp3, .venv
  data/                     # gitignored — audio + metadata live here
    raw/                    # downloaded MP3 shards
    metadata/               # MTG-Jamendo tag TSVs
  embeddings/               # gitignored — cached per-track .npz vectors
  src/audio_similarity/
    config.py               # load/validate config.yaml -> dataclass
    data/
      download.py           # fetch N MTG-Jamendo shards + metadata (Finland mirror)
      metadata.py           # parse autotagging TSV -> {track_id: tag sets}
      subset.py             # stratified sample of ~1-2k track_ids by genre
    preprocess/
      loader.py             # decode mp3 -> mono float32, resample to target sr
      segment.py            # fixed windows (default 10 s, 5 s hop); drop short tails
    embed/
      base.py               # Embedder ABC: .target_sr, .embed_segments(batch)->np.ndarray
      clap.py               # ClapEmbedder (laion/larger_clap_music); also embed_text()
      # discogs_effnet.py   # (later) same interface
      runner.py             # iterate tracks -> segment -> batch embed -> cache .npz
    aggregate.py            # mean-pool segments -> song vector; L2-normalize
    index/
      store.py              # save/load the (track_id, vector) matrix (.npz / parquet)
      search.py             # cosine top-k (normalized matmul); FAISS optional later
    evaluate/
      metrics.py            # precision@k, Jaccard, tag-weighted nDCG, random floor
      run_eval.py           # leave-one-out query over subset -> metrics report
    cli.py                  # subcommands: download | embed | index | query | evaluate
  scripts/                  # thin wrappers around cli.py for each stage
  notebooks/                # exploratory listening / sanity checks
  tests/                    # unit tests for segmenting, pooling, metrics
```

## Pipeline Design (data flow)

```
MTG-Jamendo shards ─▶ subset select ─▶ load+resample(48k mono) ─▶ segment(10s/5s hop)
   ─▶ CLAP embed per segment ─▶ mean-pool + L2-norm ─▶ song matrix
   ─▶ cosine top-k retrieval ─▶ tag-overlap evaluation / query CLI
```

**Stage details & defaults**
1. **Acquire** — `data/download.py` pulls 1–2 low-quality-audio shards (~1–2k tracks) + the
   autotagging TSV; `subset.py` genre-stratifies the sample so evaluation isn't dominated by one genre.
2. **Preprocess** — `loader.py` decodes to mono float32 and resamples to the encoder's
   `target_sr` (48 kHz for CLAP). `segment.py` makes 10 s windows with 5 s hop; config-driven.
3. **Embed** — `clap.py` behind `Embedder` ABC; `runner.py` batches segments (MPS/CPU),
   writes one `.npz` per track so re-runs are incremental. This is the compute-heavy step.
4. **Aggregate** — `aggregate.py` mean-pools segment vectors → song vector, then L2-normalizes
   (so cosine == dot product). Keep segment vectors on disk for the future fusion experiment.
5. **Index + retrieve** — `store.py` stacks song vectors into one matrix; `search.py` does
   normalized matmul for top-k. Plain NumPy is fine at this scale; FAISS is a later swap.
6. **Query** — `cli.py query` takes a `track_id` (or, via CLAP `embed_text`, a text prompt)
   and prints ranked neighbors with their tags for eyeballing.

## Evaluation Design (the crux)

No ground-truth "these two sound alike" labels exist, so **shared tags are the proxy**.
For each query `q` (leave-one-out over the subset), retrieve top-k and score:
- **Precision@k** — fraction of top-k sharing ≥1 tag with `q` (reported per category: genre / mood / instrument).
- **Jaccard overlap** — mean `|T_q ∩ T_r| / |T_q ∪ T_r|` across retrieved `r`.
- **Tag-weighted nDCG@k** — relevance = tag Jaccard, rewards ranking closer matches higher.
- **Random floor** — same metrics under random retrieval, as the baseline to beat.
- **Separation sanity check** — mean intra-genre vs inter-genre cosine similarity.
- **Spot listening** — a notebook that plays a query + its top-5 for a handful of tracks.

`run_eval.py` emits a small JSON/markdown report so encoder/segment/pooling variants are comparable.

## Milestones

- **M1 — Baseline (this build):** download → preprocess → CLAP embed → mean-pool → cosine
  top-k → tag-overlap eval + listening notebook. **Done when** precision@k clears the random
  floor and top-5 neighbors are audibly sensible.
- **M2 — Second encoder:** add `discogs_effnet.py` behind the same interface (conda/Docker TF
  on Apple Silicon); compare EffNet vs CLAP on the M1 metrics.
- **M3 — Experiments:** segment length & hop, pooling (mean vs max vs attention), similarity
  metric, and the **segment+song fusion** idea (match best segment-pairs, not just song means).
- **M4 — Scale:** more tracks + FAISS, and optional late-fusion of CLAP+EffNet vectors.

## Risks & Mitigations
- **Essentia/TF on Apple Silicon** — deferred to M2; isolate in conda/Docker so it never blocks M1.
- **Dataset size** — pull low-quality-audio shards only; `data/` and `embeddings/` are gitignored.
- **Weak evaluation signal** — tags are coarse (mood especially); lean on genre precision + listening, report per-category.
- **MPS quirks** — provide a CPU fallback flag in `runner.py`; keep batch sizes modest.
- **CLAP license checkpoint** — `larger_clap_music` is research-friendly; note any commercial limits in README (EffNet is non-commercial).

## Verification (end-to-end)
1. `pip install -e .` succeeds on the Mac; `import torch, transformers, torchaudio` clean.
2. `cli.py download` fetches a shard + TSV; subset lists ~1–2k stratified track_ids.
3. `cli.py embed` writes one `.npz` per track; spot-check vector shape (512) and finite values.
4. `pytest tests/` passes for segmenting, pooling (L2-norm → unit length), and metrics.
5. `cli.py query <track_id>` returns plausible, tag-sharing neighbors; listening notebook confirms audibly.
6. `cli.py evaluate` reports precision@k / Jaccard / nDCG **above the random floor**, with per-category breakdown.

## Longer-term direction
Whether combining **segment-level and song-level** representations (e.g., best-matching-segment
scoring layered on song-mean cosine) beats whole-song embeddings alone — set up by keeping
segment vectors on disk from M1 onward.
