# ML design: Track A anomaly hunt

This is the playbook for training and scoring. Agents must follow it, or record a reasoned change here first.

## 0. Principles
- **The deliverable is a vetted list of strange stars, not a model.** Every ML choice is judged by recall on injected and known oddballs and by the false-positive rate. Reconstruction loss alone doesn't count.
- **Classical baselines always run alongside.** If ML doesn't beat them, we say so.
- **Instrument effects are the enemy.** Most TESS "anomalies" are spacecraft systematics. Design every stage to separate star behaviour from instrument behaviour.
- **No peeking.** Known positives and injection sets are evaluation-only. Thresholds and hyperparameters are tuned on validation injections, never on known positives.

## 1. Inputs
- One example is **one star in one sector**: 30-min bins, about 1,300 bins, padded to a fixed length with a **validity mask**. Never interpolate across gaps.
- **Channels:**
  1. PDCSAP normalised flux
  2. SAP normalised flux (PDC detrending can suppress long dips, so the model sees both)
  3. per-bin error
  4. mask
- **Two normalisations.** The shape view is flux standardised by robust scatter. Amplitude goes in separately as scalar context: log robust scatter, log peak-to-peak amplitude, Tmag. Otherwise per-star scaling hides how deep a dip is.
- **Augmentations (training only):** random crop or shift, flux noise matched to the per-bin error, random extra masking of gap-like spans, small amplitude jitter.
  - **Never time-reverse.** Asymmetry (fast drop, slow recovery) is a signal for exocomets and dusty tails.
  - **Never inject astrophysical-looking dips as augmentation.** That would teach the model they're normal.

## 2. Self-supervised encoder
- **Architecture:** a patch-based transformer in the PatchTST style, with patches of about 16 bins (8 h), roughly 2–10M parameters. A 1D-CNN encoder is the fallback. It must fit a **4 GB GTX 1650**: fp16 autocast with GradScaler (no bf16 on Turing), batch size of about 64–256, gradient accumulation if needed.
- **Objective:** masked patch modelling (MAE-style). Hide about 40% of patches in contiguous blocks. The loss covers **observed bins only**, as a Gaussian NLL using each bin's error, so noisy bins count less.
  - Optional add-on: a contrastive term between two augmented crops of the same star-sector.
- **Embedding:** the mean-pooled encoder output plus a projection of the scalar context, giving about 64–128 dimensions.
- **Baseline encoder:** frozen **EncoTESS** embeddings, MIT-licensed. Expect a domain shift, since it was trained on 2-min SPOC data and ours is 30-min FFI. Report it as a baseline, not the default.

## 3. Training hygiene
- **Splits by star (TIC), never by row.** The same star in another sector stays in the same split.
- **A held-out sector** checks generalisation (e.g. train on S32 plus others, test on an unseen sector).
- **Exclude** known positives and all injection-set stars from training. Log the exclusion list.
- **Optimiser:** AdamW with cosine LR and warmup. Early stopping on validation masked NLL. Fixed seeds.
- **Every run writes** `runs/<run_id>/` (gitignored): config YAML, git SHA, data manifest (sectors, product versions), metrics JSONL, best checkpoint. Log to local TensorBoard. No cloud trackers.
- **Required sanity checks before a scale-up:**
  1. Overfit a tiny batch to near-zero loss.
  2. Plot reconstructions for random, faint, bright and known-weird stars.
  3. **Nuisance probe:** a linear probe predicting camera, CCD and Tmag from the embedding. High accuracy for camera or CCD means systematics have leaked into the embedding. Fix it with the contrastive or neighbour-differencing tricks before going further.
  4. A UMAP of the embeddings coloured by Tmag, camera/CCD and known-positive class.

## 4. Anomaly scores
Compute several scores and keep them all:
1. **kNN density:** mean cosine distance to the k nearest neighbours in embedding space (k ≈ 20; FAISS-CPU or sklearn).
2. **Reconstruction surprise:** masked-NLL averaged over several random masks. Report its time-resolved version as well, to show *where* in the light curve it's weird.
3. **Isolation Forest** on the embedding plus scalar context.
4. **Classical features (baseline):**
   - skewness
   - flux asymmetry M and quasi-periodicity Q (Cody et al. 2014 dipper metrics)
   - biggest single-dip depth and duration
   - the asymmetry of the largest dip
   - excess variance

5. **Known-class score (open-set recognition):** train a light classifier on the embeddings using catalogue labels for the common variable types:
   - Gaia DR3 `vari_classifier_result`
   - TESS eclipsing-binary catalogues
   - VSX types (EB, δ Scuti/γ Dor, RR Lyr, rotational/spotted, Cepheid, …)
   - plus a "quiet" class

   A star that confidently matches a known class (high max-probability, low energy score) is **explained away** and down-weighted. "Weird" means far from the crowds **and** fits no known class. Never train this on the anomaly target classes (dippers, Boyajian-like, exocomets). They must stay out-of-distribution.

Combine scores by **rank-averaging**. **Conditional calibration:** convert each score to a percentile *within strata* of Tmag bin × camera/CCD. Faint stars are noisier, and noise shouldn't count as "weird".

## 5. Systematics rejection
- Mask SPOC quality-flagged cadences, plus the known scattered-light and momentum-dump windows for each sector.
- **Common-mode test:** for each anomalous time window, measure the fraction of stars on the same CCD (and nearby on the sky) that are also anomalous in that window. If it's high, it's instrumental: down-weight or flag it.
- Flag stars whose time-resolved anomaly falls only in orbit-edge or scattered-light windows.

## 6. Evaluation (pre-registered)
- **The injection set** comes from real light curves, stratified by Tmag and CCD, and is held out of training. Injected types:
  - dipper (quasi-periodic and aperiodic)
  - exocomet (asymmetric exponential)
  - single deep dip
  - Boyajian-like complex dips
  - outburst (brightening)

  Depths 0.1–30%, durations 2 h–5 d.
- **Metrics:** recall within the top 1% and top 0.1% by type, depth and Tmag; AUROC; precision@k on known positives.
- **Null set:** injected *instrument-like* events (the same dip on every star of a CCD). These should **not** rank high. Report their false-positive rate.
- **Gate (from PLAN.md):** at least 50% recall at 0.3% depth, and known positives in the top 1%.
- **Ablations:** ML vs classical vs EncoTESS; with and without the amplitude context; with and without the SAP channel.
- **Stability:** the top-100 overlap across 3 seeds. Unstable rankings are not reported.

## 7. Human in the loop
- A local gallery app shows the top-ranked stars: the light curve, the time-resolved anomaly, CCD neighbours, and a TESScut pixel thumbnail.
- **Labels:** `interesting`, `systematic`, `known-boring-variable`, `unsure`. Store them as CSV in the repo.
- **Re-ranker:** a light classifier (logistic regression or GBM) over embeddings plus scores, trained on the labels.
- **Choosing what to label next:** a mix of exploitation (the re-ranker's top picks), uncertainty sampling, and **pure novelty** (unsupervised score only). The pure-novelty share stops the loop from teaching itself to see only what we already like. Start with about 30% exploration.
- **Vetting before calling anything "new":**
  - cross-match SIMBAD, VSX, Gaia DR3 variability and the TOI/CTOI lists
  - check pixel-level contamination using TESScut and Gaia neighbours
  - check the star's other sectors
  - check the same time window in its CCD neighbours

## 8. Compute budget
- One sector is about 158k light curves × 1,300 bins in fp16, roughly 0.5 GB, so it fits in RAM in chunks. Ten sectors, about 5 GB on disk, are streamed from Parquet shards.
- Expect minutes per epoch on the 1650. If training becomes the bottleneck, use Kaggle's free GPUs (16 GB) rather than a bigger model locally.

## 9. Interfaces (shared contract between stages)
- **Splits:** `oddball.common.splits.split_of(tic)` → `train` / `val` / `test` / `known`. Hash-based by TIC, 80/10/10. Every known positive is `known`.
- **Light-curve shards:** the ingest format (`src/oddball/tess/ingest.py`, read with `oddball.tess.store`).
- **Injection sets:** `data/injections/<name>/s<NNNN>/{val,test}/`. Same shard schema as ingest, plus columns `inj_id` (str, unique) and `host_tic` (int), plus an `injections.parquet` table with one row per `inj_id`: `host_tic, sector, split, kind, depth, duration_d, t0_btjd, params_json, is_null`. `is_null=True` marks instrument-like null injections. Hosts are drawn **only** from the split named by the directory.
- **Score files:** `outputs/scores/<scorer>/s<NNNN>/{real,inj_<name>}.parquet`, with columns `tic, sector, inj_id (null for real), score` (higher = weirder), plus any `score_*` sub-scores. A scorer must be runnable on any shard directory, real or injected.
- **Evaluation** reads score files only, so any scorer (classical, EncoTESS, our encoder) plugs in identically.

## 10. Implementation record: `ssl_v1` (S32, 2026-11-09)

Code: `src/oddball/ml/`, `scripts/ml_{prepare,train,sanity,score,quicklook}.py`. Reasoned changes
to §1-4, each measured:
- **fp32, not fp16.** On the GTX 1650 Super with the torch 2.15 + cu13.2 wheel, fp16 autocast
  was 2x slower per step (fp16 matmul 0.42 vs 2.26 TFLOPS fp32). The 2.0M-param model fits in
  ~2 GB in fp32. `--amp` re-enables autocast.
- **Common gap mask.** The per-CCD gap pattern alone predicts camera at 96% and CCD at 86%
  (chance 25% / 7%; linear probe on observed fraction per patch). Every star therefore uses
  only bins where at least 90% of stars on *every* CCD have data: 1128 of 1280 grid bins, about
  3% of the real data. With it, the mask-only probe is at chance.
- **Shape view compression.** Robust-sigma units, then `10*asinh(x/10)` (linear within 10
  sigma), so 30% dips on quiet stars keep their order without exploding the NLL. Errors are
  propagated through the transform.
- **beta-NLL (beta=0.5) for training.** Plain Gaussian NLL with learned excess variance
  collapsed to "predict the median, inflate the variance" (run `ssl_v1_s0`). Early stopping and
  the reported val NLL stay plain NLL.
- **Reconstruction surprise = excess NLL** `0.5 (z^2 - 1)`, so a star's noise level does not
  shift its score. Star score = mean of its 4 most surprising patches. Each patch is hidden in
  2 of 5 block masks.
- **kNN** excludes the query's own TIC, and for injections the host TIC, from the train+val
  reference.
- **Calibration** uses `oddball.common.strata` (the evaluator's 5 Tmag bins x 16 CCDs). A
  stratum with fewer than 300 reference stars falls back to its Tmag bin.
- **Known flaw in the shipped `ssl_v1` (run `ssl_v1_s0`): its reconstruction does not use
  context.** It used learned positions initialised at std 0.02, so every hidden token was
  `mask_token + ~0`. It predicts the median with an inflated variance at every hidden position:
  on 727 signal-dominated val stars, masked MSE is 1.64 vs 1.70 for predicting 0, and
  correlation with a clean 0.7-d sinusoid behind a single hidden patch is ~0. Fixed sin-cos
  positions are now the default (`pos_mode`). beta-NLL alone did not fix it, and a 400-step
  sin-cos trial did not yet use context either. **Next:** train the sin-cos model much longer
  (MAE-style schedules run 100s of epochs; use Kaggle) and re-check with
  `sigcopy`/`ml_sanity.py --copy` on signal-dominated stars. The shipped `score_recon` is
  effectively "deviation from the star's median in units of a learned per-star scatter". It
  localises events well (see the time-resolved maps) but is not a learned prediction.

### `ssl_v2` (train S33/38/43/44/55, score held-out S32, 2026-11-09)

Run `ssl_v2_s{0,1,2}` (3 seeds). Same data module, splits and exclusions as `ssl_v1`; reasoned
changes, each measured:
- **Held-out sector.** Trained on the `train` split of S33, S38, S43, S44, S55 (625,665 light
  curves, `MultiCache`), S32 never seen. Each sector keeps its own grid origin and its own
  common-gap mask; S32's mask is computed at scoring time from its real train + val stars
  (which bins have data, no labels): 1128 of 1280 bins, as for `ssl_v1`.
- **No SAP channel** (`--drop-sap`): PDCSAP + error + mask + scalar context only. Cost: PDS 110,
  whose dip PDC suppresses, falls from the 0.976 to the ~0.85 percentile.
- **Hybrid encoder** (`arch="hybrid"`): 3 residual dilated-conv blocks over the patch tokens
  (receptive field +-14 patches), then 8 pre-norm transformer layers; LayerNorm on the patch
  embedding before the sin-cos positions are added (`embed_norm`). 1.91M parameters.
  **Why:** the sin-cos fix alone did not stop the collapse. Measured with
  `oddball.ml.diagnostics` (masked-bin MSE on 600-1000 signal-dominated val stars, and a
  synthetic-sinusoid test): the pure transformer + `embed_norm` on S33, 1,200 steps: gain over
  predicting the median +0.03, sinusoid correlation 0.00. On pure synthetic sinusoids it reached
  correlation 0.15 after 900 steps, while the hybrid reached 0.90 in 300 steps and the
  CNN fallback 0.85 in 400. The hybrid on S33 at 1,600 steps: gain +0.18, sinusoid 0.53. A
  transformer with only absolute positions learns positional attention very slowly; a local conv
  stem gives a hidden token its neighbours at once.
- **Training length.** 8 x 1,500 steps of 256 (12,000 steps, ~4.9 passes over the training
  set), cosine LR, fp32, ~80 min per seed on the GTX 1650S. Val NLL still falling slowly at the
  end (0.5233 -> 0.4655 for seed 0).
- **Scoring mask block 8 patches** (was 4). Chosen on **val** injections only (test and known
  untouched): reconstruction-surprise AUROC 0.818 -> 0.836, recall@1% 0.427 -> 0.462, null
  FPR 0.553 -> 0.497 (block 2: worse; 16: similar to 8).
- **Systematics veto.** Every ML score file carries `cm_frac, dump_frac, sys_frac, sys_flag,
  n_anom` copied from the classical scorer's files for the same light curve (by TIC; by
  `inj_id` for injections), i.e. `oddball.score.classical.systematics_features` with the
  binomial common-mode test for injection sets. Nothing is reimplemented; the evaluator applies
  the veto symmetrically. Caveat: `sys_flag` is set on 77.5% of real S32 stars, so "with veto"
  ranks among the unflagged 22.5%.
- **Variants written** (same model, same sub-scores): `ssl_v2` (pre-registered rank-average of
  kNN, recon, IsoForest), `ssl_v2_recon` (recon only), `ssl_v2_iforest` (IsoForest only;
  best on val at ~0.3% depth and null FPR, so it is a val-selected variant, not pre-registered).
- **Stability bar.** §6 gives no number; we read "stable" as a mean pairwise top-100 overlap of
  at least 0.5 across the 3 seeds (`scripts/ml_stability.py`, same rule as `ml_quicklook.py`).
- **Reconstruction now uses context** (held-out S32, `ml_sanity.py --context`, seeds 0 / 1):
  on 725 signal-dominated val stars (median error < 0.3 robust sigma), masked-bin MSE with
  training-like 40% block masks is 0.930 / 0.922 vs 1.521 for predicting the median (gain
  0.39) and 2.031 for linear interpolation; with single hidden 8-h patches 0.741 / 0.702 vs
  1.556 (gain 0.52-0.55). Clean 0.7-d sinusoids behind one hidden patch: correlation 0.87 /
  0.88 (4 hidden patches: 0.90 / 0.94); 2-d sinusoids, pooled correlation 0.95. `ssl_v1` scored
  1.64 vs 1.70 and ~0 on the same checks.
- **Nuisance probe** (pooled embedding, held-out S32, train-split fit / val-split test): camera
  0.368 / 0.363 vs chance 0.259 (`ssl_v1`: 0.55 vs 0.26); camera+CCD 0.160 / 0.152 vs 0.069;
  Tmag bin 0.50 vs 0.41. Smaller but not zero: some instrument signature remains.
- **Seed stability (3 seeds, real S32, mean pairwise top-100 overlap).** Combined `score`
  0.41 (fails), `score_knn` 0.20, `score_recon` 0.84, `score_iforest` 0.78. The variant files'
  stratified `score`: `ssl_v2_recon` 0.76, `ssl_v2_iforest` 0.70 (both pass). kNN is the
  unstable part: larger k does not help (seeds 0/1 kNN overlap 0.13 / 0.13 / 0.18 at
  k = 20 / 100 / 300; combined 0.39 / 0.37 / 0.25). So no ranking from the combined `ssl_v2`
  score is reported; rankings come from `ssl_v2_recon` or `ssl_v2_iforest`.
- **Headline (S32 test, seed 0; seeds 1-2 within ~0.02 except IsoForest null FPR):** see
  PLAN.md results log. Null FPR is worse than `ssl_v1` without the veto (0.53 vs 0.41) and
  0.16 with it; the val-selected `ssl_v2_iforest` has 0.25 (0.03 with veto), but its null FPR
  varies by seed (0.25 / 0.48 / 0.24).
