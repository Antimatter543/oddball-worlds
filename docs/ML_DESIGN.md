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
