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
- Implemented per event as `cmv2` (§12). It replaces the per-star v0 `sys_flag` in the evaluator.

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
- **Injection sets:** `data/injections/<name>/s<NNNN>/{val,test}/`. Same shard schema as ingest, plus columns `inj_id` (str, unique) and `host_tic` (int), plus an `injections.parquet` table with one row per `inj_id`: `host_tic, sector, split, kind, depth, duration_d, t0_btjd, params_json, is_null`. `is_null=True` marks instrument-like null injections. Hosts are drawn **only** from the split named by the directory. A set may repeat a host in several rows (`yso1`, §15): it then carries a `replica` column (each host once per replica; `-1` = the matched null, an unchanged copy) and a `hosts.parquet` with each host's catalogue match. Cross-star vetoes must then run per replica and with the real sector's masks only (`scripts/veto3_inj_set.py`, `veto_inj_set.py --real-masks-only`).
- **Score files:** `outputs/scores/<scorer>/s<NNNN>/{real,inj_<name>}.parquet`, with columns `tic, sector, inj_id (null for real), score` (higher = weirder), plus any `score_*` sub-scores. A scorer must be runnable on any shard directory, real or injected.
- **Evaluation** reads score files only, so any scorer (classical, EncoTESS, our encoder) plugs in identically.
- **Known-class feature tables** (§4.5): `outputs/knownclass/features/<table>/s<NNNN>/{real,inj_<name>}.parquet`, keys `tic, sector, inj_id`, any numeric feature columns. Tables join on the keys, so ML embeddings plug in next to `kcf_v1`.
- **Known-class predictions:** `outputs/knownclass/<model>/s<NNNN>/{real,inj_<name>}.parquet`: `kc_pred, kc_pmax, kc_entropy, kc_energy, kc_p_<class>, kc_t_<class>, kc_w_<class>, kc_explained`. `scripts/explain_away.py --base <scorer>` turns any scorer into `<scorer>_ka`.

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
  ranks among the unflagged 22.5%. Superseded by the per-event `cmv2` veto (§12), which the
  evaluator now uses by default (`--veto sys_flag` reproduces these numbers).
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

## 11. Implementation record: known-class explain-away (`kc_v1`, `classical_ka`, 2026-11-09)

Code: `src/oddball/knownclass/`, `scripts/{build_known_class_labels,known_class_features,known_class_train,explain_away,known_class_report}.py`. Reasoned changes to §4.5, each measured on **val** only:

- **Labels** (per TIC, sectors 32/33/38/43/44/55): Prsa+2022 TESS EBs (VizieR `J/ApJS/258/16`) > VSX (`B/vsx/vsx`, nearest entry within 5″) > Gaia DR3 `vari_classifier_result` (VizieR `I/358/vclassre`, score ≥ 0.5, within 2″). The ESA Gaia archive TAP was not usable anonymously: upload crossmatches of 500 positions hit the 60-s anonymous execution limit (HTTP 408 / "Maximum execution time (60 s) reached"), so the SOS tables (`vari_summary` flags) are not used and the Gaia match is positional only (no G-mag check). Classes: EB, DSCT_GDOR, RR, CEP, LPV, ROT (incl. BY/RS/ACV/solar-like), ELL, BCEP_SPB, quiet.
- **Hard exclusions** (never a label, never quiet): any VSX entry within one TESS pixel (21″) or Gaia source within 3″ of a young-star / irregular-dimming type (`labels.VSX_EXCLUDE_EXACT`/`_PREFIX`: YSO, TTS/CTTS/WTTS, DYPER, FUOR/FU, EXOR, UXOR/UX, ORION, IN*, IS*, I/IA/IB, DIP, RCB, BOY, HAEBE; Gaia YSO, RCB), every TIC in `docs/known_positives.csv`, every injection host.
- **Targets are per light curve.** A catalogue class counts only when the sector's light curve varies (`var_frac > 0.3` or dip SNR > 10); quiet = no catalogue flag of any kind and `var_frac < 0.15`, dip and bump SNR < 6. Quiet is capped at 3x the largest class.
- **Features (`kcf_v1`):** the classical features (PDC + SAP), FFT periodogram features (peak period / power fraction / amplitude, high- and low-frequency fractions, harmonic ratios), folded-profile shape at P and 2P, dip-event counts and their spacing irregularity (robust to events lost in gaps), stellar context. ~19 min per sector with 3 sectors in parallel.
- **Closed-set over-confidence is the main failure.** The classifier pushed injected dips into EB/ROT/DSCT with high probability (dry run: 0.1%-deep injections at median p = 0.94; 36% of all injections "EB"). Two guards were compared on val: an `other` catch-all class (objective 0.155) and **typicality** (0.174, chosen): a kNN check, in *shape* features only (no amplitude/SNR/context, so a 40%-deep EB is still a typical EB), of the light curve's distance to each class's training members, as a p-value against the members' own distances (other-TIC neighbours only). The class-conditional-percentile rule (score relative to the class's score distribution) was also tried and rejected: a deep EB is extreme even among EBs, so it was not explained away (val objective at best 0.128).
- **Combiner (`explain.downweight`):** `score = pct_stratum(1 - (1 - s)^(1 - alpha*E))`, `E = sum_c p_c * min(1, t_c / tau)` over the explaining classes (all but quiet), `s` = the base scorer's stratum percentile. alpha and tau are picked on val injections (mean recall in the top 1% at ~0.3 / 1 / 3% depth). Injected stars inherit their host's *catalogue* class but are scored only from their own light curve: a host EB with an injected dip is explained away only as far as the light curve still looks like a typical EB.
- **Catalogue fetch:** all 745,630 TICs cone-matched against VSX and Gaia vclassre, with no failed positions. The CfA mirror's database failed for some sky regions, so the remaining chunks came from the IDIA mirror (`vizier.idia.ac.za`). The two mirrors returned identical matches on a 1000-star check. `provenance.json` records which mirror served each chunk.
- **Labels:** 58,765 labelled TICs (train 47,415 / val 5,617 / test 5,733). By class: ROT 33,805, DSCT_GDOR 15,829, EB 7,318 (Prša 1,448), LPV 1,088, BCEP_SPB 277, RR 232, ELL 128, CEP 88. Excluded: 6,600 injection hosts, 1,386 VSX young/irregular types (top types YSO 904, TTS/ROT 164, WTTS/ROT 52, TTS 39, CTTS/ROT 35, IB 32, INB 29, INS 27), 279 Gaia YSO/RCB, 47 known positives.
- **Classifier `kc_v1`:** 59 boosting iterations, temperature 1.04. Val recall (n): EB 0.89 (870), DSCT_GDOR 0.91 (901), ROT 0.95 (3,037), LPV 0.91 (127), BCEP_SPB 0.80 (25), RR 0.45 (22), CEP 0.29 (7), ELL 0.00 (9), quiet 1.00. The rare classes are too small to learn.
- **Val sweep** picked alpha 0.5, tau 0.05 (objective 0.183 vs 0.127 at alpha 0). AUROC falls as alpha rises (0.867 at 0, 0.844 at 0.5, 0.48 at 1).
- **Test (S32):** recall in the top 1% at ~0.3 / ~1 / ~3% depth goes from 0.027 / 0.09 / 0.28 to 0.042 / 0.13 / 0.33. With the `sys_flag` veto it goes from 0.03 / 0.10 / 0.39 to 0.04 / 0.15 / 0.43. Null FPR 0.177 -> 0.197 (veto 0.013 -> 0.020). AUROC 0.870 -> 0.848. Known positives in the top 1%: 11 -> 10 of 18. Gate still fails.
- **What it does to the real top 1%:** 559 of 1,600 classical top-1% stars are explained away (35%; predicted ROT 305, EB 142, DSCT 86, LPV 26). 83% of their replacements carry `sys_flag`, and 139 are predicted quiet, so without the veto the freed slots go to systematics. With the veto only 213 stars (13%) swap, and the vetoed top 1% is still ROT 685 / EB 574 / DSCT 227 by prediction (EB 283 by catalogue).
- **Known positives:** the model gives 15 of 18 an explained share E >= 0.9 (mostly "typical ROT" or "typical EB"). Typicality does not separate dippers from rotators and EBs in these features. They survive only because their base percentile is extreme and alpha < 1. TIC 4350339 drops out of the top 1% (0.993 -> 0.987, predicted ROT, E 0.96). This is a structural limit of the down-weight. Pushing alpha higher would explain away the dippers too.
- Reports: `outputs/eval/compare/s0032/classical__classical_ka/report.md`, `outputs/knownclass/kc_v1/s0032/report_classical_ka.md`.

## 12. Implementation record: per-event veto `cmv2` and ensemble `ens_v1` (S32, 2026-11-10)

Code: `src/oddball/score/veto.py`, `scripts/{build_veto,tune_veto,veto_report,ensemble}.py`, the
evaluator's `--veto` option (default `cmv2`) and the injected-SNR diagnostic in `oddball.eval`.
Each change below was measured on **val** only. Test and the known positives were read only
after the choices were made.

**Why v0 failed.** `sys_flag` fires when at least 50% of a star's dip chi^2 sits in common-mode
bins. SAP carries common-mode scattered light in 22-26% of every CCD's bins (orbit starts and
ends, and the run-up to each momentum dump); PDCSAP has 2-6%. On most quiet stars the SAP ramp
outweighs everything else, so 77.5% of real S32 stars were flagged. "With veto" then ranked only
the remaining 22%, mostly stars with strong real variability. A weak injected dip next to a ramp
was thrown away along with the ramp: 30% of astrophysical injections were flagged.
- **PDS 110:** its SAP orbit-start ramp (BTJD 2174) holds 79% of its dip chi^2 and the real
  2190-2192 dip only ~10%, so `sys_frac` is 0.81.
- **β Pic (T = 3.7):** its strongest PDC dips are its CCD's orbit-start (2174.3) and orbit-end
  (2185.6) common-mode dips, plus a dump-adjacent SAP dip, so `sys_frac` is 0.63.

**`cmv2` design.** The module docstring has the details.
- **Events.** One series per channel, sign and box width (2 h, 8 h, 32 h). An event is a run of
  bins with box SNR > 3 on the running-median residual. Each series is divided by its own robust
  scatter when that is above 1 (red noise), so an event's SNR measures how much it stands out
  within its own light curve.
- **Common-mode maps.** Per series x CCD, and per series x CCD x 512-px cell (4 x 4 cells from
  `tess-point` pixel positions). A cell is common-mode when the fraction of anomalous stars
  exceeds the median + 8 sigma, with at least 5 stars. Sigma is the MAD, floored at binomial
  noise. Injection sets add their own binomial CCD map (p < 1e-7, as in v0).
- **Known windows from QUALITY.** Momentum dumps: bit 32 set on most stars (BTJD 2179.48 and
  2193.48). SPOC scattered light: bit 4096 set on at least 20% of a CCD's stars (~2185.2-2185.95
  and 2186.95-2187.2, 25-55 bins per CCD). Both are widened by one bin.
- **Per-event decision.** An event is systematic when at least 50% of its SNR^2 lies in
  common-mode or scattered-light bins, or when its peak falls between 1 bin before and 1 + W/2
  bins after a dump. Then, within each channel, an event also counts as systematic if it lies
  mostly inside a systematic event of the same or a *narrower* box, or contains the peak of one
  at least as strong as itself. The first version allowed any box width. The 32-h view of
  PDS 110's real dip then touched the SAP pre-dump window and relabelled its 2-h and 8-h events.
  A unit test now covers this case.
- **Per star.** `cmv2_res_snr` is the strongest non-systematic event (the astrophysical
  residual) and `cmv2_sys_snr` the strongest systematic one. Both also come in PDCSAP-only
  versions, along with the time and label of the strongest event. The flag is
  `cmv2_flag = sys_snr >= t_sys and res_snr < t_res`: a strong systematic event with nothing
  astrophysical left. A star with a real dip plus a shared ramp keeps its dip.
- **Thresholds** (`tune_veto.py`, val only). The objective is J = mean over classical and
  `ssl_v2_iforest` of (mean recall in the top 1% at ~0.3 / 1 / 3%) minus null FPR. The grid
  covers t_sys x t_res x {both channels, PDCSAP only}.
  - **One-SE rule:** among grid points within 0.01 of the best J (about one standard error on
    val), take the one that flags the fewest real stars. The tolerance was first 0.002. It was
    changed to one SE after the flat val plateau was seen; both choices are val-only.
  - **Chosen:** PDCSAP only, t_sys 10, t_res 10 (val J 0.274, 4.6% flagged). The best J (0.280)
    flags 14.5%. Any both-channel rule near the top flags more than 50%.
- **Cost:** one 3.5-min map pass and one 4.5-min event pass over S32, plus ~3 min of
  `tess-point` (cached).

**Veto results** (`outputs/veto/cmv2/s0032/report.md`). Evaluator rule: flagged real and
injected stars go to the bottom, then everything is re-ranked. Recall is in the top 1% at
~0.3 / 1 / 3% depth.

| scorer, split | veto: none / v0 / cmv2 | flagged real | null FPR | recall ~0.3 / 1 / 3% | astro flagged |
|---|---|---|---|---|---|
| classical, val | | 0 / 0.774 / 0.046 | 0.187 / 0.110 / 0.010 | .020 .063 .299 / .019 .122 .423 / .025 .100 .378 | 0 / 0.30 / 0.04 |
| ssl_v2_iforest, val | | same | 0.213 / 0.067 / 0.020 | .239 .403 .611 / .105 .289 .560 / .234 .383 .614 | same |
| classical, test | | 0 / 0.777 / 0.047 | 0.177 / 0.013 / 0.007 | .027 .092 .275 / .033 .104 .385 / .032 .112 .330 | 0 / 0.31 / 0.03 |
| ssl_v2_iforest, test | | same | 0.247 / 0.027 / 0.027 | .299 .422 .574 / .144 .311 .506 / .286 .413 .574 | same |

- v0's higher classical recall at ~3% comes from ranking against only the unflagged 22% of real
  stars.
- **Known positives:** `cmv2` flags 1 of 18, v0 flagged 2 (β Pic and PDS 110).
  - **β Pic** is flagged by PDC orbit-edge common-mode dips at SNR 17. Its strongest residual
    event, a 6-sigma 8-h dip at 2175.2, is below t_res. Its red-noise rescale is mild (1.5-2.1).
  - **PDS 110** is unflagged only because the rule is PDCSAP-only (its PDC systematic SNR is
    4.0). Its SAP dip survives as a residual event, but at only ~5 sigma against the star's own
    variability (the SNR series' scatter is 11-25x white noise). A both-channel rule would flag
    it: its SAP orbit-start ramp has SNR 33.5.

**Ensemble `ens_v1`** (`scripts/ensemble.py`).
- **Inputs.** Each component is a stratum percentile `p_k`, with surprise `s_k = -log(1 - p_k)`.
- **Candidate combiners:** rank-average; max; weighted max `max_k w_k s_k`, with w in {0.5, 0.75,
  1.25, 1.5, 2} relative to classical; Fisher (sum of s_k); Stouffer.
- **Pipeline:** combine, re-percentile within stratum, then apply `cmv2` (flagged stars get
  score -1). The val objective is the same J as for the veto.
- **Component set** (also chosen on val):
  - {classical, ssl_v2_iforest}: best J 0.388 (weighted max 1 : 1.5, which is in effect ML
    alone; ML alone scores J 0.390).
  - Adding `ssl_v2_recon`: J **0.505** (weighted max 1 : 1.5 : 1.5; plain max 0.457, Fisher
    0.394, rank-average 0.281). IsoForest leads at ~0.3% depth, reconstruction at 1-3%.
- **`ens_v1_ka`** adds `explain.downweight` on top (alpha 0.25, tau 1.0; val J 0.525).

Test results, with the cmv2 veto applied:

| scorer | recall ~0.3 / 1 / 3% | all depths | null FPR | known top 1% |
|---|---|---|---|---|
| classical | 0.032 / 0.112 / 0.330 | 0.284 | 0.007 | 12/18 |
| ssl_v2_iforest | 0.286 / 0.413 / 0.574 | 0.470 | 0.027 | 3/18 |
| `ens_v1` | 0.303 / 0.574 / 0.788 | 0.614 | 0.040 | 2/18 |
| `ens_v1_ka` | 0.324 / 0.607 / 0.788 | 0.628 | 0.050 | 2/18 |
| `ens_v1_2c` (no recon) | 0.283 / 0.407 / 0.566 | 0.472 | 0.027 | 5/18 |
| `ens_v1_max` (post hoc, equal weights) | 0.259 / 0.510 / 0.752 | 0.584 | 0.027 | 6/18 |

- **The ensemble loses the known positives.** A weight above 1 on the ML scores lets their top
  ~0.5% outrank anything classical can reach. In a stratum of ~2,000 stars, classical's best
  percentile is ~0.9997 (s ~8), while 1.5 x s at an ML percentile of 0.995 is 7.9. Most known
  dippers sit at classical 0.99-0.9997, so in the ensemble they fall to 0.93-0.99. Val
  injections cannot show this cost. The known positives are evaluation-only, so the weights
  were not changed. `ens_v1_max` was built *after* seeing this, to measure the trade-off; it is
  not a selected scorer.
- **Seed stability** (ssl_v2 seeds 0 / 1 / 2, same rule via `--fixed ens_v1`):

  | metric | seed 0 / 1 / 2 |
  |---|---|
  | recall at ~0.3% | 0.303 / 0.286 / 0.237 |
  | recall at ~1% | 0.574 / 0.614 / 0.555 |
  | recall at ~3% | 0.788 / 0.775 / 0.778 |
  | null FPR | 0.040 / 0.037 / 0.033 |
  | known positives in the top 1% | 2 / 4 / 2 of 18 |

  The real top-100 overlap is 0.68 (pairs 0.71 / 0.66 / 0.67), which passes the 0.5 bar;
  `ssl_v2_iforest` alone scores 0.70.
- **Recall vs injected SNR** (test, `ens_v1`, ~0.3% depth, n = 598):

  | SNR | n | recall |
  |---|---|---|
  | < 3 | 32 | 0.03 |
  | 3-10 | 88 | 0.07 |
  | 10-30 | 248 | 0.17 |
  | 30-100 | 197 | 0.54 |
  | > 100 | 33 | 0.82 |

  20% of the ~0.3% bin is below SNR 10 and essentially unrecoverable. Reaching 50% overall
  would need ~0.64 in the SNR 10-30 bin.

## 13. Implementation record: per-event veto `cmv3`, null set `n2` and ensemble `ens_v2` (S32, 2026-11-10)

Code: `src/oddball/score/veto3.py` (module docstring has every rule and threshold),
`src/oddball/inject/nulls2.py`, `scripts/{make_nulls2,ml_score_inj,build_veto3,tune_veto3,
veto3_report,skybot_check}.py`, `tests/test_veto3.py`. The evaluator takes `--veto cmv3`
(the default stays `cmv2`) and `ensemble.py --veto cmv3` builds `ens_v2`. This implements
docs/VETTING.md §4 build-order items 1 and 3 (F1, F2, F3, F4a/b, F5, F8, F12). `cmv2` is
unchanged; its outputs are reused.

**Labels.** `cmv3` adds stage-1 labels to `cmv2`'s per-event decision, before stage 2. A
relabelled event therefore still drags its wider-box views with it, and a star keeps any real
event the new labels do not touch.

| label | VETTING | an event is systematic when |
|---|---|---|
| `bkg` | F1 | SAP moves >= 3 sigma, the background moves the other way by >= 0.5 of that, and `sap_flux + bkg` has box SNR < 3 |
| `pdc` | F12 | a PDCSAP event whose SAP depth / CROWDSAP is < 0.5 of the PDCSAP depth (>= 3 sigma). SAP-deeper (PDS 110) is never labelled |
| `gap` | F4b | within 2 box widths of an orbit edge or a >= 0.3-d gap, with mean \|d bkg/dt\| >= the CCD's 90th percentile |
| `qbits` | F4a | >= 50% of SNR^2 in bins with QUALITY 16/128/2048/32768 (own flags, or set on >= 20% of the CCD). In S32 only 128 and 32768 occur |
| `dumpedge` | F5 | the event run *starts or ends* within 1 + W/2 bins of a dump (a step; `cmv2` checks only the peak) |
| `track` | F2 | a straight, steady track (5-300"/h, >= 3 stars) of short background or SAP bumps passes within 6 px during the event |
| `flank` | F2 | a bump with a `bkg`-labelled dip of the same star 1-24 bins away (dip-spike-dip: a body crossing the aperture) |
| `column` | F8 | >= 3 other stars within +-2 columns have an event of the same series at the same bin, Poisson p < 1e-4 against control columns 10-60 px away |

Thresholds are the VETTING priors and were not tuned. Which labels to enable and the star flag
(`sys_snr >= t_sys and res_snr < t_res`, as in `cmv2`) were chosen on val.

**Null set `n2`** (`nulls2.py`; val 1,157 light curves, 329 evaluated; test 1,085, 331
evaluated). This is the VETTING §2 list.
- **`null_asteroid`.** A body on a straight line (15-80"/h, 3-10 d) across real stars.
  In the aperture (1-px PSF) it adds to SAP. In the background pixels (2.5-6 px) it raises
  `bkg` and lowers SAP by K_BKG = 0.5 of its flux. PDCSAP gets the SAP change / CROWDSAP.
  Every star the track passes gets the signal; only stars of the set's split are evaluated
  (the rest are `split = "carrier"` and never ranked).
- **`null_step`.** A step at a momentum dump with exponential recovery (tau 0.5-5 d) in 20
  stars of one CCD, random sign.
- **`null_column`.** One event shape in every star within +-1.5 columns of one CCD column.
- **Realism.** `n2` sets are vetoed with the real sector's common-mode maps only
  (`--real-masks-only n2`). The set's own binomial map would see a column's ~15 hosts as a
  CCD-wide event, which the real sky would not.
- **Cross-star tests.** These run on the set's events together with the real sector's
  (minus the hosts' own real copies).
- The ML scorers score `n2` through `ml_score_inj.py` from `runs/ssl_v2_s0/ref_state.pkl` (now `ref_state_s0032.pkl`, §16)
  (reproduces `inj_v1` exactly).

**Tuning** (`tune_veto3.py`, val only).
- **Grid:** 18 label configurations (`cmv2` alone, all labels, leave-one-out, only-one) x
  {both channels, PDCSAP only} x t_sys {5, 6, 8, 10, 15, 20, 30} x t_res {4, 5, 6, 7, 8, 10, 15}.
- **Objective:** J3 = mean over classical and `ssl_v2_iforest` of (mean recall@1% at ~0.3 / 1
  / 3%, v1 injections) minus the mean over the five null types (null_ccd, null_dump,
  null_asteroid, null_step, null_column) of val null FPR. The per-type mean keeps the 200 step
  nulls from outweighing the 55 asteroids. As for `cmv2`, the rule is one SE: within 0.01 of
  the best J3, take the point that flags the fewest real val stars.
- **Chosen:** all labels, PDCSAP only, t_sys 15, t_res 10 (val J3 0.277; 2.4% of real stars
  flagged, `cmv2` 4.6%). The best J3 (0.286) is at t_sys 5, t_res 7 and flags 24%.
- **Ablation** (best val J3 per configuration):

  | configuration | J3 |
  |---|---|
  | all labels | 0.286 |
  | no `pdc` | 0.286 |
  | no `qbits` | 0.286 |
  | no `track` | 0.283 |
  | no `gap` | 0.276 |
  | no `flank` | 0.274 |
  | no `bkg` | 0.261 |
  | no `column` | 0.236 |
  | no `dumpedge` | 0.201 |
  | `cmv2` labels only | 0.123 |

  `dumpedge` and `column` carry the step and column nulls, `bkg` and `flank` the asteroids.
  `pdc` and `qbits` add nothing on S32 val. The `qbits` windows coincide with `cmv2`'s dump and
  orbit-edge windows; no PDC-only null type exists to reward `pdc`. They are kept because they
  cost nothing measurable.
- **Pass 1 had no `flank`**, and track points were "run above SNR 3 <= 8 bins". It reached
  best val J3 0.273 and chose the same rule (2.0% flagged), but flagged only 3 of the 10
  eyeballed asteroids.
  - **Diagnosis** (`outputs/veto/cmv3/s0032/events_watch.parquet`): a bright crossing's run
    above SNR 3 is far longer than its core, so the brightest crossings made no track points.
    The spike in the aperture also stayed an unlabelled residual between two `bkg`-labelled dips.
  - **Fix:** points now use the width at half maximum, and `flank` was added. Both are
    VETTING F1/F2 signatures ("often dip-spike-dip"). Thresholds are priors, and the rule was
    re-tuned on val.
  - **Caveat:** the fix was motivated by the evaluation-only triage set, so the triage numbers
    below are not an independent test of it. Val and test nulls are.
- **Label prevalence** (light curves with >= 1 labelled event, real): `gap` 96%, `bkg` 84%,
  `dumpedge` 70%, `qbits` 69%, `flank` 47%, `pdc` 34%, `column` 11%, `track` 7.5%. Most of
  these are noise-level events. Among flagged real stars, the strongest systematic event is
  `cm_ccd` (85%), `flank` (5%), `track` (5%) or `bkg` (2%).
- **Chance tracks:** shuffling the point times within each CCD gives 7.6k track pairs (real
  31.9k) and 16k light curves in windows (real 21.9k). Most windowed light curves are chance
  passes, so `track` must stay an event label (an event has to overlap the window), never a
  star label.
- **Cost:** 37 min for S32 (scan 3.5 min, cross-star 1 min, classify 30 min for 18
  configurations, injections 3 min), with CPU shared with another job.

**Results** (`outputs/veto/cmv3/s0032/report.md`; evaluator rule; top 1%).

Test null FPR by type, veto none / `cmv2` / `cmv3`:

| scorer | ccd | dump | asteroid | step | column |
|---|---|---|---|---|---|
| classical | .17 / .01 / .01 | .19 / .01 / .01 | .04 / .06 / .00 | .31 / .32 / .02 | .26 / .31 / .01 |
| ssl_v2_iforest | .30 / .04 / .02 | .11 / .00 / .00 | .13 / .13 / .04 | .70 / .54 / .20 | .26 / .29 / .06 |
| `ens_v2` | .67 / .05 / .02 | .43 / .02 / .02 | .51 / .51 / .04 | .75 / .60 / .20 | .69 / .68 / .08 |

Recall@1% at ~0.3 / 1 / 3%, test, same order:

| scorer | none | `cmv2` | `cmv3` |
|---|---|---|---|
| classical | .027 .092 .275 | .032 .112 .330 | .027 .097 .291 |
| ssl_v2_iforest | .299 .422 .574 | .286 .413 .574 | .294 .407 .561 |
| `ens_v2` | .306 .593 .803 | .303 .590 .788 | .321 .590 .794 |

- **Step nulls stay the weakest:** test FPR 0.20 for the ML scorers (val 0.07).
  - `cmv3` flags 70% of test step nulls (val 60%). The unflagged ones are shallow: median
    depth 0.6%, against 3.1% for the flagged ones.
  - Their step events are mostly labelled systematic (`cm_ccd`, `dump`, `dumpedge`) but stay
    below t_sys = 15, and the ML scorers still rank the shape high.
  - Only 7 test step nulls keep a residual event at >= 10 sigma away from the dump (the
    recovery tail).
- **Classical loses a little at 3% depth** with `cmv3` (0.330 -> 0.291), because `cmv3`
  removes half as many real stars. Its v1 null FPR goes 0.007 -> 0.013.
- **Astrophysical injections flagged:** val 0.05 (`cmv2` 0.04), test 0.04 (`cmv2` 0.03).

Evaluation-only checks (none of them was used for tuning):
- **Known positives:** `cmv3` flags 1 of 18 (β Pic), the same star as `cmv2`.
- **Eyeballed S32 triage** (`labels/triage_s0032_v1.csv`, 96 stars):

  | class | n | `cmv2` flags | `cmv3` flags |
  |---|---|---|---|
  | artefact | 29 | 0 | 11 |
  | asteroid | 10 | 0 | 9 |
  | scattered light | 19 | 0 | 2 |
  | EB | 53 | 1 | 1 (167713851, flagged by both) |
  | ordinary variable | 14 | 0 | 0 |

  - The asteroid that is not flagged, 412059784, keeps a 32-h residual bump at SNR 12.
  - The scattered-light cards are mostly quiet stars that the ML picks for shape: their
    strongest events are at SNR 3-5, so no rule with t_sys >= 5 can flag them.
  - **Provenance of the labels:** the first triage pass stored only counts and named examples.
    18 unnamed artefact cards and 4 variables were re-labelled from the cards for this record
    (`source = triage-2`), matching the original 29 / 53 / 14 counts.

**`ens_v2`** = `ens_v1`'s recipe (weighted max of classical, `ssl_v2_iforest`, `ssl_v2_recon`
surprises; rule and weights re-chosen on val) with the `cmv3` veto. The rule chosen is
weighted max 1 : 2 : 2 (val J 0.524; 1 : 1.5 : 1.5 gives 0.519).

Test, `evaluate.py --compare classical ens_v1 ens_v2 --split test --veto cmv3`, v1 injections:

| scorer | recall ~0.3 / 1 / 3% | all depths | null FPR | known top 1% |
|---|---|---|---|---|
| classical (cmv2 veto) | 0.032 / 0.112 / 0.330 | 0.284 | 0.007 | 12/18 |
| classical (cmv3 veto) | 0.027 / 0.097 / 0.291 | 0.261 | 0.013 | 11/18 |
| `ens_v1` (cmv2) | 0.303 / 0.574 / 0.788 | 0.614 | 0.040 | 2/18 |
| `ens_v2` (cmv3) | 0.321 / 0.590 / 0.794 | 0.624 | 0.020 | 3/18 |

`ens_v2` still loses the known positives (3 of 18), for the reason given in §12. The larger
ML weights do not help them.

**SkyBoT (F3)** (`scripts/skybot_check.py`).
- **Method:** a 5' cone, observer C57, asteroids only, at the epoch of each star's strongest
  `cmv3` event. Run on the `ens_v2` top 200 that survive the veto plus the 96 triage stars:
  296 anonymous queries, ~20 min. Responses are cached under `data/skybot/s0032/`.
- **"Explained"** means a body with V <= 19 within 126" (6 px) over +-1 h.
- **Top 200: 2 explained.** These are ranks 52 (464770254, Novosibirsk, V 16.4) and 92
  (454191483, Sholokhov, V 15.5). Both are unlabelled PDC bumps that `cmv3` missed. 18 of the
  200 have some body within 300".
- **Triage artefacts: 5 of 29 explained.**
  - Nemausa (V 10.9) explains 396863456 and 454149279, and also 402215073, which the triage
    called scattered light.
  - Asporina (V 13.2) explains 298849841.
  - The fifth is 302345512.
- **Asteroid cards SkyBoT cannot explain.** Three (436707310, 302675203, 47398410) have only
  V 21-22 bodies within 126", far too faint for crossings at SNR 35-2800. Two (71026277,
  37951066, camera 2) have no catalogued body within 5'.
  - These may be comets (excluded by the asteroid filter), uncatalogued bodies, or the wrong
    epoch (the strongest event need not be the crossing). This is not resolved.
  - `cmv3` flags all five, from `track` or `flank`.
- **EBs and variables:** 0 explained.
- **Epoch:** BJD_TDB is sent as the epoch. Its offset from the spacecraft's UTC (barycentric
  light time up to ~8 min, plus 69 s) is <= 7" (0.3 px) at 40"/h, well inside the 126" match.

**Caveats and unverified.**
- **The `n2` physics is a model.** K_BKG = 0.5, a 1-px Gaussian PSF and a 2.5-6 px background
  annulus approximate SPOC's FFI apertures; real stamps vary. The step nulls assume one step
  per dump with exponential recovery.
- **The thresholds are VETTING priors.** Only the label set and the star flag were tuned.
- **S32 is a southern-hemisphere sector, so only camera 1 is fairly close to the ecliptic.**
  8 of the 10 eyeballed asteroids are on camera 1, two on camera 2. The asteroid labels need
  checking on S43/S44 (ecliptic), where VETTING expects the most crossings.
- **Asteroid nulls are few:** 55 on val and 53 on test, so an FPR of 0.04 is ~2 stars.

## 14. Implementation record: light-curve EB tests and `ens_v1_eb` (S32, 2026-11-10)

Code: `src/oddball/knownclass/ebtests.py`, `scripts/{eb_features,eb_explain,veto_inj_set}.py`,
the `null_eb` / `dusty_tail` kinds in `oddball.inject.sampling` (`make_injections.py --kinds
--exclude-sets`), Kostov+2025 and Gaia DR3 `veb` in `oddball.knownclass.{catalogs,labels}`.
Why: the eyeballed S32 top list was 53 EBs out of 96 stars, 35 of them in no catalogue we use
(PLAN, 2026-11-10 triage). `kc_v1` cannot separate EBs from dippers (§11: it called 15 of 18
known dippers "typical EB or rotator"), so EBs need tests that dippers fail by construction.

**Tests (`ebf_v1`, per light curve, PDCSAP, 30-min bins; docs/VETTING.md §3 E1-E4, E6, E7).**
The module docstring has the full recipe. In short: a 1.5-d running-median residual; events =
2-h box SNR > 5; an anchored period search on the two deepest events (window depths, so an
unresolved eclipse is not lost inside a bin); at the best period, the hit fraction, excess depth
scatter (also odd/even separately), timing residuals, width scatter, stacked-profile asymmetry,
a secondary at any phase, unexplained extra events and the physical-duration ratio for the TIC
radius. `eb_periodic` is the product of soft factors in [0, 1], so a single strong violation
(a dipper's varying depths, an exocomet's tail, extra dips) is enough to reject.
`eb_single_raw` (lone event: informative SNR x symmetric shape x no extra events) is kept as a
diagnostic. **E7 (`eb_xsec`):** the same star in S33/38/43/44/55 supports an EB when an event
there matches this sector's depth and width (within 1.5x) and, if both sectors are periodic,
the period (2%, also at 1/2 and 2x). Two lone events count fully when depth and width agree
within 1.15x and 1.25x (89% of catalogued-EB lone-event pairs in the train split, 25% for other
catalogued classes), else at half weight. Absence elsewhere never counts against an EB.
~150-310 s per sector with 5 workers.

**Check set `eb1`** (`--kinds null_eb null_eb dusty_tail --exclude-sets v1`, seed 20261110;
1,600 `null_eb` + 800 `dusty_tail` per split, hosts disjoint from v1). `null_eb` is a per-host
*null* that must not rank high: P log-uniform 0.4-40 d, primary depth 0.3-50%, physical
duration from the host radius, trapezoid or (grazing) V, secondary 0 or 5-100% at phase 0.5 or
eccentric. `dusty_tail` is a *target* (disintegrating planet): strictly periodic, P 0.3-3 d,
1-4 h, sharp ingress and exponential egress, each transit's depth U(0.05, 1) of the maximum. It
is the hardest case for an EB test (strict period, constant width), so it guards against the
test eating periodic oddballs. `eb1` was scored by classical, `ssl_v2` (seed 0) and the `cmv2`
veto (`veto_inj_set.py`: the real maps reused, flag rule unchanged; 1.4% flagged).

**Combiner (`eb_explain.py`).** The base statistic is recomputed from `ens_v1`'s rule (asserted
equal to its `score_raw` on real, v1 and eb1), then the §11 down-weight with `E = eb_like`:

    eb_like = max(eb_periodic, w_single * eb_single_raw, eb_xsec * c_shape * c_extra)
    score   = pct_stratum(1 - (1 - s)^(1 - alpha * eb_like)),  cmv2-flagged -> -1

`alpha` in {0, .25, .5, .75, .9, 1} and `w_single` in {0, .25, .5, .75, 1} are chosen on **val**:
J = mean v1 recall in the top 1% at ~0.3 / 1 / 3% - v1 null FPR - eb1 EB-null FPR, with the
dusty-tail recall not allowed to fall more than 0.02; ties within 0.01 go to the smallest
intervention. The cross-sector weight is fixed at 1 a priori: val cannot see it (the injected
EBs have no other sectors, and injected targets get support only when the host's own eclipses
match them).

**Val sweep** (`outputs/scores/ens_v1_eb/s0032/provenance.json`): alpha 0.75, w_single 0
(J 0.213; best 0.221 at alpha 0.9, inside the tie band). Every w_single > 0 lowers J: a lone
symmetric eclipse and the `single_dip` target are the same thing in one sector, and lone events
are where most real EBs in our top list sit (below).

| val | ~0.3% | ~1% | ~3% | null FPR | EB-null FPR | dusty-tail recall |
|---|---|---|---|---|---|---|
| alpha 0 (= `ens_v1`) | 0.256 | 0.521 | 0.798 | 0.020 | 0.696 | 0.279 |
| chosen | 0.307 | 0.606 | 0.828 | 0.027 | 0.341 | 0.365 |

**Test (S32, `cmv2` veto):** `outputs/eval/compare/s0032/classical__ens_v1__ens_v1_eb/report.md`
(v1) and `report_eb1.md` (eb1).

| test | ~0.3% | ~1% | ~3% | all | null FPR | known top 1% | EB-null FPR | dusty-tail recall |
|---|---|---|---|---|---|---|---|---|
| classical | 0.032 | 0.112 | 0.330 | 0.284 | 0.007 | 12/18 | 0.191 | 0.039 |
| `ens_v1` | 0.303 | 0.574 | 0.788 | 0.614 | 0.040 | 2/18 | 0.664 | 0.279 |
| `ens_v1_eb` | 0.348 | 0.649 | 0.817 | 0.655 | 0.053 | 2/18 | 0.323 | 0.347 |

- **No target kind loses recall** at any depth bin except aperiodic dippers at ~0.3% (0.42 ->
  0.41). Targets with `eb_like > 0.5` on test: 0.4-1.1% per kind (dusty tails 1.1%). The
  gain is the mechanism working: real EBs leave the top 1%, so its threshold drops.
- **Cost:** null FPR 0.040 -> 0.053 (val 0.020 -> 0.027). Some freed slots go to systematics.
- **Known positives:** none has `eb_like` above 0.002, and all 17 unvetoed ones keep or raise
  their percentile (e.g. TIC 43488669 0.996 -> 0.998), but still only 2/18 are in the top 1%.
  The limit is still `ens_v1`'s ML weights (§12), not EBs. Diagnostic, not selected: the same
  rule on the equal-weight `ens_v1_max` (`--fixed ens_v1_eb`) gives 6/18 -> 7/18 and ~0.3% recall
  0.259 -> 0.309. Two known positives drop 0.002 there (TIC 50745441, 4350574): `combine` clips
  percentiles at 1 - 1e-7, so the top of a stratum is tied, and a near-zero `eb_like` (< 0.001)
  breaks the tie downwards.
- **EB nulls still in the top 1%** (test, approx.): 60% of P > 13 d (lone eclipses), 34% of
  P < 2 d, 7% of P 2-6 d. The short-period survivors have `eb_like` above 0.5 but base
  percentiles so extreme that alpha 0.75 does not bring them below the cut.

**Real stars (evaluation only, nothing tuned on them).** Of the 53 eyeballed EBs, 48 can be
identified from the record: the 35 uncatalogued TICs in the triage plus the 13 gallery stars
with a catalogue EB label (the other 5 were not recorded per TIC).
- `eb_like > 0.5`: **11 of 35 uncatalogued** (8 periodic in S32, 3 more from other sectors) and
  **10 of 13 catalogued**. 24 of the 35 show a single eclipse in S32, which no single-sector test
  can tell from a symmetric single dip.
- Top 1% (stratum percentile, veto applied, all splits): catalogued EBs 377 -> 66, `eb_like >
  0.5` 538 -> 3, identifiable eyeballed EBs 46 -> 28 of 48. Top 60 overall: eyeballed EBs 38 -> 17.
- **Catalogues do not rescue them:** none of the 35 is in Kostov+2025 (tables 3/4) or in Gaia
  DR3 `veb` within 2″; 26 of 35 are in Kostov's unvetted NN list (a soft flag that also fires
  on dippers). 8 of the 13 catalogued ones are in `veb`.

**Rejected: a flat-bottom test for lone eclipses (E4 "U shape").** The width ratio W90/W50 of
real lone EB eclipses in the train split (median 0.39 for 18 eyeballed, 0.40 for 26
catalogued) equals a Gaussian single dip's (0.38). The trapezoid `null_eb` injections sit at
0.71. Limb darkening, partial eclipses and 30-min bins round real eclipse bottoms. The test would
have looked good on val and done nothing on real stars. It also means `null_eb` is flatter
than real EBs, so val's EB-null numbers are somewhat optimistic for shape.

**Open:** a single-sector lone eclipse remains unexplained without another sector (E7) or an
ephemeris (TESS EBs in later sectors, ground surveys). A lone eclipse plus a shallower,
width-matched second event in the same sector (an eccentric secondary) is now counted as an
"extra" event (c_extra 0.6), which costs 7 of the 24 lone-eclipse EBs. `kc_v2` with the EB
features and the Kostov/`veb` labels (`build_known_class_labels.py --eb-sources kostov
gaia_veb --out labels_eb`) is not trained yet.

## 15. Implementation record: young-host val target `yso1` and ensemble `ens_v3` (S32, 2026-11-10)

Code: `src/oddball/inject/yso.py` (host pool, replicas, matched nulls), `src/oddball/score/ens3.py`
(objective, tie rule, bootstraps), `scripts/{make_yso_set,veto3_inj_set,ensemble_v3,yso_report}.py`,
`--real-masks-only` in `scripts/veto_inj_set.py`, an optional `depth_range` in
`oddball.inject.sampling.draw_params` / `draw_dusty_tail`, `tests/test_{yso,ens3}.py`.
`build_injection_set` was split into helpers (`_write_shards`, `_write_tables`, `_provenance`);
a v1 val rebuild with the refactored code is identical to the stored set (all columns and flux).

**Why.** Every ensemble so far was tuned on val injections into generic hosts, and the tuning
let the ML scores dominate: `ens_v1`, `ens_v1_eb` and `ens_v2` put 2-3 of the 18 S32 known
positives in the top 1%, classical alone 11-12 (§12-§14). The known positives are evaluation
only, so the fix has to come from a val target that looks like them. 16 of the 18 are young
dippers, mostly faint Orion OB1 members on camera 1 CCD 2. Their light curves are dominated by the
star's own large, irregular variability, which generic hosts almost never have.

**The set `yso1`** (`data/injections/yso1/s0032/{val,test}`, ~1 min to build).
- Host pool: S32 stars that are catalogued young. VSX within 21″ with a young-star type
  (`vsx_young`: YSO / T Tauri / Orion-type / UXOR / DIP / FU Ori etc.; RCB, BOY, ZZO excluded) or
  Gaia DR3 within 3″ flagged YSO. 473 stars: train 373, val 31, test 59, known 10.
- Exclusions: every star within 120″ of a known positive (0 in val/test after the split), and the
  29 hosts already used by v1 / eb1 / n2. Final hosts: **val 25, test 36** (val sources vsx 10 /
  gaia 3 / both 12; test 13 / 3 / 20). Camera 1 CCD 2: val 12, test 22.
- Each host gets **20 replicas** (each a fresh injection from `YSO_KINDS`: aperiodic and
  quasi-periodic dippers, Boyajian, exocomet, dusty tail) plus **one matched null** (`null_yso`,
  the unchanged light curve, `replica = -1`). val 500 + 25, test 720 + 36 light curves.
- Depths: log-uniform in `YSO_DEPTH_RANGE` = 0.17-17%, i.e. the v1 range shifted up one decade,
  because young hosts' own variability hides v1's 0.3% events.
- Hosts are noisy like the known positives: median rms percentile in their stratum val 0.97,
  test 0.92 (known positives 0.976).
- RNG stream key [seed, sector, split, 11, inj_index]; `hosts.parquet` records each host's match.

**Pipeline on a replicated set.** A set that repeats each host 21 times breaks two stages that
assume one row per star, and both had to be fixed:
- Cross-star masks (cmv2, cmv3) are built from the set's own light curves by default. Twenty
  copies of a host make every event in it look "common", so `--real-masks-only` builds the
  masks from the real sector only.
- cmv3's cross labels compare an event with other stars at the same time. On yso1 they are
  computed **per replica** (each replica against the real sector), else the 20 replicas of a
  host vouch for each other.
- Checks: the matched nulls veto exactly like their real hosts (cmv2 and cmv3: 0 mismatches);
  `veto3_inj_set.py --check` on v1 and n2 reproduces the stored cmv3 flags (0 mismatches, max
  diff 0). cmv3 flags 1.9% of yso1, 1.4% of eb1.
- Not usable on yso1: the v0 `sys_flag` and `classical_cm`, which build their map from the set.
- Matched nulls are a pipeline check: `score(null copy)` should equal `score(host)`. classical and
  iforest reproduce exactly. **`ssl_v2_recon` does not** (see the caveat below).

**Pre-registered objective** (`ens3.py`, fixed before the sweep):
J = R_gen + R_yso − N_inst − N_eb, all on val, top 1% of stratum after cmv3.
- R_gen: v1 recall averaged over the ~0.3 / 1 / 3% depth bins.
- R_yso: yso1 recall averaged over the ~0.3 / 1 / 3 / 10% bins.
- N_inst: mean FPR of the 5 instrument null types (v1 CCD / dump nulls, n2 asteroid / step /
  column). N_eb: eb1 EB-null FPR. N_yso (matched-null rate) is reported, not scored.
- Guard: dusty-tail recall may not fall more than 0.02 below the same rule at alpha 0.
- Ties within 0.01 of the best J go to the simplest rule (`simplicity`: fewer free weights, then
  lower alpha).

**Grid** (`ensemble_v3.py`, ~2.5 min): rules mean, max, fisher, stouffer, and a weighted max
`wmax` of the stratum percentiles with weights (classical 1, iforest w2, recon w3), w in
{0.25, 0.5, 0.75, 1, 1.25, 1.5, 2}; × EB down-weight alpha in {0, .25, .5, .75, .9, 1}.
The single-transit weight stays 0. The script reproduces `ens_v2` and `ens_v1_eb` from their
own files (max diff 0) and writes their missing yso1 / n2 / eb1 scores.

**Pick: `ens_v3` = wmax 1 : 2 : 0.75 (classical : iforest : recon), alpha 0.9.** Val J 0.402
(best 0.404 at alpha 1.0, within the tie). R_gen 0.522, R_yso 0.145, N_inst 0.031, N_eb 0.234,
N_yso 0.12 (3 of 25 hosts), dusty 0.369.

| family best (val) | rule | J | R_gen | R_yso | N_eb |
|---|---|---|---|---|---|
| mean | alpha 1 | .365 | .361 | .181 | .156 |
| max | alpha 1 | .356 | .556 | .157 | .329 |
| classical-heavy wmax | 1 : 1 : .25, alpha 1 | .394 | .443 | .174 | .194 |
| fisher | best | .326 | | | |
| stouffer | best | .330 | | | |
| `ens_v2` rule | alpha 0 | −.097 | .541 | .098 | .715 |
| `ens_v1_eb` rule | alpha .75 | .355 | .596 | .131 | .342 |

- The highest R_yso on the grid is 0.214 (wmax 1 : .25 : .5), at R_gen 0.259.
- **The yso term did not decide the pick.** Maximising J − R_yso picks the same rule. What moved
  the pick away from `ens_v2` is the EB term (`ens_v2`'s alpha 0 and EB-null FPR 0.72). Under the
  old ens_v2 objective the same grid picks wmax 1 : 1.5 : 1.5, alpha 0.5.
- Paired cluster bootstrap of J(pick) − J(other), host-clustered and stratum-wise, 90%:
  `ens_v2` rule +0.499 [0.423, 0.579]; `ens_v1_eb` rule +0.047 [−0.008, 0.105]; max alpha .75
  +0.051 [−0.015, 0.122]. So on val `ens_v3` is clearly better than `ens_v2` and not clearly
  better than `ens_v1_eb`.
- Val yso1 recall of the pick: 0.148 [0.078, 0.228] (host bootstrap).

**Published dipper lists (step 1b): not used.** The Capistrant+2022 dippers (VarF D, 293 rows) that
fall in S32 are already known positives; Cody & Hillenbrand and Ansdell+2016 cover K2 Upper Sco / ρ Oph, and
S32 has no star within 1° of Upper Sco, Taurus or NGC 2264. val has no VSX DIP / UXOR star (test
has 2 UXOR: TIC 284996069, 287840183). So there is no independent real-dipper val set in S32.

**Test (cmv3, top 1%; nothing tuned on it).**

| scorer | ~0.3% | ~1% | ~3% | all | v1 null | step | asteroid | column | EB null | dusty | known top 1% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| classical | .027 | .097 | .291 | .261 | .013 (4/300) | 4/200 | 0/53 | 1/78 | .164 | .029 | 11/18 |
| ens_v1_eb | .361 | .649 | .828 | .663 | .040 (12/300) | 26/200 | 2/53 | 5/78 | .324 | .379 | 2/18 |
| ens_v2 | .321 | .590 | .794 | .624 | .020 (6/300) | 40/200 | 2/53 | 6/78 | .697 | .310 | 3/18 |
| ens_v3 | .333 | .517 | .762 | .613 | .033 (10/300) | 41/200 | 3/53 | 4/78 | .245 | .362 | 5/18 |

- Known positives: `ens_v3` 5/18 in the top 1% (TIC 50745441, 34397579, 43488669, 248992945,
  38090218). All 17 unvetoed ones sit higher than in `ens_v2` (e.g. PDS 110 0.884 -> 0.936);
  β Pic is vetoed. Classical with cmv3 still gets 11/18 (12/18 with cmv2).
- Costs against `ens_v2`: ~1% recall 0.590 -> 0.517, ~3% 0.794 -> 0.762, v1 null FPR
  0.020 -> 0.033. Gain: EB-null FPR 0.697 -> 0.245, ~0.3% 0.321 -> 0.333, dusty 0.310 -> 0.362.
- Step nulls stay the weakest instrument type (~0.20 FPR for every ensemble).

**Seeds** (`--fixed`, components from `runs/ssl_v2_s{1,2}`), test:

| metric | ens_v3 seeds 0 / 1 / 2 | ens_v2 seeds 0 / 1 / 2 |
|---|---|---|
| known in top 1% | 5 / 8 / 5 | 3 / 4 / 3 |
| ~0.3% recall | .333 / .375 / .286 | .321 / .318 / .261 |
| ~1% recall | .517 / .633 / .498 | .590 / .640 / .588 |
| EB-null FPR | .245 / .256 / .215 | .697 / .736 / .714 |
| v1 null FPR | .033 / .040 / .023 | .020 / .047 / .020 |
| top-100 overlap (pairwise mean) | 0.75 | 0.71 |

The known-positive gain and the EB gain hold for every seed; the ~1% cost is seed-dependent.

**yso1 test** (36 hosts, 720 injections; `yso_report.py`, host-bootstrap 90% CI):

| scorer | recall | matched nulls top 1% | recall ~10% depth |
|---|---|---|---|
| classical | .142 [.099, .188] | 1/36 | .330 |
| ens_v1_eb | .210 [.154, .274] | 0/36 | .462 |
| ens_v2 | .183 | 1/36 | .412 |
| ens_v3 | .158 [.110, .213] | 0/36 | .390 |

- `ens_v3` recall by injected SNR: <3: 0, 3-10: 0, 10-30: 0.073, 30-100: 0.443, >100: 0.836.
- Seeds of `ens_v3` on yso1 test: .158 / .233 / .190 (`ens_v2` .183 / .212 / .182).
- val yso1: classical .206 (3/25 nulls; conditional .139), ens_v1_eb .136, ens_v2 .102,
  ens_v3 .148 (3/25 nulls; conditional .070). **The scorer ordering differs between val and test**,
  so yso1 with 25-36 hosts is host-dominated noise at the level of these differences.
- yso1 recall does not track the known positives: classical has the lowest yso1 test recall and
  the most known positives. The set does not yet measure what makes the known positives score.
- EB tests on young hosts: `eb_like > 0.5` for 1 of 1,220 yso1 injections and 0 of 61 hosts.
  The EB down-weight does not hit young-host dippers.

**Gate (known positives up without losing recall or null FPR elsewhere): FAIL.** Known positives
improve (5/18 vs 3/18, every seed), and EB nulls drop by a factor of ~3, but ~1% recall falls by
0.07 and v1 null FPR rises to 0.033, and the yso term did not cause the change. `ens_v3` is a
better default than `ens_v2` on the EB and known-positive axes; it is not the young-dipper fix.

**Caveats found.**
- **`ssl_v2_recon` is noisy per row (fixed in §16).** Scoring masks came from one RNG stream in row order
  (`oddball/ml/scoring.py`, `rng = default_rng(seed)` per run), so the same light curve at a
  different row gets a different `score_recon`. Null copies vs hosts: raw diff mean −0.105, SD
  0.707 (n 61), against a real-star IQR of 0.219; final percentile median |Δ| 0.022, max 0.495
  (TIC 24297713). It looks unbiased but limits stability. Fix: seed masks per TIC or average more
  masks.
- **R_yso as defined is dominated by hosts already in the top 1%**: all 20 replicas of such a host
  count as recovered. Conditional recall (hosts whose null is below the cut) should be the
  pre-registered term next time.
- **Too few hosts.** 25 val hosts give a recall CI ±0.075. More need other sectors (young hosts in
  S33/38/43/44/55).
- **Too shallow.** ≤17% is shallower than real dippers (literature 10-50%).
- A `yso2` (conditional term, deeper, more hosts) was **not** built here: it would be motivated by
  these test results. Build it as the next pre-registered step, on new sectors' val splits.

**Costs:** yso1 build ~1 min; classical seconds; ML scoring ~1 min per seed; EB features 20 s;
cmv3 on yso1 16 min (per-replica cross labels); `ensemble_v3.py` ~2.5 min.

**Commands.**
```
uv run --group ml python -I scripts/make_yso_set.py --sector 32 --name yso1 --exclude-sets v1 eb1 n2 --replicas 20
uv run --group ml python -I scripts/score_classical.py --sector 32 --reuse-real --inj eb1 n2 v1 yso1
uv run --group ml python -I scripts/ml_score_inj.py --run runs/ssl_v2_s0 --shards data/injections/yso1/s0032 \
    --variant iforest=ssl_v2_iforest --variant recon=ssl_v2_recon
#   seeds 1, 2 on n2 / eb1 / yso1: --run runs/ssl_v2_s{s} --ref-scorer ssl_v2_seed{s}_iforest
uv run --group ml python -I scripts/eb_features.py --sector 32 --skip-real --inj yso1 n2 --workers 4
uv run --group ml python -I scripts/eb_features.py --xsec 32
uv run --group ml python -I scripts/veto_inj_set.py --sector 32 --inj yso1 --real-masks-only   # also --inj n2
uv run --group ml python -I scripts/veto3_inj_set.py --sector 32 --inj eb1 yso1 --real-masks-only yso1
uv run --group ml python -I scripts/veto3_inj_set.py --sector 32 --inj v1 n2 --real-masks-only n2 --check
uv run --group ml python -I scripts/ensemble_v3.py --sector 32 --name ens_v3
uv run --group ml python -I scripts/ensemble_v3.py --sector 32 --name ens_v3_seed1 --fixed ens_v3 \
    --components classical ssl_v2_seed1_iforest ssl_v2_seed1_recon   # same for ens_v2, seed 2
uv run --group ml python -I scripts/evaluate.py --compare classical ens_v1_eb ens_v2 ens_v3 --split test --veto cmv3 --inj v1   # and eb1, n2, yso1
uv run --group ml python -I scripts/yso_report.py --scorers classical ens_v1_eb ens_v2 ens_v3 --split test --veto cmv3   # and --split val
```
Reports: `outputs/eval/compare/s0032/classical__ens_v1_eb__ens_v2__ens_v3__veto-cmv3[__inj-<set>]/report.md`,
`outputs/eval/yso1/s0032/<scorers>__<split>__veto-cmv3/`. `evaluate.py --compare` now suffixes
`__inj-<name>` for non-v1 sets (before, every set overwrote the v1 report).

## 16. Implementation record: `ssl_v2_recon` masks seeded per TIC (S32, 2026-11-10)

Code: `oddball.ml.scoring.encode_and_surprise(..., mask_seeding="tic")` and `mask_rng`,
`scripts/ml_score.py` / `ml_score_inj.py --mask-seeding`, `tests/test_ml_scoring_order.py`.

**Bug (§15 caveat).** The reconstruction surprise hides each light curve's patches in 5 block
masks (every patch hidden in 2). The masks came from one `default_rng(seed)` consumed in row
order, so a light curve's score depended on its row and on the rows scored before it. The yso1
matched nulls (unchanged host copies) exposed it: percentile median |Δ| 0.022, max 0.495.

**Fix.** Each light curve draws its masks from `default_rng([seed, host_tic])` (`SeedSequence`
hashes the pair, so consecutive TICs get unrelated streams). For real stars `host_tic = tic`; an
injected light curve gets exactly its host's masks, so host and injection differ only by the
injected signal (a paired design). `encode_and_surprise` now also returns rows in the order
given: before, it relied on `iter_batches` returning sorted rows, which only held for
`rows = arange(n)`. The old behaviour stays available as `mask_seeding="stream"` for
reproduction. Encodings and surprise files made with the new masks carry a `_pt` suffix, so a
cached old encoding is never reused. The IsolationForest state is now per sector
(`ref_state_sNNNN.pkl`), so scoring another sector no longer overwrites S32's.

**Checks.**
- Unit test: scores are equal (1e-5) under a row permutation with a different batch
  composition; a duplicated light curve with the same host scores like the original; the old
  stream mode does not (documents the bug).
- yso1 matched nulls (61 hosts): raw recon |Δ| max 4.8e-7 (stream: 2.6), percentile |Δ| max
  3.5e-4 (stream: 0.495). iforest and kNN are bit-identical to the stored files (the pooled
  embedding uses no masks).
- **Mask sensitivity remains.** New vs old masks move real S32 recon percentiles by median 0.022
  (p90 0.11; 12% of stars by more than 0.1, 2% by more than 0.3); the recon top-1% overlap is 0.79.
  The fix makes the score deterministic, not mask-independent. Averaging more masks would reduce
  this; not done.

**Rebuilt on S32** (`ml_score.py --rescore-real` with v1, eb1, n2, yso1; 6 min, peak RSS 2.8 GB;
then `ensemble_v3.py --name ens_v3 --fixed ens_v3_rowrng`, where `ens_v3_rowrng` is the copy of
the old files). `ens_v2` and `ens_v1_eb` were not rebuilt (they keep the old recon), so
`ensemble_v3.py` skips their reproduction check under `--fixed` / `--no-ref-check`.

Test, cmv3 veto, top 1% (`evaluate.py --compare classical ens_v3_rowrng ens_v3 --split test
--veto cmv3 --inj <set>`):

| ens_v3 | ~0.3% | ~1% | ~3% | all | v1 null FPR | known top 1% | EB-null FPR | n2 null FPR | yso1 recall |
|---|---|---|---|---|---|---|---|---|---|
| old masks | 0.333 | 0.517 | 0.762 | 0.613 | 0.033 | 5/18 | 0.245 | 0.145 | 0.158 |
| per-TIC masks | 0.334 | 0.510 | 0.728 | 0.604 | 0.033 | 5/18 | 0.243 | 0.145 | 0.157 |

- The ~3% bin loses 0.034 (about 2 SE at ~600 injections): mostly exocomet, single-dip and
  outburst injections where recon is the deciding component. This is mask-draw variance, not a
  bias we can name; it is reported, not tuned away. Known-positive percentiles are unchanged to
  0.002, and the real `ens_v3` top 100 / 300 / 1600 are the same stars as before.
- Val J of the frozen rule: 0.3997 (was 0.402).
- **Diagnostic only, not adopted:** a fresh val sweep with the fixed recon
  (`ensemble_v3.py --name ens_v3_resweep_diag --no-ref-check`) has its best J at the same
  weights (1 : 2 : 0.75, alpha 1.0, J 0.4016), but the tie rule (simplest within 0.01) now picks
  wmax 1 : 1 : 0.5, alpha 0.9 (J 0.394). The pick is unstable inside the tie band. `ens_v3`
  stays frozen, since the search on the other sectors uses it.
