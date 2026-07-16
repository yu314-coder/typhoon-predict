# ChatGPT (GPT-5, High reasoning) consultation — TrackFormer negative transfer

Question: how to let intensity features help without hurting track (720->737 km WP-2020+ after
adding motion-dynamics features; upweighting track made it worse).

## Diagnosis
**Negative transfer**, not insufficient track weighting. The model spends scarce shared capacity
fitting easier thermodynamic structure, slightly degrading the kinematic representation. Raising
the track coefficient only makes shared optimization more unstable (gradient conflict — a known
failure mode of naive multitask scalarization).

## Recommended architecture: asymmetric, protected sharing
- Kinematic encoder inputs: east/north motion, heading sin/cos, speed, turn-rate, latitude, season, validity.
- Thermodynamic encoder inputs: vmax, pressure, RMW, radii, trends, validity.
- Track decoder receives primarily the kinematic encoder.
- Intensity decoder consumes **stopgrad(H_kin)** so motion helps intensity but intensity grads can't corrupt kinematics.
- Thermo->track only through a small adapter:  H_track = H_kin + alpha_l * A(H_thermo),
  where each lead gate alpha_l is **zero-initialized and trained only with track loss**.
  => adding intensity tasks initially reproduces the track-only model exactly.
- Gradient routing: kinematic encoder = track grads only; thermo encoder = intensity/radius grads only;
  kinematic->intensity = detached; thermo->track = gated adapter trained only by track.
- If layers remain shared: log per-layer cosine similarity of track vs thermo grads; apply PCGrad/CAGrad
  only to genuinely shared params.
- Config: kinematic enc 4 layers d=256 h=8; thermo enc 3 layers d=256; track dec 4; intensity/radii dec 4-6;
  adapters bottleneck 32-64 zero-init; do NOT share output-query embeddings across tasks.
- Training sequence: (1) train track-only; (2) init joint from it, freeze kinematic enc + track dec;
  (3) train thermo branch; (4) unfreeze only track adapters + final 1-2 kinematic layers at 0.1x LR;
  (5) projected gradients on partially shared params.

## Idea critiques
1. Persistence + recurvature residual: GOOD, with changes. Don't predict 20 free position residuals.
   Baseline: p_hat_l = p0 + dt * sum_{j<=l} rho_j v0 (velocity-damping schedule rho). Predict residual in
   the initial-motion Frenet frame (along-track & cross-track acceleration), integrate accel->vel->pos.
   Better: predict 6-8 spline/DCT coeffs per axis (smooth recurvature, lower effective dim).
   Add auxiliary heads (recurvature within 72/120h, left/right turning, slowdown/stall, ET proximity)
   as representation supervision only — do not hard-condition final track on a predicted class.
2. Separate streams: correct, but separate decoders alone are insufficient if the encoder stays fully
   shared. The key is **gradient routing** (above).
3. Full 340-dim covariance + RMT: interesting but NOT the track-error fix. It improves joint calibration/
   trajectory coherence, not the conditional mean. Full covariance creates another thermo->track pathway;
   Gaussian NLL can cut loss by inflating variance instead of improving the mean; classical MP assumptions
   are poorly matched to overlapping windows / masks / lead heteroscedasticity / basin mixtures.
   => Use **block-structured** covariance:  Sigma = blkdiag(Sigma_track, Sigma_thermo).
   Track: 40-dim (20 leads x 2 axes), rank 4-8 + diagonal, or matrix-normal Sigma_track = K_lead (x) K_axis + diag(s^2),
   K_lead a learned temporal kernel, K_axis 2x2. Thermo: separate blocks for intensity/pressure/RMW and radii,
   rank 8-16 each. Train covariance heads AFTER freezing/converging the mean heads. Prefer multivariate CRPS/
   energy score, or clip log-scales + mix NLL with Huber (Gaussian log-score is outlier-sensitive).

## Track loss
L_track = sum_l w_l [ d_gc(p_hat_l, p_l) + lambda_perp |eps_perp,l| + lambda_v ||dp_hat_l - dp_l||_1 ]
          + lambda_a sum_l ||second_diff p_hat_l||_Huber
- Great-circle / tangent-plane position loss; explicit along- & cross-track errors.
- Mild lead weighting w_l = sqrt(l), normalized to mean 1.
- Do NOT let predicted track variance auto-downweight this deterministic loss.
- Normalize motion residuals by lead time, not by the thermodynamic-variable scaler.

## Highest-value ablation order
1. Paired storm-level bootstrap the 720 vs 737 difference (resample storms, not windows) — is it even real?
2. Track-only checkpoint vs current multitask.
3. Stop all thermo grads entering the kinematic encoder.
4. Add zero-init thermo->track adapters.
5. Persistence + Frenet-frame acceleration residual.
6. Recurvature auxiliary labels.
7. Environmental steering features (ERA5 deep-layer winds) — likely the real track floor-breaker.
8. Only then add blockwise structured covariance.

Bet: protected kinematic encoder + environmental steering branch + persistence-residual track head,
with thermodynamic features contributing through zero-init gated adapters.
