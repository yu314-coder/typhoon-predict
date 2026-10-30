# v1.2.73: controlled moving-core improvement

Baseline: v168 E3, SHA256
99ed776097e05ad6df03375465d9618741bed63a4ee57724d3ba98e295039d30.
The baseline is frozen as an archived reference, not frozen during v173 training.

The 21,452,595-parameter model transfers every baseline parameter. A two-scale
environment encoder pools physical state, recurrent memory and geographic
features. Two transformer blocks exchange regional context; cross attention
returns a correction to the environmental recurrent state. A zero-initialized
output makes the initial forecast equal the baseline. The moving core continues
to evolve pressure through relative transport and learned tendencies.

Both objectives matter: spatial pressure evolution and derived storm numbers.
The existing area-weighted atmospheric, outer-core pressure/gradient, track,
central-pressure and auxiliary wind losses remain. New masked losses supervise
six-hour pressure change and displacement against corresponding label changes.
Future fields and best tracks enter only the objective. Validation always uses
120 hours; training horizons are 24, 48, 72 then 120 hours. Best selection uses
validation, never Tip or the displayed benchmark storms.

The route is a pressure association within the moving core; central pressure is
sampled from that same field at that location. Maximum wind remains an auxiliary
scalar estimate: MSLP alone does not uniquely determine surface wind. The fixed
regional rendering is a resampled composite; the core is authoritative for the
extracted central pressure. Display +0 from the initialized core, and distinguish
it from raw input analysis. Do not connect raw +0 to conditioned +6.

Deliberately retained limitations: the v168 initialization includes a 100 km
Gaussian correction using issue-time observed pressure. It is a reconstruction,
not resolved truth. Fine inner geometry is underdetermined with only 800 native
training windows. Replacing the initializer, tracker, transport and backbone all
at once would destroy the controlled baseline. Learned initialization and a
probabilistic ensemble are later separate ablations, not implemented claims.

Only the verified full existing corpus is accepted: 13949/1041/1195 windows and
800/100/100 native windows. No new weather downloads. Tip is excluded, and its
repeatedly inspected forecasts are development diagnostics. Region is Western
Pacific, not a global weather model. Report native/coarse cases separately and
validate on untouched storms before calling this an improvement.

Mac Metal tests passed: baseline equivalence before optimization, coarse/native
four-step finite backward with attention gradients, field-to-pressure equality,
120h finite rollout, label invariance, rejected future inputs, missing intensity,
optimizer resume and full sampler coverage. CUDA/T4 and skill remain unverified.

Training command after the existing notebook's setup cell:
`python v173_pressure_train.py 10 3 1 --microbatch 4`
The final positional flag is accepted for compatibility; dynamics use FP32.
Increase physical microbatch only after measuring throughput and memory. Effective
batch alone does not increase GPU memory. Checkpoints and train.log persist to
MyDrive/typhoon_predict/training/v173; restarting resumes last.pt.
