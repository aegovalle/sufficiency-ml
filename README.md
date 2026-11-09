# sufficiency-ml: Sufficiency-Aware Representation Learning for Microstructure-to-Failure Prediction

Companion code, evaluation pipeline, and tabular/trajectory verification artifacts for the manuscript:

> **A.E. Gómez-Ovalle and C. Ramírez Ovalle (2026).**
> *Sufficiency-aware representation learning for microstructure-to-failure prediction: Certified Lipschitz barriers, cross-fitted McShane regularization, and transfer to phase-field fracture.*
> Submitted to *Computer Methods in Applied Mechanics and Engineering*.

This repository builds upon the frozen phase-field fracture validation benchmark archived in our companion descriptor study at [`https://github.com/aegovalle/energy-failure-descriptors`](https://github.com/aegovalle/energy-failure-descriptors).

---

## Overview

Surrogate models for microstructure–property relations are typically evaluated by average prediction error ($R^2$ or RMSE), conflating three distinct levels of representation adequacy for an encoder $z = E_\theta(x)$:
1. **Level 1 — Functional / informational sufficiency:** whether $R(x)$ is determined by $z = E_\theta(x)$ (or lower-bounded on symmetry orbits by $b_G(x) = \frac{1}{2}\sup_{g,h\in G}|R(g\cdot x)-R(h\cdot x)|$).
2. **Level 2 — Metric regularity ($\Omega_\Psi(q)$, $e^\star_\theta(\Lambda)$):** whether distinct mechanical responses are separated at slope budget $\Lambda$ under a normalized latent distance $d_\Psi(z_i, z_j)$, governed by the exact finite-sample Lipschitz barrier
   $$e^\star_\theta(\Lambda) = \frac{1}{2}\max_{i,j}\bigl[|R_i - R_j| - \Lambda\,d_\Psi(z_i, z_j)\bigr]_+.$$
3. **Level 3 — Readout / interpolant accuracy:** whether a specific prediction head $g_\phi(z)$ or the central McShane–Whitney midpoint extension $M_A(z) = \frac{1}{2}[U_A(z) + L_A(z)]$ accurately approximates $R$ out of sample.

To regularize worst-case latent neighborhoods during neural training, we introduce the **Cross-Fitted McShane Barrier Regularizer (`SBR-CF`)** combined with cross-batch anchor memory, hard-query mining, periodic-translation and dihedral $D_2$ invariance (`E-circ`, `E-FNO`), generalized-mean ($\operatorname{GeM}_p$) pooling, and a spectrally normalized $\Lambda$-Lipschitz head.

---

## Repository Structure

```text
sufficiency-ml/
├── configs/
│   └── data_elastic.yaml                      # Elastic corpus generation parameters
├── data/
│   ├── stage0/
│   │   ├── design_cells.csv                   # 216 design cells (elastic + AT1 phase-field targets)
│   │   └── stage0_report.json                 # Stage 0 reproduction report
│   ├── frozen/
│   │   ├── blind_pf.csv                       # Frozen AT1 phase-field Set B (45 cells) & Set O (12 cells)
│   │   ├── tcd_pf.csv                         # Frozen AT1 phase-field Set C1 (30 cells) & Set C2 (12 cells)
│   │   └── frozen_hashes.json                 # Canonical D2/periodic-translation SHA-256 exclusion hashes
│   └── processed/
│       ├── splits.json                        # Exact train/val/test/test2/ood_morphology split indices
│       ├── elastic_pilot.csv                  # 3,300-cell elastic corpus targets and physical descriptors
│       └── ood_physics_and_rotation.npz       # OOD contrast, biaxial, and 90-deg rotation targets
├── src/suffml/
│   ├── metrics/
│   │   ├── metric.py                          # d_white, d_reg, d_trunc, d_std, d_N, effective_rank
│   │   ├── sufficiency.py                     # Omega(q), Omega/Omega^null
│   │   ├── lipschitz_barrier.py               # Exact barrier e*(Lambda), McShane envelopes U_A, L_A, M_A, e^cf
│   │   ├── permutation_null.py                # Global response-permutation null
│   │   ├── conditional_test.py                # Local-swap k-NN conditional permutation tests
│   │   └── counterexamples.py                 # Consensus worst-case collision pair extraction
│   ├── models/
│   │   ├── encoder_circ.py                    # Circular-padded E-circ, E-std, and spectral E-FNO backbones
│   │   ├── pooling.py                         # GAP and learnable GeM_p pooling
│   │   ├── normalization.py                   # LatentNorm (frozen running-moment standardization)
│   │   ├── lipschitz_head.py                  # SpectralLinear and certified Lambda-Lipschitz head
│   │   └── autoencoder.py                     # ConvAutoencoder and high-fidelity residual ConvAutoencoderHiFi
│   ├── losses/
│   │   ├── crossfit_barrier.py                # Cross-fitted McShane barrier loss (SBR-CF) and SBR-in
│   │   └── continuous_label.py                # Rank-N-Contrast (RNC) and RankSim continuous losses
│   ├── train/
│   │   ├── train_gate.py                      # Stage 0 / Gate training routines
│   │   └── train_objectives.py                # Full campaign training (MSE, Tail-MSE, RNC, RankSim, SBR-in, SBR-CF)
│   ├── physics/
│   │   ├── fft_rve.py                         # Moulinec-Suquet periodic Lippmann-Schwinger FFT elasticity solver
│   │   ├── fft_targets.py                     # Homogenized energy R_E and initiation hazard R_H(beta)
│   │   ├── critical_distance.py               # Process-zone functional J_r and baseline descriptors
│   │   ├── descr_tools.py                     # Geometric ligament and inclusion descriptors
│   │   └── pf_fft.py                          # Staggered FFT AT1 phase-field fracture solver & min_path
│   └── analysis/
│       ├── evaluate.py                        # Multi-metric evaluation protocol
│       ├── subsample.py                       # Paired half-sample split bootstrap (B=200)
│       └── certify.py                         # Certified head excess gap Delta_head evaluation
├── scripts/
│   ├── reproduce_stage0.py                    # Stage 0 reproduction and frozen-geometry hash registry
│   ├── build_dataset.py                       # Deterministic generation of the 3,300-cell elastic corpus
│   ├── run_baselines.py                       # Experiment A: Physical descriptor baselines
│   ├── run_gate.py                            # Experiments B & C: Autoencoders and initial supervised gate
│   ├── run_experiments_de.py                  # Experiments D & E: Lambda sweep and latent dimension sweep
│   ├── run_consolidation_train.py             # Multi-seed training campaign
│   ├── run_consolidation_eval.py              # Multi-seed evaluation and paired subsampling
│   ├── run_phase1_counterexamples_abl7_transfer.py # Counterexample resolution & cross-horizon transfer (Exp. G1)
│   ├── run_phase2_pf_transfer_and_cond_test.py     # AT1 phase-field fracture transfer & conditional tests (Exp. H)
│   ├── run_phase3_ablations.py                # Experiment F: Memory, mining, pooling, E-FNO, and symmetry ablations
│   ├── run_phase4_ood_physics_and_rotation.py # Experiment G2: Contrast, biaxial, and 90-deg rotation transfer
│   ├── run_phase5a_ml_and_stats_controls.py   # Tail-MSE, ConvAE-HiFi, whitening sweep, 254x254 FFT, power curve
│   ├── run_phase5b_pf_strip_shifts.py         # AT1 strip translation sensitivity & empirical barrier b_shift
│   ├── make_paper_figures.py                  # Generates Figures 1-5 (.pdf and .png)
│   └── build_manuscript_v2.py                 # Generates Manuscript_v2.2_Sufficiency_Aware_Representation_Learning.tex
├── results/
│   ├── *.csv, *.json                          # All precomputed summary tables and statistical test outputs
│   ├── pf_shifts/*.json                       # All 30 shifted AT1 phase-field solver trajectories
│   └── models/*.npz, *.json                   # Saved latent embeddings and run metadata across all seeds
└── tests/
    └── test_*.py                              # Unit tests for barriers, McShane extension, invariance, and hashes
```

---

## Installation

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -e ".[test]"
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu128
```

*(Tested on Windows 11, Python 3.12, PyTorch 2.11+cu128.)*

---

## Reproducing the Verification Suite, Figures, and Manuscript

1. **Run the unit test suite (barrier identity, McShane bounds, $D_2$ & periodic translation invariance, canonical hash exclusion):**
   ```powershell
   .venv\Scripts\python -m pytest -q
   ```
2. **Regenerate all manuscript figures (`results/fig_*.pdf` and `results/fig_*.png`):**
   ```powershell
   .venv\Scripts\python scripts\make_paper_figures.py
   ```
3. **Regenerate the LaTeX manuscript source:**
   ```powershell
   .venv\Scripts\python scripts\build_manuscript_v2.py
   ```
4. **Re-run individual evaluation or verification scripts from precomputed embeddings / solvers:**
   ```powershell
   .venv\Scripts\python scripts\run_phase1_counterexamples_abl7_transfer.py
   .venv\Scripts\python scripts\run_phase2_pf_transfer_and_cond_test.py
   .venv\Scripts\python scripts\run_phase5a_ml_and_stats_controls.py
   .venv\Scripts\python scripts\run_phase5b_pf_strip_shifts.py
   ```
5. **Regenerate the full 3,300-cell elastic corpus from scratch (optional, deterministic `seed = 28`):**
   ```powershell
   .venv\Scripts\python scripts\build_dataset.py 28
   .venv\Scripts\python scripts\extend_test.py
   ```

---

## License & Citation

Released under the [MIT License](LICENSE). See [`CITATION.cff`](CITATION.cff) for citation metadata.
