"""Build the complete CMAME submission package and GitHub repository."""
import hashlib
import pathlib
import shutil
import subprocess
import zipfile

ROOT = pathlib.Path(r"c:\Users\aegov\Downloads\sufficiency-ml")
DOWNLOADS = pathlib.Path(r"c:\Users\aegov\Downloads")
ARTIFACTS = pathlib.Path(r"C:\Users\aegov\.gemini\antigravity\brain\dec3b864-fbf8-451a-877d-8f3d1c473cac")
SUB_DIR = DOWNLOADS / "CMAME_Submission_Package"


def build_referee_zip() -> tuple[pathlib.Path, str]:
    zip_paths = [
        DOWNLOADS / "Sufficiency_ML_Referee_Artifacts.zip",
        ARTIFACTS / "Sufficiency_ML_Referee_Artifacts.zip",
    ]
    include_dirs = ["src", "scripts", "results", "data/frozen", "tests", "configs"]
    include_files = [
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "pyproject.toml",
        "paper/Manuscript_v2.2_Sufficiency_Aware_Representation_Learning.tex",
        "paper/Manuscript_v2.2_Sufficiency_Aware_Representation_Learning.pdf",
        "paper/references_v2.bib",
        "data/stage0/design_cells.csv",
        "data/stage0/stage0_report.json",
        "data/processed/splits.json",
    ]
    for zp in zip_paths:
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in include_files:
                p = ROOT / f
                if p.exists():
                    zf.write(p, f)
            for d in include_dirs:
                dp = ROOT / d
                if dp.exists():
                    for fp in sorted(dp.rglob("*")):
                        if (
                            fp.is_file()
                            and "__pycache__" not in fp.parts
                            and fp.suffix not in {".pt", ".pth", ".npz"}
                        ):
                            zf.write(fp, fp.relative_to(ROOT).as_posix())
    sha = hashlib.sha256(zip_paths[0].read_bytes()).hexdigest()
    return zip_paths[0], sha


def main() -> None:
    if SUB_DIR.exists():
        shutil.rmtree(SUB_DIR)

    d_ms = SUB_DIR / "01_manuscript"
    d_fig = SUB_DIR / "01b_figures_numbered_and_png"
    d_sup = SUB_DIR / "02_supplementary_material"
    d_hi = SUB_DIR / "03_highlights"
    d_cov = SUB_DIR / "04_cover_letter"
    d_dec = SUB_DIR / "05_declarations"

    for d in [d_ms, d_fig, d_sup, d_hi, d_cov, d_dec]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Manuscript & exact figure filenames required by \includegraphics
    ms_tex_src = ROOT / "paper" / "Manuscript_v2.2_Sufficiency_Aware_Representation_Learning.tex"
    ms_pdf_src = ROOT / "paper" / "Manuscript_v2.2_Sufficiency_Aware_Representation_Learning.pdf"
    shutil.copy2(ms_tex_src, d_ms / "manuscript.tex")
    shutil.copy2(ms_pdf_src, d_ms / "manuscript.pdf")

    fig_map = [
        ("Fig1", "fig_counterexample_resolution", "Figure 1: Resolution of consensus worst-case MSE collision pairs by SBR-CF on IID test2 (n = 1,000, d = 8, d_white)."),
        ("Fig2", "fig_lambda_and_dim_sweep", "Figure 2: Effect of cross-fitted barrier weight lambda_cf and latent dimension d in {4, 8, 16} on worst-case insufficiency Omega/Omega^null on R_H(beta = 20)."),
        ("Fig3", "fig_ablation7_metrics", "Figure 3: Metric and whitening sensitivity on test2 (n = 1,000, d = 8): Omega/Omega^null and McShane R^2 across five latent metric normalizations and whitening ridge sweep."),
        ("Fig4", "fig_phase3_ablations", "Figure 4: Component ablations (d = 8, 3-seed medians and min/max ranges on test2 and ood_morphology): memory, mining, pooling, E-FNO backbone, and E-std symmetry breakdown."),
        ("Fig5", "fig_transfer_and_pf", "Figure 5: Cross-horizon transfer matrix across R_E and R_H(beta) and prospective AT1 phase-field peak-strength R_sigma transfer on high-multiplicity Set C2."),
    ]

    for num_prefix, stem, _caption in fig_map:
        pdf_src = ROOT / "results" / f"{stem}.pdf"
        png_src = ROOT / "results" / f"{stem}.png"
        # Copy exact filename into 01_manuscript so Editorial Manager finds it from manuscript.tex
        shutil.copy2(pdf_src, d_ms / f"{stem}.pdf")
        # Copy numbered and named PDF + PNG into 01b_figures_numbered_and_png
        shutil.copy2(pdf_src, d_fig / f"{num_prefix}_{stem[4:]}.pdf")
        shutil.copy2(png_src, d_fig / f"{num_prefix}_{stem[4:]}.png")

    # 2. Supplementary Material & Related Submitted Manuscript
    zip_path, zip_sha = build_referee_zip()
    shutil.copy2(zip_path, d_sup / "Sufficiency_ML_Referee_Artifacts.zip")
    related_pdf = DOWNLOADS / "Draft_v05_descriptor_discovery" / "Draft_v09_Energy_Failure_Descriptors.pdf"
    if related_pdf.exists():
        shutil.copy2(
            related_pdf,
            d_sup / "Related_Submitted_Manuscript_GomezOvalle2026_For_Review_Only.pdf",
        )

    # 3. Highlights (strictly <= 85 characters per bullet including spaces)
    bullets = [
        "Separates functional sufficiency, metric regularity, and readout accuracy",
        "Links exact finite-sample Lipschitz barriers to cross-fitted McShane training",
        "SBR-CF cuts worst-case latent collisions by 30% and mitigates d=16 dilution",
        "Compares RNC 1D compression and multi-D SBR-CF across five latent metrics",
        "Frozen SBR-CF embeddings lift prospective fracture peak-strength R2 to 0.939",
    ]
    for b in bullets:
        assert len(b) <= 85, f"Highlight exceeds 85 chars ({len(b)}): {b}"

    highlights_txt = "Highlights\n\n" + "\n".join(f"- {b}" for b in bullets) + "\n"
    (d_hi / "highlights.txt").write_text(highlights_txt, encoding="utf-8")

    highlights_tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=2.5cm]{geometry}
\pagestyle{empty}
\begin{document}
\section*{Highlights}
\noindent\textbf{Manuscript:} Sufficiency-aware representation learning for microstructure-to-failure prediction: Certified Lipschitz barriers, cross-fitted McShane regularization, and transfer to phase-field fracture\\
\textbf{Authors:} A.E.~G\'omez-Ovalle and C.~Ram\'irez Ovalle

\bigskip
\begin{itemize}
    \item Separates functional sufficiency, metric regularity, and readout accuracy
    \item Links exact finite-sample Lipschitz barriers to cross-fitted McShane training
    \item SBR-CF cuts worst-case latent collisions by 30\% and mitigates $d=16$ dilution
    \item Compares RNC 1D compression and multi-D SBR-CF across five latent metrics
    \item Frozen SBR-CF embeddings lift prospective fracture peak-strength $R^2$ to $0.939$
\end{itemize}
\end{document}
"""
    (d_hi / "highlights.tex").write_text(highlights_tex, encoding="utf-8")
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "highlights.tex"],
        cwd=d_hi,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    for ext in [".aux", ".log"]:
        p = d_hi / f"highlights{ext}"
        if p.exists():
            p.unlink()

    # 4. Cover Letter (using the exact text provided by the user)
    cover_letter_txt = """Dear Editors,

Please consider our manuscript, "Sufficiency-aware representation learning for microstructure-to-failure prediction: Certified Lipschitz barriers, cross-fitted McShane regularization, and transfer to phase-field fracture," by A.E. Gómez-Ovalle and C. Ramírez Ovalle, for publication as a research article in Computer Methods in Applied Mechanics and Engineering.

Reliable surrogates for mechanical failure require representations that preserve distinctions between mechanically different microstructures. Good average prediction accuracy can conceal local ambiguities in latent space, particularly for responses governed by narrow ligaments and stress concentrations. This manuscript develops a framework that separates functional sufficiency, latent metric regularity, and prediction accuracy, and uses these distinctions to guide the evaluation and training of neural representations.

Building on classical Lipschitz extension theory, we connect an exact finite-sample error barrier, defined for a prescribed latent metric and Lipschitz budget, to a cross-fitted McShane regularizer for training microstructure encoders. This links an interpretable worst-case diagnostic to a computable training objective. Controlled comparisons across metric normalizations, latent dimensions, and regression and contrastive objectives identify representation deficiencies that conventional prediction or reconstruction scores can overlook.

We evaluate the methodology on a corpus of 3,300 periodic two-phase elastic cells and a frozen AT1 phase-field fracture benchmark kept separate from encoder training. On a 12-cell fracture test set with higher inclusion counts, augmenting an elastic-field-based critical-distance descriptor with frozen elastic embeddings increases the median peak-strength R-squared from 0.679 to 0.939, compared with 0.922 using embeddings trained by mean squared error. The resulting mean relative peak-stress error is 0.94%. The study also identifies the dependence of performance on the evaluation metric and delineates the limits of transfer to a post-critical crack-path observable.

The physical descriptors, initial barrier-based diagnostic, and fracture benchmark originate in our related submitted manuscript, "Energy and failure select different descriptors: counterexample-driven descriptor discovery in heterogeneous solids," which is cited in the present paper. The present submission develops the neural representation-learning methodology, cross-fitted regularization, and comparative assessment of learned embeddings across mechanical responses. This relationship and the reuse of the frozen benchmark are explicitly described in the manuscript. The source code, evaluation scripts, and verification artifacts are openly available at https://github.com/aegovalle/sufficiency-ml and included in the supplementary material.

We believe the combination of mathematical analysis, a concrete learning method, and mechanical validation makes this work well suited to CMAME. Its contribution should be of interest to researchers developing and assessing machine-learning surrogates for heterogeneous materials, damage initiation, and fracture.

This manuscript has not been published previously and is not under consideration by another journal. Both authors have approved the manuscript and its submission to Computer Methods in Applied Mechanics and Engineering, and declare no competing interests.

Thank you for considering our manuscript.

Yours sincerely,

Álvaro Enrique Gómez-Ovalle
Corresponding author, on behalf of both authors
Escuela Politécnica Superior
University of Burgos
09006 Burgos, Spain
Email: aegovalle@ubu.es
"""
    (d_cov / "cover_letter.txt").write_text(cover_letter_txt, encoding="utf-8")

    cover_letter_tex = r"""\documentclass[11pt,a4paper]{letter}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage[margin=2.4cm]{geometry}
\usepackage[hidelinks]{hyperref}

\setlength{\parskip}{5pt}
\setlength{\parindent}{0pt}

\signature{\textbf{\'Alvaro Enrique G\'omez-Ovalle}\\
Corresponding author, on behalf of both authors\\
Escuela Polit\'ecnica Superior, University of Burgos\\
09006 Burgos, Spain\\
Email: \href{mailto:aegovalle@ubu.es}{\texttt{aegovalle@ubu.es}}}

\address{Escuela Polit\'ecnica Superior\\
University of Burgos\\
09006 Burgos, Spain\\
\texttt{aegovalle@ubu.es}}

\date{\today}

\begin{document}\sloppy
\begin{letter}{The Editors\\
\textit{Computer Methods in Applied Mechanics and Engineering}\\
Elsevier}

\opening{Dear Editors,}

Please consider our manuscript, ``Sufficiency-aware representation learning for microstructure-to-failure prediction: Certified Lipschitz barriers, cross-fitted McShane regularization, and transfer to phase-field fracture,'' by A.E.~G\'omez-Ovalle and C.~Ram\'irez Ovalle, for publication as a research article in \textit{Computer Methods in Applied Mechanics and Engineering}.

Reliable surrogates for mechanical failure require representations that preserve distinctions between mechanically different microstructures. Good average prediction accuracy can conceal local ambiguities in latent space, particularly for responses governed by narrow ligaments and stress concentrations. This manuscript develops a framework that separates functional sufficiency, latent metric regularity, and prediction accuracy, and uses these distinctions to guide the evaluation and training of neural representations.

Building on classical Lipschitz extension theory, we connect an exact finite-sample error barrier, defined for a prescribed latent metric and Lipschitz budget, to a cross-fitted McShane regularizer for training microstructure encoders. This links an interpretable worst-case diagnostic to a computable training objective. Controlled comparisons across metric normalizations, latent dimensions, and regression and contrastive objectives identify representation deficiencies that conventional prediction or reconstruction scores can overlook.

We evaluate the methodology on a corpus of 3,300 periodic two-phase elastic cells and a frozen AT1 phase-field fracture benchmark kept separate from encoder training. On a 12-cell fracture test set with higher inclusion counts, augmenting an elastic-field-based critical-distance descriptor with frozen elastic embeddings increases the median peak-strength R-squared from $0.679$ to $0.939$, compared with $0.922$ using embeddings trained by mean squared error. The resulting mean relative peak-stress error is $0.94\%$. The study also identifies the dependence of performance on the evaluation metric and delineates the limits of transfer to a post-critical crack-path observable.

The physical descriptors, initial barrier-based diagnostic, and fracture benchmark originate in our related submitted manuscript, ``Energy and failure select different descriptors: counterexample-driven descriptor discovery in heterogeneous solids,'' which is cited in the present paper. The present submission develops the neural representation-learning methodology, cross-fitted regularization, and comparative assessment of learned embeddings across mechanical responses. This relationship and the reuse of the frozen benchmark are explicitly described in the manuscript. The source code, evaluation scripts, and verification artifacts are openly available at \url{https://github.com/aegovalle/sufficiency-ml} and included in the supplementary material.

We believe the combination of mathematical analysis, a concrete learning method, and mechanical validation makes this work well suited to CMAME. Its contribution should be of interest to researchers developing and assessing machine-learning surrogates for heterogeneous materials, damage initiation, and fracture.

This manuscript has not been published previously and is not under consideration by another journal. Both authors have approved the manuscript and its submission to \textit{Computer Methods in Applied Mechanics and Engineering}, and declare no competing interests.

Thank you for considering our manuscript.

\closing{Yours sincerely,}

\end{letter}
\end{document}
"""
    (d_cov / "cover_letter.tex").write_text(cover_letter_tex, encoding="utf-8")
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "cover_letter.tex"],
        cwd=d_cov,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    for ext in [".aux", ".log", ".out"]:
        p = d_cov / f"cover_letter{ext}"
        if p.exists():
            p.unlink()

    # 5. Declarations
    (d_dec / "declaration_of_competing_interest.txt").write_text(
        "Declaration of competing interest\n\n"
        "The authors declare that they have no known competing financial interests or personal relationships "
        "that could have appeared to influence the work reported in this paper.\n\n"
        "A. E. Gómez-Ovalle\n"
        "C. Ramírez Ovalle\n",
        encoding="utf-8",
    )
    (d_dec / "credit_author_statement.txt").write_text(
        "CRediT authorship contribution statement\n\n"
        "A.E. Gómez-Ovalle: Conceptualization, Methodology, Software, Formal analysis, Investigation, Visualization, Writing – original draft.\n"
        "C. Ramírez Ovalle: Methodology, Investigation, Validation, Writing – review & editing.\n",
        encoding="utf-8",
    )
    (d_dec / "data_availability.txt").write_text(
        "Data and code availability\n\n"
        "All source modules, execution and verification scripts, dataset split manifests, and CSV/JSON artifacts "
        "required to reproduce every table and figure in this manuscript are openly available in a public GitHub "
        "repository at https://github.com/aegovalle/sufficiency-ml (building on the frozen phase-field validation "
        "protocol archived at https://github.com/aegovalle/energy-failure-descriptors) and packaged in the "
        "supplementary verification archive Sufficiency_ML_Referee_Artifacts.zip.\n",
        encoding="utf-8",
    )
    (d_dec / "funding.txt").write_text(
        "Funding\n\n"
        "This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.\n",
        encoding="utf-8",
    )

    # 6. Figure list & README_submission.md
    fig_list_txt = "Figure List — CMAME Submission\n" + "=" * 60 + "\n\n"
    for num_prefix, stem, cap in fig_map:
        fig_list_txt += f"{num_prefix} -> 01_manuscript/{stem}.pdf (preview: 01b_figures_numbered_and_png/{num_prefix}_{stem[4:]}.png)\n  Caption: {cap}\n\n"
    (SUB_DIR / "figure_list.txt").write_text(fig_list_txt, encoding="utf-8")

    readme_sub = f"""# Submission Package — Computer Methods in Applied Mechanics and Engineering (CMAME)

**Title:** Sufficiency-aware representation learning for microstructure-to-failure prediction: Certified Lipschitz barriers, cross-fitted McShane regularization, and transfer to phase-field fracture
**Authors:** A. E. Gómez-Ovalle (a, corresponding), C. Ramírez Ovalle (b)
- (a) University of Burgos, Escuela Politécnica Superior, 09006 Burgos, Spain — `aegovalle@ubu.es`
- (b) Department of Natural Sciences & Mathematics, Pontificia Universidad Javeriana – Cali, Cali, Colombia

---

## Folder Inventory & Editorial Manager Item Types

| Folder / File | Content | Upload in Editorial Manager as |
| :--- | :--- | :--- |
| `04_cover_letter/cover_letter.pdf` (`.tex`, `.txt`) | Cover letter disclosing the related submitted manuscript (`GomezOvalle2026`) and summarizing the contribution | **Cover Letter** |
| `03_highlights/highlights.txt` (`.pdf`, `.tex`) | 5 bullet points, each strictly $\le 85$ characters including spaces (73–77 chars) | **Highlights** |
| `01_manuscript/manuscript.pdf` | Compiled 38-page manuscript PDF (`elsarticle` format, embedded bibliography) | **Manuscript** (or reference PDF) |
| `01_manuscript/manuscript.tex` | Single-file LaTeX manuscript source (zero external `.bib` dependencies) | **LaTeX Source File** / **Manuscript** |
| `01_manuscript/fig_*.pdf` (5 vector PDFs) | Exact figure filenames referenced by `\\includegraphics` in `manuscript.tex` (kept in the same folder because Editorial Manager flattens uploaded files) | **Figure** (upload all 5 `.pdf` files from `01_manuscript/`) |
| `01b_figures_numbered_and_png/` | Numbered vector PDFs (`Fig1_*.pdf`–`Fig5_*.pdf`) and 300 DPI `.png` previews | For visual inspection / optional PNG upload |
| `02_supplementary_material/Sufficiency_ML_Referee_Artifacts.zip` | Self-contained code, scripts, frozen CSVs, and shifted AT1 JSON trajectories (`SHA-256 = {zip_sha}`) | **Supplementary Material** / **Code & Data for Review** |
| `02_supplementary_material/Related_Submitted_Manuscript_GomezOvalle2026_For_Review_Only.pdf` | Copy of the companion submitted paper (`GomezOvalle2026`) | **Related Manuscript (For Review Only)** (optional, recommended by Elsevier policy) |
| `05_declarations/*.txt` | Competing interests, CRediT author statement, data/code availability, funding | **Declaration of Interest** & submission metadata fields |

---

## Highlights Character-Count Verification (Max 85 chars per bullet)

1. `Separates functional sufficiency, metric regularity, and readout accuracy` (**73 chars**)
2. `Links exact finite-sample Lipschitz barriers to cross-fitted McShane training` (**77 chars**)
3. `SBR-CF cuts worst-case latent collisions by 30% and mitigates d=16 dilution` (**75 chars**)
4. `Compares RNC 1D compression and multi-D SBR-CF across five latent metrics` (**73 chars**)
5. `Frozen SBR-CF embeddings lift prospective fracture peak-strength R2 to 0.939` (**76 chars**)

---

## Recommended Upload Order in CMAME Editorial Manager

1. **Cover Letter:** `04_cover_letter/cover_letter.pdf`
2. **Highlights:** `03_highlights/highlights.txt` (or `highlights.pdf`)
3. **Manuscript:** `01_manuscript/manuscript.tex` (and `01_manuscript/manuscript.pdf`)
4. **Figures:** Select all 5 vector PDFs inside `01_manuscript/`:
   - `fig_counterexample_resolution.pdf`
   - `fig_lambda_and_dim_sweep.pdf`
   - `fig_ablation7_metrics.pdf`
   - `fig_phase3_ablations.pdf`
   - `fig_transfer_and_pf.pdf`
5. **Supplementary Material:** `02_supplementary_material/Sufficiency_ML_Referee_Artifacts.zip` (and optionally `Related_Submitted_Manuscript_GomezOvalle2026_For_Review_Only.pdf` as Related Manuscript for Review Only)
6. **Declarations:** Copy-paste from `05_declarations/` into the Editorial Manager prompts (or upload `05_declarations/declaration_of_competing_interest.txt`).
"""
    (SUB_DIR / "README_submission.md").write_text(readme_sub, encoding="utf-8")
    print("Successfully built CMAME_Submission_Package at:", SUB_DIR)
    print("Referee ZIP SHA-256:", zip_sha)


if __name__ == "__main__":
    main()
