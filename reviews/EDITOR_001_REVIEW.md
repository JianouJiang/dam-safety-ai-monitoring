# EDITOR Review #001 (Writing / LaTeX / Presentation)

**Project:** `projects/dam_safety_ai`  
**Date:** 2026-03-05  
**Scope:** Writing clarity & style, LaTeX/template compliance, layout/float hygiene, reference integrity. (No scientific-method critique.)

## Executive summary
- The paper is structurally close to an Elsevier `elsarticle` submission and is under the 35-page limit (current PDF is 30 pages total; references begin ~p.27 per the layout report).
- The main presentation blockers are **float/cross-reference hygiene** (one undefined figure reference; several figures not cited in-text), **tables with placeholder entries**, and **overuse of enumerated lists with bold lead-ins** (reads “AI-outline-y”).
- Programmatic layout analysis is currently dominated by a “white space” detector that appears poorly calibrated for text-heavy pages; treat its L1 flags as **low-confidence** until the metric is revised.

## Template & LaTeX compliance (Elsevier / `elsarticle`)
- `\documentclass[preprint,12pt,review]{elsarticle}` matches the Elsevier template pattern for review mode. Good.
- Line numbers are enabled (`lineno` + `\linenumbers`), appropriate for review.
- Missing common Elsevier end-matter sections (journal-dependent, but typically expected):
  - `\section*{Declaration of competing interest}`
  - `\section*{Data availability}` (or a clear data/code availability statement)
  - CRediT author contributions statement (often requested)
  These can be short, but should exist before submission.

## Cross-references & float hygiene (CRITICAL)
1. **Undefined reference:** `Fig.~\ref{fig:framework}` is undefined (confirmed in `manuscript/main.log`). This is a hard stop for submission-quality compilation.
   - Likely cause: the manuscript refers to a “framework” figure that does not exist (`manuscript/figures/` starts at `fig02_...`; no `fig01_...` present).
2. **Figures lacking in-text callouts:** The following figure labels are defined but never referenced in the text (detected by static scan):
   - `fig:raw_data`, `fig:roc`, `fig:score_comparison`
   Each figure should be introduced in prose **before** it appears (Elsevier reviewers will flag “floating figures with no discussion”).
3. **Figure label/text mismatch risk:** The first figure shown in Methodology is the sensor network, but the surrounding text currently claims it “illustrates the overall architecture.” Ensure the text aligns with the actual figure content (and with the correct `\label{...}`).

## Tables (CRITICAL / incomplete placeholders)
- At least two tables contain `---` placeholders for multiple metrics (e.g., method comparison table and ablation table). This is fine mid-draft, but it severely degrades presentation in the compiled PDF and should be resolved before internal circulation.

## Writing style (Strunk: omit needless words; avoid outline tone)
- The manuscript repeatedly uses `enumerate` blocks with **bold colon lead-ins** (e.g., “Interpretability:”, “Low false alarm rate:”, “Synthetic data:”, etc.).
  - This reads like slideware/outline, not journal prose.
  - Recommendation: convert these to short paragraphs with clear topic sentences, using plain text (no bold label) unless the journal style explicitly encourages it.
- The “contributions” section is currently a list. Many journals allow this, but your project style rule (“no bullet lists in main text”) suggests rewriting as a compact paragraph.

## Layout analyzer results (use with caution)
- `manuscript/layout_report.md` reports **91 CRITICAL defects**, all of which are L1 “high white space / vertical gaps” and are triggered on essentially every page.
- Given the pixel-level nature of text on white paper, this looks like a **false-positive regime** rather than a real “blank page / huge float gap” problem. Until the analyzer is recalibrated for `elsarticle` preprint pages, treat these as informational only.

## Figure inspector results (actionable)
- `manuscript/figure_report.md`: average sophistication **7.6/10**, **0 AI-lazy flags** — good overall.
- However, several figures are flagged for **excessive internal white space** (likely undersized panels/fonts or too much padding):
  - `fig03_raw_data.png` (83% white space)
  - `fig04_pinn_residuals.png` (92% white space; lowest score 6.5/10)
  - `fig09_roc_curves.png` (92% white space)
  - `fig10_pr_curves.png` (94% white space)
  - `fig11_ablation.png` (74% white space)
  - `fig12_lead_time.png` (79% white space)
  Suggested fix: tighten subplot spacing, enlarge axis fonts/lines, and crop unused margins so each figure earns its page real estate.

## References: integrity checks (anti-hallucination)
**Citekey hygiene**
- All `\cite{...}` keys in `manuscript/main.tex` exist in `manuscript/references.bib` (no missing bib entries for cited keys found).
- Uncited bib entries currently present (potential orphans): `wen2018data`, `shao2022novel`, `kang2017structural`, `wang2021deep`, `liu2008isolation`, `chandola2009anomaly`. Either cite them in relevant locations or remove them to avoid “reference padding.”

**Existence verification (10 DOI checks) — PASS**
The following DOIs resolve via `https://doi.org/...` (HTTP 302), indicating the entries are real:
- 10.1016/j.engstruct.2010.12.011
- 10.1016/j.jcp.2018.10.045
- 10.1007/s11269-010-9760-3
- 10.1016/j.autcon.2021.103762
- 10.1016/j.engstruct.2021.113742
- 10.1038/s42254-021-00314-5
- 10.1016/j.cma.2021.113741
- 10.1162/neco.1997.9.8.1735
- 10.1109/ICDM.2008.17
- 10.1063/1.1712886

## Top priority checklist for the Worker
1. Add/restore the missing “framework” figure (or fix the label/reference so `fig:framework` resolves).
2. Add explicit in-text mentions for `fig:raw_data`, `fig:roc`, `fig:score_comparison` before the floats.
3. Replace `---` placeholders in tables with real values (or temporarily remove the tables from the compiled draft if values are not ready).
4. Rewrite bold-labeled enumerations into paragraph prose to reduce “outline” tone.
5. Add required end-matter statements (Competing interest, Data availability, CRediT) as short starred sections.

Score: 6.5/10

