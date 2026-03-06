# ILLUSTRATOR_001 — Figure Quality & Data Visualization Review (Tufte lens)

**Scope reviewed (what exists now):** `manuscript/figures/fig02`–`fig12` (11 figures).  
**Programmatic evidence used:** existing `manuscript/figure_report.md` (generated **2026-03-05 21:28:13**).  
**Note on required workflow:** I did **not** run `figure_inspector.py` because this reviewer role forbids running scripts; I relied on the already-generated report plus manual inspection.

## Blockers / High-priority issues (fix before submission)

1. **Fig. 04 has a major axis/data mapping error in panel (b).** The “score distribution” x-axis shows year-like ticks (e.g., ~1970–2020) rather than residual score. This is a *lie-factor* problem: it reads like the wrong variable is being plotted (likely timestamps or an index) or the axis formatter is wrong.
2. **Color semantics are inconsistent across figures (especially Fig. 12).** Example: in time-series figures Proposed Framework is magenta, GAT-LSTM is orange; in Fig. 12 GAT-LSTM is teal/green and Proposed is orange. This breaks “same color = same method” and damages cognition.
3. **AI-lazy visuals present (bar charts) in Fig. 11 and Fig. 12.** These are reviewer-magnet figures: easy targets for “too simplistic / no uncertainty / no distribution shown”.
4. **Excessive white space reported in multiple figures** (Fig. 03/04/09/10/11/12). Even if the content is correct, the current scaling wastes page real estate and reduces data density.

## Missing reference standard (process gap)

- `related_papers/FIGURE_QUALITY_STANDARDS.md` is **not present**, and `related_papers/` is essentially empty. I cannot do the requested “closest competitor figure” comparisons yet. Please have the Librarian/Worker populate this so figure ambition can be benchmarked against target journals.

## figure_report.md — Summary table (scores copied)

| Figure (PNG) | Score (/10) | Notable metrics / flags (from report) |
|---|---:|---|
| fig02_sensor_network.png | 8.5 | 14 panels est.; ~49% white space |
| fig03_raw_data.png | 8.0 | **83% white space** flagged |
| fig04_pinn_residuals.png | 6.5 | **92% white space** flagged |
| fig05_gat_attention.png | 8.5 | 9 panels est.; ~45% white space |
| fig06_knowledge_activation.png | 7.0 | 3 panels est.; ~50% white space |
| fig07_fusion_contributions.png | 9.0 | 4 panels est.; ~57% white space |
| fig08_score_comparison.png | 9.0 | 20 panels est.; **69% white space** flagged |
| fig09_roc_curves.png | 7.0 | **92% white space** flagged |
| fig10_pr_curves.png | 7.0 | **94% white space** flagged |
| fig11_ablation.png | 6.5 | **74% white space** flagged |
| fig12_lead_time.png | 6.5 | **79% white space** flagged |

## Style consistency vs plotting_utils.py (project standard)

`codes/utils/plotting_utils.py` sets serif fonts (Times New Roman preferred) and defines a palette `COLORS`. Many figures appear to respect the serif typography, but **method-to-color mapping is not stable across figures**.

**Recommended “method colors” (lock this across every figure):**

- Proposed (DMK): `COLORS["quaternary"]` (magenta)
- PINN (Physics): `COLORS["primary"]` (green)
- GAT-LSTM (Data): `COLORS["secondary"]` (orange)
- Knowledge: `COLORS["tertiary"]` (purple)
- Baselines: map consistently (e.g., Threshold = `octonary`, HST = `quinary`, IF = `senary`, LSTM-AE = `septenary`)

## Per-figure review (visual + checklist)

### Fig. 02 — Sensor network configuration

**What works**
- Clear two-panel story: physical layout + adjacency matrix.
- Adjacency heatmap has a colorbar with numeric scale (good).

**Fixes**
- Legend in (a) sits on data; move legend outside or use direct labeling near clusters.
- Reduce heavy border/box ink in (a) (data-ink ratio).
- Tick labels on adjacency matrix are sparse; consider **group separators** (D / P / T blocks) and thin lines to show sensor-type partitions explicitly.

### Fig. 03 — Raw monitoring data

**What works**
- Four stacked time series is the right format; anomaly windows + split lines are helpful.

**Fixes**
- White space is high: expand traces vertically (slightly larger axes height; tighter margins).
- Legends are small and repeat across panels; consider labeling lines directly at right edge (Tufte: “legend is a tax”).
- Ensure the anomaly shading color/alpha is identical across all time-series figures.

### Fig. 04 — PINN residual analysis (critical)

**What works**
- The intended narrative (time-series residual + distribution) is strong.

**Critical problems**
- Panel (b) x-axis looks like *time* (years) rather than “residual score”. This is not a cosmetic issue; it reads as the wrong data on the x-axis or incorrect axis formatting.
- Panel (a) shows the action only in a small region at the far right; the rest is blank ink.

**Redesign recommendation (journal-quality, distribution-forward)**
- Show **(a)** residual time series with an **inset zoom** around the test window (or use broken axis).
- Show **(b)** **violin + box + swarm** (Normal vs Anomaly), with log-scaled x if needed, and a threshold reference line.

**Python outline (≥30 lines)**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from codes.utils.plotting_utils import COLORS

def plot_pinn_residuals(time, residual, is_anomaly, split_dates, out_path):
    # Build dataframe for distribution panel
    df = pd.DataFrame({
        "time": pd.to_datetime(time),
        "residual": np.asarray(residual, dtype=float),
        "label": np.where(is_anomaly, "Anomaly", "Normal"),
    }).dropna()

    fig = plt.figure(figsize=(7.0, 5.5), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.6], hspace=0.25)
    ax_ts = fig.add_subplot(gs[0])
    ax_dist = fig.add_subplot(gs[1])

    # (a) time series with consistent anomaly shading
    ax_ts.plot(df["time"], df["residual"], color=COLORS["primary"], lw=1.2)
    for (t0, t1) in split_dates.get("anomaly_windows", []):
        ax_ts.axvspan(pd.to_datetime(t0), pd.to_datetime(t1),
                      color=COLORS["quaternary"], alpha=0.12, lw=0)
    for t in split_dates.get("splits", []):
        ax_ts.axvline(pd.to_datetime(t), color=COLORS["octonary"], ls="--", lw=0.9, alpha=0.8)
    ax_ts.set_ylabel("PINN residual score")
    ax_ts.set_title("(a) PINN physics residual over time", pad=6)

    # Optional inset zoom (test period)
    ax_in = ax_ts.inset_axes([0.62, 0.55, 0.36, 0.40])
    test_mask = df["time"] >= pd.to_datetime(split_dates["test_start"])
    ax_in.plot(df.loc[test_mask, "time"], df.loc[test_mask, "residual"],
               color=COLORS["primary"], lw=1.0)
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    ax_ts.indicate_inset_zoom(ax_in, edgecolor=COLORS["octonary"], lw=0.8)

    # (b) distribution: violin + box + points (shows uncertainty / dispersion)
    sns.violinplot(data=df.loc[test_mask], x="residual", y="label", orient="h",
                   inner=None, cut=0, linewidth=0.8, ax=ax_dist,
                   palette={"Normal": "#cccccc", "Anomaly": COLORS["quaternary"]})
    sns.boxplot(data=df.loc[test_mask], x="residual", y="label", orient="h",
                width=0.25, showcaps=True, showfliers=False,
                boxprops={"facecolor": "none", "edgecolor": "k", "linewidth": 0.8},
                whiskerprops={"linewidth": 0.8}, medianprops={"linewidth": 0.9},
                ax=ax_dist)
    sns.stripplot(data=df.loc[test_mask].sample(min(1200, df.loc[test_mask].shape[0]), random_state=0),
                  x="residual", y="label", orient="h", size=1.2, alpha=0.25,
                  color="k", ax=ax_dist)

    ax_dist.set_title("(b) Score distribution in test period", pad=6)
    ax_dist.set_xlabel("PINN residual score")
    ax_dist.set_ylabel("")
    ax_dist.grid(False, axis="y")

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
```

### Fig. 05 — GAT attention weights

**What works**
- Side-by-side heatmaps support the interpretation claim.

**Fixes (comparability)**
- Use a **single shared color scale** (same `vmin/vmax`) and preferably the **same colormap** for both panels. Right now Blue vs Red makes it hard to compare magnitude changes.
- Add a third panel: **Δattention = anomaly − normal** with a diverging colormap centered at 0 (this is where the story is).
- Add sensor-type block annotations (D/P/T) along axes to communicate structure without relying on a caption.

### Fig. 06 — Knowledge layer activations

**What works**
- Activation “bands” convey persistence and intensity.

**Fixes**
- Legend is oversized and competes with the data; move outside or replace with direct labels at the right edge of bands.
- Overplotting/opacity: reduce alpha so overlaps are readable; the current fills can obscure the anomaly shading.
- Color choices introduce many non-`COLORS` hues; ensure colorblind-safe separation among failure modes.

### Fig. 07 — Dempster–Shafer fusion contributions

**What works**
- Two-panel narrative works: per-layer evidence then fused belief.
- Shaded anomaly windows align with time series.

**Fixes**
- Stacked fills can mislead about absolute contributions (stacking implies parts-of-a-whole). Consider small multiples (three aligned axes) or lines with uncertainty bands.
- The legend could be replaced by inline labeling at the right edge (less ink, more clarity).

### Fig. 08 — Score comparison (test period)

**What works**
- Excellent multi-panel story: separates layers, proposed, and baselines.

**Fixes**
- Baseline panel mixes wildly different scales; consider **z-scoring** or **normalized scores** to compare shapes, plus a second inset showing raw scale if needed.
- Method color mapping should match the global mapping (see above).
- Reduce whitespace via tighter vertical spacing; current layout leaves large margins between panels.

### Fig. 09 — ROC curves

**What works**
- Correct baseline diagonal, AUCs provided.

**Fixes (journal expectation)**
- Too many curves with similar salience; add hierarchy: emphasize proposed method and best baselines; fade others.
- Add **confidence bands across seeds** (already mentioned in text) or at least show mean±std AUC in legend.
- Add an inset focusing on the low-FPR region (engineering relevance is often at FAR < 1–5%).

### Fig. 10 — Precision–Recall curves

**What works**
- PR curves are appropriate for imbalanced anomaly detection.

**Fixes**
- Same issues as ROC: hierarchy + uncertainty + low-recall/high-precision inset.
- PR-AUC labeling: consider direct labels to reduce legend footprint.

### Fig. 11 — Ablation study (AI-lazy)

This figure is **AI-lazy**: three bar charts with point estimates only. A reviewer will ask: *Where is the variability across seeds? Are differences statistically meaningful?*

**Required redesign**
- One composite figure with **three rows (metrics)** and a shared x-axis for configurations, showing:
  - distribution across seeds (violin + swarm),
  - mean±std overlays,
  - and **Δ vs Full** annotations (effect sizes).

**Python outline (≥30 lines)**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from codes.utils.plotting_utils import COLORS

def plot_ablation_distributions(df_runs, out_path):
    """
    df_runs columns:
      - config: {"Full","No Physics","No Knowledge","No Data"}
      - seed: int
      - f1, roc_auc, far: floats
    """
    order = ["Full", "No Physics", "No Knowledge", "No Data"]
    metrics = [("f1", "F1 score"), ("roc_auc", "ROC-AUC"), ("far", "False alarm rate")]
    palette = {
        "Full": COLORS["quaternary"],
        "No Physics": COLORS["secondary"],
        "No Knowledge": COLORS["tertiary"],
        "No Data": COLORS["primary"],
    }

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7.0, 6.3), dpi=300, sharex=True)
    for ax, (m, title) in zip(axes, metrics):
        sns.violinplot(data=df_runs, x="config", y=m, order=order,
                       cut=0, inner=None, linewidth=0.8,
                       palette=palette, ax=ax)
        sns.boxplot(data=df_runs, x="config", y=m, order=order, width=0.20,
                    showfliers=False, whis=(5, 95),
                    boxprops={"facecolor": "none", "edgecolor": "k", "linewidth": 0.8},
                    medianprops={"color": "k", "linewidth": 1.0},
                    whiskerprops={"linewidth": 0.8},
                    ax=ax)
        sns.stripplot(data=df_runs, x="config", y=m, order=order,
                      color="k", size=2.3, alpha=0.35, jitter=0.18, ax=ax)

        # Effect size annotations vs Full (mean deltas)
        full_mean = df_runs.loc[df_runs["config"] == "Full", m].mean()
        for i, cfg in enumerate(order):
            cfg_mean = df_runs.loc[df_runs["config"] == cfg, m].mean()
            delta = cfg_mean - full_mean
            ax.text(i, ax.get_ylim()[1], f"Δ={delta:+.3f}",
                    ha="center", va="top", fontsize=8)

        ax.set_title(title, pad=4)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("")

    axes[-1].set_xlabel("Configuration")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
```

### Fig. 12 — Detection lead time (AI-lazy)

This is **AI-lazy**: a single horizontal bar chart (point estimates). Lead time is an event-level quantity; bars conceal distribution, censoring (missed events), and per-event spread.

**Required redesign**
- **Forest plot / dot+CI** for each method (mean with bootstrap CI).
- Add per-event points (jittered) and optionally facet by anomaly type (4 injected events).
- Ensure “lower is better” is encoded visually (e.g., invert x-axis or annotate direction).

**Python outline (≥30 lines)**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from codes.utils.plotting_utils import COLORS

def bootstrap_ci(values, n=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, (np.nan, np.nan)
    samples = rng.choice(values, size=(n, values.size), replace=True).mean(axis=1)
    lo = np.quantile(samples, alpha / 2.0)
    hi = np.quantile(samples, 1 - alpha / 2.0)
    return values.mean(), (lo, hi)

def plot_lead_time_forest(df_events, out_path):
    """
    df_events columns:
      - method: str
      - event_id: str/int
      - anomaly_type: str (optional)
      - lead_time_days: float (NaN if missed)
    """
    method_order = [
        "Proposed Framework", "PINN (Physics)", "GAT-LSTM (Data)", "Knowledge Layer",
        "Threshold", "HST", "Isolation Forest", "LSTM-AE",
    ]
    method_color = {
        "Proposed Framework": COLORS["quaternary"],
        "PINN (Physics)": COLORS["primary"],
        "GAT-LSTM (Data)": COLORS["secondary"],
        "Knowledge Layer": COLORS["tertiary"],
        "Threshold": COLORS["octonary"],
        "HST": COLORS["quinary"],
        "Isolation Forest": COLORS["senary"],
        "LSTM-AE": COLORS["septenary"],
    }

    rows = []
    for m in method_order:
        vals = df_events.loc[df_events["method"] == m, "lead_time_days"].to_numpy()
        mean, (lo, hi) = bootstrap_ci(vals)
        miss_rate = np.mean(~np.isfinite(vals)) if vals.size else np.nan
        rows.append({"method": m, "mean": mean, "lo": lo, "hi": hi, "miss_rate": miss_rate})
    summ = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=300)
    y = np.arange(len(method_order))[::-1]
    ax.hlines(y, summ["lo"], summ["hi"], color="#444444", lw=1.2, alpha=0.9)
    ax.scatter(summ["mean"], y, s=48,
               c=[method_color[m] for m in summ["method"]], zorder=3)

    # Event-level points (jittered), exposes distribution and censoring
    rng = np.random.default_rng(0)
    for i, m in enumerate(method_order):
        vals = df_events.loc[df_events["method"] == m, "lead_time_days"].to_numpy()
        yy = np.full_like(vals, fill_value=y[i], dtype=float) + rng.normal(0, 0.06, size=vals.size)
        ax.scatter(vals[np.isfinite(vals)], yy[np.isfinite(vals)],
                   s=12, color="k", alpha=0.25, zorder=2)
        ax.text(ax.get_xlim()[1] if np.isfinite(ax.get_xlim()[1]) else 0,
                y[i], f"miss={summ.loc[summ['method']==m,'miss_rate'].iloc[0]:.0%}",
                ha="left", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(method_order)
    ax.set_xlabel("Detection lead time (days) — lower is better")
    ax.set_title("Lead time with bootstrap 95% CI and event-level points", pad=6)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
```

## Manuscript integration check (figure callouts)

- `manuscript/main.tex` references figs 02–09 and 11. **Fig. 10 and Fig. 12 exist but are not referenced** (as of this snapshot). If they’re intended for the paper, add them; if not, remove from figure set to avoid confusion.

## Anti-patterns found (flagged)

- **Lie factor / wrong axis:** Fig. 04(b).
- **Inconsistent semantic color mapping:** Fig. 12 (and potential minor drift elsewhere).
- **AI-lazy bar charts:** Fig. 11, Fig. 12 (no uncertainty/distribution).
- **Legend tax:** repeated legends inside axes when direct labeling would work (Fig. 03/06/09/10).
- **Whitespace waste:** flagged by report in multiple figures; tighten layouts.

## Score

Score: 6/10

*Rationale:* strong overall direction (multi-panel storytelling, consistent anomaly shading, mostly publication-ready typography), but current set has one critical correctness/encoding issue (Fig. 04), plus two prominent “simple bar chart” figures that will draw reviewer criticism unless upgraded to distribution+uncertainty displays with consistent method colors.

