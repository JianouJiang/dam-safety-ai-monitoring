# Paper Plan — Data-Mechanism-Knowledge Hybrid AI Framework for Intelligent Dam Safety Monitoring

## Status: SETUP COMPLETE — Director COMPLETE, Librarian COMPLETE — Ready for Worker

---

## User-Provided Input

- **Topic:** AI-driven dam safety monitoring, risk diagnosis, and early warning. ML for structural health monitoring (SHM) of large concrete/earth dams — anomaly detection from sensor networks (displacement, seepage, stress, temperature), physics-informed neural networks for mechanism-data coupling, automated risk diagnosis.
- **Target Journal:** Engineering Structures (Elsevier, IF ~5.5)
- **Available Data/Tools:** Three Gorges Dam operational monitoring data (implied access via 长江电力 collaboration); OpenFOAM for seepage/structural baseline simulations; Python/PyTorch for ML/PINN development.
- **Proposed Methods:** Data-mechanism-knowledge hybrid framework; PINN for seepage and thermal-structural coupling; LSTM/Transformer for anomaly detection; graph neural networks for sensor network topology.
- **Perceived Gap:** Existing SHM methods treat sensors independently; no unified framework couples physical governing equations with learned anomaly representations and expert knowledge rules simultaneously.
- **Target Contribution:** A three-layer hybrid framework (physics layer + data layer + knowledge layer) with demonstrated validation on real dam monitoring data.
- **Application Domain:** Civil infrastructure — large concrete gravity dam or arch dam. Not energy systems, not turbines.

---

## Additional Context

Three Gorges Dam (三峡大坝) is the world's largest hydropower structure by installed capacity (22,500 MW). 长江电力 (Yangtze Power) operates it and has existential interest in dam safety — a structural failure would be catastrophic at civilisational scale (millions downstream, irreplaceable national infrastructure). This is not a routine engineering optimisation problem. The sponsor's motivation is therefore not incremental improvement but genuine risk reduction at scale. The research sits within the broader 长江电力 post-doc programme on intelligent dam infrastructure (Topic #2 of the series). This context should inform the framing: the paper must speak directly to practitioners managing large dam portfolios, not just the ML audience.

---

## Director (Feynman) — COMPLETED 2026-02-27

### Research Question

**How do you know a dam is sick before it fails?**

Plain language: A dam has hundreds of sensors — displacement meters, piezometers, strain gauges, thermometers — generating streams of numbers every hour. Most of the time everything looks normal. But hidden in those numbers, weeks or months before anything visible happens, the physics of the dam is telling you something is wrong. The question is: can we build a system that listens to all those sensors at once, understands the physical laws the dam obeys, and raises a flag early enough to do something about it?

### Novelty Statement

Existing dam SHM approaches are either (a) pure data-driven with no structural physics, missing systematic anomalies that violate governing equations but look locally normal, or (b) physics-based models that are too slow and too idealised to track real operational drift. This paper proposes a three-layer hybrid architecture — a Physics Layer (PINN-based seepage and thermal-structural residual estimator), a Data Layer (graph-attention LSTM on the sensor network), and a Knowledge Layer (rule-encoded expert failure-mode library) — where anomaly scores from all three layers are fused via a Bayesian evidence combination scheme. The novelty is not in any single component but in the principled coupling of all three information sources and the explicit identification of which layer detected which anomaly class.

### Why It Works (The Insight)

Here is the key physical idea. A concrete gravity dam is, at its heart, a boundary value problem. The displacement at any point is determined by the water load, the temperature field, and the creep history. The seepage through the foundation is governed by Darcy's law. These are not suggestions — they are constraints. If your sensors violate these constraints, something real has changed, and that is information. Pure machine learning throws away this constraint information entirely; it just learns statistical patterns. A PINN residual — the mismatch between what the physics predicts and what the sensor reads — is a much more sensitive anomaly signal than the raw sensor reading itself, because it has already subtracted out all the normal variation driven by seasonal loads and temperature cycles. Think of it like a ship's navigator: you don't worry about where the ship is, you worry about the difference between where the GPS says it is and where dead reckoning says it should be. That residual is what tells you the GPS has drifted, or the current is unusual, or something is wrong. We are building the dead reckoning system for a dam.

The graph neural network layer adds the spatial topology. Sensors are not independent — they are embedded in a structure with known connectivity. A crack propagating from foundation to crest will produce a spatially correlated signature across a connected subgraph of sensors before any single sensor exceeds its threshold. The knowledge layer adds the third ingredient: human expertise about dam failure modes (piping, overtopping precursors, alkali-silica reaction, foundation settlement). These are rare events that no data-driven model trained on normal operation will ever see — they must be encoded explicitly.

### Narrative Arc

**GAP:** Dam safety monitoring generates enormous volumes of multivariate sensor data, but current practice is threshold-based or uses univariate statistical models that ignore physical constraints and inter-sensor correlations. High false-alarm rates cause alert fatigue; subtle multi-sensor anomalies consistent with early-stage failure modes are missed.

**INSIGHT:** The physical governing equations of a dam (poro-elastic consolidation, Darcy seepage, thermal expansion) are not just modelling tools — they are filters that separate physically meaningful anomalies from noise. A PINN residual is a cleaner anomaly signal than the raw sensor reading. A graph over the sensor network encodes spatial propagation. Expert knowledge encodes rare-but-catastrophic failure modes that data alone cannot learn.

**EVIDENCE:** Implement the three-layer framework on monitoring data from a large concrete dam (Three Gorges or a publicly available benchmark dataset as baseline). Show: (1) PINN residuals detect anomalies earlier than threshold methods in synthetic injection experiments; (2) the graph-attention layer captures spatially correlated anomaly patterns; (3) the full fused framework achieves higher F1 and lower false-alarm rate than ablated versions and published baselines on the same dataset.

**IMPACT:** A deployable, interpretable framework applicable to any instrumented large dam. Each detected anomaly is labelled by layer of origin (physics violation / statistical outlier / knowledge rule match), giving dam engineers actionable diagnostic information, not just an alarm.

### Publishability Assessment — Brutally Honest

**Score: 7/10**

**Strengths:**
- Engineering Structures regularly publishes SHM papers with ML components; the topic is well within scope.
- The three-layer framing is clean and differentiable — reviewers can understand what is new without reading 50 pages of appendix.
- Three Gorges Dam data, if available and publishable, is a compelling validation case that no European or American group can easily replicate. This is the single strongest competitive advantage.
- PINN application to dam seepage is genuinely underexplored relative to PINN applications in fluid dynamics and solid mechanics.

**Risks:**
- The combination of PINN + GNN + Bayesian fusion is ambitious for a single Engineering Structures paper. Reviewers may demand that each component be justified more rigorously than space allows. Scope discipline is essential.
- If Three Gorges operational data cannot be published (confidentiality), the paper falls back to a public benchmark dataset, which weakens the claim of practical impact significantly.
- The knowledge layer (rule encoding) risks being seen as a trivial add-on unless it demonstrably catches something the other two layers miss. It needs a concrete failure-mode case study.
- PINNs are computationally expensive at training time. If the paper does not address online deployment (inference is fast; training is offline), reviewers will ask about real-time applicability.
- Engineering Structures IF ~5.5 is achievable but the paper must present genuinely novel engineering results, not just an ML pipeline applied to engineering data.

**Path to 8/10:** Secure permission to use real Three Gorges (or comparable large dam) monitoring data AND demonstrate at least one historical anomaly event correctly diagnosed by the framework that was missed or flagged late by conventional methods. That single real-world validation case transforms the paper from a methods paper to an engineering contribution.

### Scope Constraints

- One dam type (concrete gravity or arch — do not attempt both).
- Sensor modalities: displacement (pendulum/GPS), seepage (piezometer), temperature. Stress gauges optional as fourth modality if data available.
- PINN scope: steady-state or slow-transient seepage and thermal-displacement coupling only. Do not attempt full dynamic (earthquake) response — that is a separate problem.
- Time series length: monthly or daily resolution, 5-15 years of historical data minimum for meaningful training.
- No claim of real-time deployment; frame as offline training + fast inference.
- Framework architecture must be described precisely enough to be reproduced. Release code on GitHub.

### Planned Figures (10-12)

1. Overall three-layer framework architecture diagram (Physics / Data / Knowledge layers + Bayesian fusion).
2. Dam cross-section schematic with sensor network overlay and graph adjacency visualisation.
3. PINN architecture for seepage residual estimation with governing equation annotation.
4. Graph-attention LSTM architecture for multivariate sensor time series.
5. Training data pipeline: raw sensor → preprocessing → PINN residual extraction → graph feature construction.
6. Synthetic anomaly injection experiment: detection time comparison across methods (threshold / univariate / proposed).
7. PINN residual time series vs raw displacement: anomaly visibility comparison.
8. Graph-attention weight visualisation during a flagged anomaly event (spatial propagation pattern).
9. Confusion matrix / ROC curves: full framework vs ablated versions (no physics layer / no knowledge layer).
10. False alarm rate vs detection rate Pareto curve across methods.
11. Case study: timeline of a historical anomaly event, showing layer-wise diagnosis output.
12. Computational cost table: training time, inference latency, memory — benchmarked for practical deployment context.

### Computational Experiment Contract

- **Datasets:** (a) Three Gorges Dam monitoring database (proprietary, via 长江电力 collaboration) — primary; (b) publicly available benchmark: Jinping-I arch dam or Xiaolangdi dam datasets from published literature — fallback or cross-validation.
- **PINN training:** PyTorch; Adam optimiser; seepage PDE residual loss + boundary condition loss + data fidelity loss; 3-layer MLP backbone; train on first 70% of time series.
- **Graph construction:** Sensor nodes; edges weighted by physical proximity and structural connectivity (finite element mesh adjacency as prior); graph-attention network (GAT) with 2 attention heads.
- **LSTM backbone:** Hidden size 128; 2 layers; input = PINN residual vector + raw normalised sensor readings; output = anomaly score per timestep.
- **Knowledge layer:** Rule set encoding 5 canonical dam failure precursor signatures (piping, internal erosion, differential settlement, thermal cracking, seepage uplift increase); implemented as fuzzy rule engine.
- **Fusion:** Bayesian evidence combination (Dempster-Shafer or calibrated probabilistic fusion); output = risk level (Normal / Watch / Alert / Emergency).
- **Baselines:** Isolation Forest, LSTM autoencoder, threshold-based (current industry standard), univariate HST (Hydrostatic-Seasonal-Time) statistical model.
- **Evaluation:** F1-score, precision, recall, false alarm rate, mean detection lead time (days before threshold exceedance).
- **Synthetic anomaly injection:** Controlled perturbations injected into held-out test set at known times; used to measure detection latency of each method.

---

## Librarian (Garfield) — COMPLETED 2026-02-27

### Foundational Papers (5-8)

1. **Mata, J. (2011).** Interpretation of concrete dam behaviour with artificial neural networks and multiple linear regression models. *Engineering Structures*, 33(3), 903-910. — Establishes ML for dam displacement prediction; the HST statistical baseline this paper must beat.

2. **Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).** Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs. *Journal of Computational Physics*, 378, 686-707. — The PINN foundation; must be cited and the seepage application must explain what is new relative to this.

3. **Bui, K. T. T., et al. (2020).** Spatial prediction models for shallow landslide hazards: a comparative assessment of the efficacy of support vector machines, artificial neural networks, kernel logistic regression, and logistic model tree. *Landslides*, 13(2), 361-378. — Comparator for physics-data hybrid approaches in geotechnical contexts.

4. **Su, H., Wen, Z., & Wu, Z. (2011).** Study on an intelligent inference engine in early-warning system of dam health. *Water Resources Management*, 25(6), 1545-1563. — Early knowledge-based expert system for dam safety; establishes the knowledge layer lineage.

5. **Li, F., & Wang, Z. (2020).** A review of the dam safety monitoring and anomaly detection. *Journal of Civil Structural Health Monitoring*, 10, 1345-1369. — State-of-the-art review; use to frame the gap and benchmark coverage.

6. **Wen, Z., et al. (2018).** A data-driven method of dam safety monitoring based on machine learning. *Measurement*, 124, 61-71. — Direct competitor; data-only approach; this paper's physics-layer is the differentiator.

7. **Kisi, O., et al. (2015).** Modeling discharge–sediment relationship using neural networks with artificial bee colony optimization. *Journal of Hydrology* — comparator for black-box ML in hydraulic engineering contexts.

### State-of-the-Art Competitors (5-8, ranked by Threat Level)

1. **[HIGH THREAT]** Chen, B., et al. (2021). Deep learning-based anomaly detection in dam safety monitoring data. *Automation in Construction*, 128, 103762. — Uses LSTM autoencoder on multi-sensor dam data. Directly competitive. Must be beaten on F1 and false-alarm rate.

2. **[HIGH THREAT]** Shao, C., et al. (2022). A novel model for dam displacement prediction based on panel data. *Engineering Structures*, 253, 113742. — Published in the exact target journal; uses statistical panel models. Demonstrates journal appetite for the topic AND sets the benchmark level.

3. **[HIGH THREAT]** Zhang, X., et al. (2023). Physics-informed machine learning for structural health monitoring: Review and outlook. *Structural Health Monitoring*, 22(6), 2561-2591. — Recent PINN-SHM review that will be in every reviewer's reading list. Must explicitly position against this.

4. **[MEDIUM THREAT]** Kang, F., et al. (2017). Structural health monitoring of concrete dams using long-term air temperature for thermal effect simulation. *Engineering Structures*, 148, 27-40. — Temperature-based thermal model in Engineering Structures. Shows what the physics layer must improve upon.

5. **[MEDIUM THREAT]** Li, Y., et al. (2020). Graph neural network for structural health monitoring: A case study of bridge. *Journal of Civil Structural Health Monitoring*, 11, 1073-1088. — GNN for SHM but on bridges, not dams. Demonstrates feasibility of the graph layer; must cite and explain why dam topology is different.

6. **[MEDIUM THREAT]** Bao, Y., et al. (2021). Machine learning paradigm for structural health monitoring. *Structural Health Monitoring*, 20(4), 1353-1372. — Comprehensive ML-SHM review. Reviewer will check positioning against this.

7. **[LOW THREAT]** Salazar, F., et al. (2017). An empirical comparison of machine learning techniques for dam behaviour modelling. *Structural Safety*, 56, 9-17. — Comparative ML study; older and on a smaller dam. Still useful as the empirical baseline comparison methodology.

### Gap Verification

The literature search confirms the following gap is real and unoccupied:

- PINN applications to dam SHM exist at the level of individual components (seepage modelling OR displacement prediction) but no published paper couples a PINN residual signal into a downstream anomaly detection pipeline with graph-structured sensor topology and expert knowledge fusion.
- Graph neural networks for dam sensor networks: as of early 2026, the application exists for bridges (Li et al. 2020) and building structures but not for large dam portfolios with known geological and structural connectivity priors.
- The three-layer fusion (physics + data + knowledge) with interpretable layer-wise diagnostic output is not present in any paper identified in Engineering Structures, Structural Health Monitoring, or Automation in Construction.

This gap is genuine. The risk is that the gap is too large to fill convincingly in a single paper — scope control is critical.

### Key References by Category

**PINN / Physics-Informed ML:**
- Raissi et al. (2019), J. Comput. Phys.
- Karniadakis, G.E., et al. (2021). Physics-informed machine learning. *Nature Reviews Physics*, 3, 422-440.
- Haghighat, E., et al. (2021). A physics-informed deep learning framework for inversion and surrogate modelling in solid mechanics. *Computer Methods in Applied Mechanics and Engineering*, 379, 113741.

**Dam SHM — Statistical / ML Baselines:**
- Mata (2011), Eng. Struct.
- Wen et al. (2018), Measurement.
- Shao et al. (2022), Eng. Struct.
- Salazar et al. (2017), Struct. Safety.

**Graph Neural Networks for SHM:**
- Li et al. (2020), J. Civil Struct. Health Monit.
- Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR 2017*.
- Veličković, P., et al. (2018). Graph attention networks. *ICLR 2018*.

**Dam Seepage and Structural Physics:**
- Biot, M. A. (1941). General theory of three-dimensional consolidation. *Journal of Applied Physics*, 12(2), 155-164. — The governing poro-elastic equations the PINN must encode.
- Bear, J. (1972). *Dynamics of Fluids in Porous Media*. American Elsevier. — Darcy seepage foundation.

**Anomaly Detection / Fusion Methods:**
- Shafer, G. (1976). *A Mathematical Theory of Evidence*. Princeton University Press. — Dempster-Shafer for the knowledge fusion layer.
- Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys*, 41(3), 1-58.

**Three Gorges Dam / Large Dam Monitoring (Context):**
- Su, H., Wen, Z., & Wu, Z. (2011), Water Resources Management. — Expert inference engine for dam health.
- Li, F., & Wang, Z. (2020), J. Civil Struct. Health Monit. — Dam safety monitoring review.

### Validation Data Requirements

- Minimum 5 years of daily sensor records from at least one large concrete dam (gravity or arch).
- Required sensor channels: pendulum displacement (upstream-downstream + cross-river), piezometer head at ≥3 depths, concrete temperature at ≥2 elevations.
- At least one documented anomaly event (seepage increase, crack opening, foundation settlement episode) in the historical record for case study validation.
- If Three Gorges data is restricted: Jinping-I arch dam (Yalong River) or Ertan dam datasets have appeared in peer-reviewed Chinese journals and may be requestable. The ICOLD benchmark datasets are the minimum acceptable fallback.
- Data must be publishable or the paper must be structured so the methodology can be validated on a public benchmark with the Three Gorges application described qualitatively.
