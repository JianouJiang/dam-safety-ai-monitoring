# Layout Analysis Report
**Generated**: 2026-03-05 21:27:56
**Paper**: PAPER_paper

## Summary
- Total pages: 31
- Content pages (excl. references): 26
- Reference section starts: page 27
- **CRITICAL defects: 91**
- WARNING defects: 4

## Section Structure
- Page 2: 1. Introduction
- Page 3: 10   Seasonal–Time (HST) statistical model [3], which decomposes sensor read-
- Page 5: 1. A three-layer hybrid framework integrating physics-informed, data-
- Page 5: 3. A GAT-LSTM architecture that learns adaptive attention weights over
- Page 5: 5. Comprehensive experimental validation including ablation studies demon-
- Page 5: 2. Methodology
- Page 6: 25                            Displacement             T06
- Page 6: 0                            Temperature
- Page 8: 130      Each PINN is trained on the normal operating period by minimizing:
- Page 9: 140   LSTM.
- Page 11: 170   Differential settlement:. Displacement divergence between adjacent mono-
- Page 11: 185      The three layer scores are fused using Dempster–Shafer (DS) evidence
- Page 12: 1    X
- Page 13: 3. Results and Discussion
- Page 14: 2. Differential settlement (20 days): displacement drift in monolith 2
- Page 14: 3. Thermal crack (10 days): sudden temperature change at monolith 1
- Page 14: 4. Uplift increase (15 days): step increase in foundation piezometric
- Page 15: 260   LSTM, LSTM-AE, and the full framework) are run 5 times with different
- Page 16: 30      T01          T04          T07
- Page 21: 1. Interpretability: The framework provides failure-mode diagnosis (which
- Page 23: 2. Low false alarm rate: The physics constraints filter out statistically
- Page 23: 3. Uncertainty quantification: The DS belief–plausibility interval pro-
- Page 24: 20       PINN (Physics)
- Page 24: 100       HST               LSTM-AE
- Page 26: 2. Simplified physics: The PINN encodes 1D seepage and monotonic-
- Page 26: 3. Knowledge layer sensitivity: The fuzzy rule thresholds require
- Page 26: 4. Source reliability weights: The DS source reliabilities (wphys = 0.4,
- Page 27: 1. The PINN physics layer with spatial PDE residuals achieves strong
- Page 27: 2. The GAT-LSTM with learned attention weights achieves the best individual-
- Page 27: 385   Acknowledgements
- Page 29: 420       L. Yang, Physics-informed machine learning, Nature Reviews Physics 3
- Page 30: 435       Graph attention networks, in: International Conference on Learning
- Page 30: 445       Royal Statistical Society: Series B 30 (2) (1968) 205–247.

## Per-Page Analysis

| Page | White% | MaxGap% | Figures | Status |
|------|--------|---------|---------|--------|
| 1 | 96.0% | 17.4% | 0 | CRITICAL |
| 2 | 95.7% | 16.7% | 0 | CRITICAL |
| 3 | 95.2% | 17.0% | 0 | CRITICAL |
| 4 | 95.2% | 17.0% | 0 | CRITICAL |
| 5 | 96.0% | 16.7% | 0 | CRITICAL |
| 6 | 89.9% | 18.1% | 0 | CRITICAL |
| 7 | 96.6% | 21.3% | 0 | CRITICAL |
| 8 | 97.0% | 19.9% | 0 | CRITICAL |
| 9 | 97.0% | 16.7% | 0 | CRITICAL |
| 10 | 97.0% | 18.5% | 0 | CRITICAL |
| 11 | 96.9% | 23.1% | 0 | CRITICAL |
| 12 | 97.2% | 19.6% | 0 | CRITICAL |
| 13 | 95.7% | 16.7% | 0 | CRITICAL |
| 14 | 96.0% | 16.7% | 0 | CRITICAL |
| 15 | 95.7% | 16.3% | 0 | CRITICAL |
| 16 | 93.7% | 22.3% | 0 | CRITICAL |
| 17 | 96.3% | 16.3% | 0 | CRITICAL |
| 18 | 89.8% | 16.7% | 0 | CRITICAL |
| 19 | 90.6% | 18.0% | 0 | CRITICAL |
| 20 | 91.0% | 32.4% | 0 | CRITICAL |
| 21 | 96.3% | 17.3% | 0 | CRITICAL |
| 22 | 98.4% | 33.7% | 0 | CRITICAL |
| 23 | 96.3% | 16.7% | 0 | CRITICAL |
| 24 | 88.4% | 19.9% | 0 | CRITICAL |
| 25 | 95.6% | 21.7% | 0 | CRITICAL |
| 26 | 96.0% | 16.7% | 0 | CRITICAL |
| 27 | 96.4% | 16.7% | 0 | CRITICAL |
| 28 | 95.8% | 20.0% | 0 | CRITICAL |
| 29 | 95.7% | 16.7% | 0 | CRITICAL |
| 30 | 98.0% | 47.5% | 0 | CRITICAL |

## Defects Found

- **L1 CRITICAL**: Page 1 has 96.0% white space (threshold: 30%)
- **L1 CRITICAL**: Page 1 has a 16% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 1 has a 17% vertical gap at 83% from top (likely post-float white space)
- **L1 CRITICAL**: Page 2 has 95.7% white space (threshold: 30%)
- **L1 CRITICAL**: Page 2 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 2 has a 17% vertical gap at 83% from top (likely post-float white space)
- **L1 CRITICAL**: Page 3 has 95.2% white space (threshold: 30%)
- **L1 CRITICAL**: Page 3 has a 16% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 3 has a 17% vertical gap at 83% from top (likely post-float white space)
- **L1 CRITICAL**: Page 4 has 95.2% white space (threshold: 30%)
- **L1 CRITICAL**: Page 4 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 4 has a 17% vertical gap at 83% from top (likely post-float white space)
- **L1 CRITICAL**: Page 5 has 96.0% white space (threshold: 30%)
- **L1 CRITICAL**: Page 5 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 5 has a 15% vertical gap at 85% from top (likely post-float white space)
- **L1 CRITICAL**: Page 6 has 89.9% white space (threshold: 30%)
- **L1 CRITICAL**: Page 6 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 WARNING**: Page 6 has a 9% gap at 48% from top
- **L1 CRITICAL**: Page 6 has a 18% vertical gap at 82% from top (likely post-float white space)
- **L1 CRITICAL**: Page 7 has 96.6% white space (threshold: 30%)
- **L1 CRITICAL**: Page 7 has a 16% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 7 has a 21% vertical gap at 79% from top (likely post-float white space)
- **L1 CRITICAL**: Page 8 has 97.0% white space (threshold: 30%)
- **L1 CRITICAL**: Page 8 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 8 has a 20% vertical gap at 80% from top (likely post-float white space)
- **L1 CRITICAL**: Page 9 has 97.0% white space (threshold: 30%)
- **L1 CRITICAL**: Page 9 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 9 has a 15% vertical gap at 85% from top (likely post-float white space)
- **L1 CRITICAL**: Page 10 has 97.0% white space (threshold: 30%)
- **L1 CRITICAL**: Page 10 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 10 has a 19% vertical gap at 81% from top (likely post-float white space)
- **L1 CRITICAL**: Page 11 has 96.9% white space (threshold: 30%)
- **L1 CRITICAL**: Page 11 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 11 has a 23% vertical gap at 77% from top (likely post-float white space)
- **L1 CRITICAL**: Page 12 has 97.2% white space (threshold: 30%)
- **L1 CRITICAL**: Page 12 has a 20% vertical gap at 0% from top (likely post-float white space)
- **L1 WARNING**: Page 12 has a 10% gap at 57% from top
- **L1 CRITICAL**: Page 12 has a 18% vertical gap at 82% from top (likely post-float white space)
- **L1 CRITICAL**: Page 13 has 95.7% white space (threshold: 30%)
- **L1 CRITICAL**: Page 13 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 13 has a 16% vertical gap at 84% from top (likely post-float white space)
- **L1 CRITICAL**: Page 14 has 96.0% white space (threshold: 30%)
- **L1 CRITICAL**: Page 14 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 14 has a 15% vertical gap at 85% from top (likely post-float white space)
- **L1 CRITICAL**: Page 15 has 95.7% white space (threshold: 30%)
- **L1 CRITICAL**: Page 15 has a 16% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 15 has a 16% vertical gap at 84% from top (likely post-float white space)
- **L1 CRITICAL**: Page 16 has 93.7% white space (threshold: 30%)
- **L1 CRITICAL**: Page 16 has a 22% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 16 has a 21% vertical gap at 79% from top (likely post-float white space)
- **L1 CRITICAL**: Page 17 has 96.3% white space (threshold: 30%)
- **L1 CRITICAL**: Page 17 has a 16% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 17 has a 16% vertical gap at 84% from top (likely post-float white space)
- **L1 CRITICAL**: Page 18 has 89.8% white space (threshold: 30%)
- **L1 CRITICAL**: Page 18 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 18 has a 15% vertical gap at 85% from top (likely post-float white space)
- **L1 CRITICAL**: Page 19 has 90.6% white space (threshold: 30%)
- **L1 CRITICAL**: Page 19 has a 16% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 19 has a 18% vertical gap at 82% from top (likely post-float white space)
- **L1 CRITICAL**: Page 20 has 91.0% white space (threshold: 30%)
- **L1 CRITICAL**: Page 20 has a 32% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 20 has a 31% vertical gap at 69% from top (likely post-float white space)
- **L1 CRITICAL**: Page 21 has 96.3% white space (threshold: 30%)
- **L1 CRITICAL**: Page 21 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 21 has a 15% vertical gap at 85% from top (likely post-float white space)
- **L1 CRITICAL**: Page 22 has 98.4% white space (threshold: 30%)
- **L1 CRITICAL**: Page 22 has a 34% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 22 has a 16% vertical gap at 35% from top (likely post-float white space)
- **L1 CRITICAL**: Page 22 has a 32% vertical gap at 68% from top (likely post-float white space)
- **L1 CRITICAL**: Page 23 has 96.3% white space (threshold: 30%)
- **L1 CRITICAL**: Page 23 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 WARNING**: Page 23 has a 10% gap at 34% from top
- **L1 CRITICAL**: Page 23 has a 15% vertical gap at 85% from top (likely post-float white space)
- **L1 CRITICAL**: Page 24 has 88.4% white space (threshold: 30%)
- **L1 CRITICAL**: Page 24 has a 20% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 24 has a 20% vertical gap at 80% from top (likely post-float white space)
- **L1 CRITICAL**: Page 25 has 95.6% white space (threshold: 30%)
- **L1 CRITICAL**: Page 25 has a 22% vertical gap at 0% from top (likely post-float white space)
- **L1 WARNING**: Page 25 has a 14% gap at 47% from top
- **L1 CRITICAL**: Page 25 has a 20% vertical gap at 80% from top (likely post-float white space)
- **L1 CRITICAL**: Page 26 has 96.0% white space (threshold: 30%)
- **L1 CRITICAL**: Page 26 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 26 has a 16% vertical gap at 84% from top (likely post-float white space)
- **L1 CRITICAL**: Page 27 has 96.4% white space (threshold: 30%)
- **L1 CRITICAL**: Page 27 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 27 has a 15% vertical gap at 85% from top (likely post-float white space)
- **L1 CRITICAL**: Page 28 has 95.8% white space (threshold: 30%)
- **L1 CRITICAL**: Page 28 has a 16% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 28 has a 20% vertical gap at 80% from top (likely post-float white space)
- **L1 CRITICAL**: Page 29 has 95.7% white space (threshold: 30%)
- **L1 CRITICAL**: Page 29 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 29 has a 15% vertical gap at 85% from top (likely post-float white space)
- **L1 CRITICAL**: Page 30 has 98.0% white space (threshold: 30%)
- **L1 CRITICAL**: Page 30 has a 16% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 30 has a 48% vertical gap at 52% from top (likely post-float white space)

## Pages Needing Attention
The Editor should visually inspect these page images:

### Page 1
- Image: page_images/page_001.png
- **L1 CRITICAL**: Page 1 has 96.0% white space (threshold: 30%)
- **L1 CRITICAL**: Page 1 has a 16% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 1 has a 17% vertical gap at 83% from top (likely post-float white space)

### Page 2
- Image: page_images/page_002.png
- **L1 CRITICAL**: Page 2 has 95.7% white space (threshold: 30%)
- **L1 CRITICAL**: Page 2 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 2 has a 17% vertical gap at 83% from top (likely post-float white space)

### Page 3
- Image: page_images/page_003.png
- **L1 CRITICAL**: Page 3 has 95.2% white space (threshold: 30%)
- **L1 CRITICAL**: Page 3 has a 16% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 3 has a 17% vertical gap at 83% from top (likely post-float white space)

### Page 4
- Image: page_images/page_004.png
- **L1 CRITICAL**: Page 4 has 95.2% white space (threshold: 30%)
- **L1 CRITICAL**: Page 4 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 4 has a 17% vertical gap at 83% from top (likely post-float white space)

### Page 5
- Image: page_images/page_005.png
- **L1 CRITICAL**: Page 5 has 96.0% white space (threshold: 30%)
- **L1 CRITICAL**: Page 5 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 5 has a 15% vertical gap at 85% from top (likely post-float white space)

### Page 6
- Image: page_images/page_006.png
- **L1 CRITICAL**: Page 6 has 89.9% white space (threshold: 30%)
- **L1 CRITICAL**: Page 6 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 WARNING**: Page 6 has a 9% gap at 48% from top
- **L1 CRITICAL**: Page 6 has a 18% vertical gap at 82% from top (likely post-float white space)

### Page 7
- Image: page_images/page_007.png
- **L1 CRITICAL**: Page 7 has 96.6% white space (threshold: 30%)
- **L1 CRITICAL**: Page 7 has a 16% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 7 has a 21% vertical gap at 79% from top (likely post-float white space)

### Page 8
- Image: page_images/page_008.png
- **L1 CRITICAL**: Page 8 has 97.0% white space (threshold: 30%)
- **L1 CRITICAL**: Page 8 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 8 has a 20% vertical gap at 80% from top (likely post-float white space)

### Page 9
- Image: page_images/page_009.png
- **L1 CRITICAL**: Page 9 has 97.0% white space (threshold: 30%)
- **L1 CRITICAL**: Page 9 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 9 has a 15% vertical gap at 85% from top (likely post-float white space)

### Page 10
- Image: page_images/page_010.png
- **L1 CRITICAL**: Page 10 has 97.0% white space (threshold: 30%)
- **L1 CRITICAL**: Page 10 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 10 has a 19% vertical gap at 81% from top (likely post-float white space)

### Page 11
- Image: page_images/page_011.png
- **L1 CRITICAL**: Page 11 has 96.9% white space (threshold: 30%)
- **L1 CRITICAL**: Page 11 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 11 has a 23% vertical gap at 77% from top (likely post-float white space)

### Page 12
- Image: page_images/page_012.png
- **L1 CRITICAL**: Page 12 has 97.2% white space (threshold: 30%)
- **L1 CRITICAL**: Page 12 has a 20% vertical gap at 0% from top (likely post-float white space)
- **L1 WARNING**: Page 12 has a 10% gap at 57% from top
- **L1 CRITICAL**: Page 12 has a 18% vertical gap at 82% from top (likely post-float white space)

### Page 13
- Image: page_images/page_013.png
- **L1 CRITICAL**: Page 13 has 95.7% white space (threshold: 30%)
- **L1 CRITICAL**: Page 13 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 13 has a 16% vertical gap at 84% from top (likely post-float white space)

### Page 14
- Image: page_images/page_014.png
- **L1 CRITICAL**: Page 14 has 96.0% white space (threshold: 30%)
- **L1 CRITICAL**: Page 14 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 14 has a 15% vertical gap at 85% from top (likely post-float white space)

### Page 15
- Image: page_images/page_015.png
- **L1 CRITICAL**: Page 15 has 95.7% white space (threshold: 30%)
- **L1 CRITICAL**: Page 15 has a 16% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 15 has a 16% vertical gap at 84% from top (likely post-float white space)

### Page 16
- Image: page_images/page_016.png
- **L1 CRITICAL**: Page 16 has 93.7% white space (threshold: 30%)
- **L1 CRITICAL**: Page 16 has a 22% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 16 has a 21% vertical gap at 79% from top (likely post-float white space)

### Page 17
- Image: page_images/page_017.png
- **L1 CRITICAL**: Page 17 has 96.3% white space (threshold: 30%)
- **L1 CRITICAL**: Page 17 has a 16% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 17 has a 16% vertical gap at 84% from top (likely post-float white space)

### Page 18
- Image: page_images/page_018.png
- **L1 CRITICAL**: Page 18 has 89.8% white space (threshold: 30%)
- **L1 CRITICAL**: Page 18 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 18 has a 15% vertical gap at 85% from top (likely post-float white space)

### Page 19
- Image: page_images/page_019.png
- **L1 CRITICAL**: Page 19 has 90.6% white space (threshold: 30%)
- **L1 CRITICAL**: Page 19 has a 16% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 19 has a 18% vertical gap at 82% from top (likely post-float white space)

### Page 20
- Image: page_images/page_020.png
- **L1 CRITICAL**: Page 20 has 91.0% white space (threshold: 30%)
- **L1 CRITICAL**: Page 20 has a 32% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 20 has a 31% vertical gap at 69% from top (likely post-float white space)

### Page 21
- Image: page_images/page_021.png
- **L1 CRITICAL**: Page 21 has 96.3% white space (threshold: 30%)
- **L1 CRITICAL**: Page 21 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 21 has a 15% vertical gap at 85% from top (likely post-float white space)

### Page 22
- Image: page_images/page_022.png
- **L1 CRITICAL**: Page 22 has 98.4% white space (threshold: 30%)
- **L1 CRITICAL**: Page 22 has a 34% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 22 has a 16% vertical gap at 35% from top (likely post-float white space)
- **L1 CRITICAL**: Page 22 has a 32% vertical gap at 68% from top (likely post-float white space)

### Page 23
- Image: page_images/page_023.png
- **L1 CRITICAL**: Page 23 has 96.3% white space (threshold: 30%)
- **L1 CRITICAL**: Page 23 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 WARNING**: Page 23 has a 10% gap at 34% from top
- **L1 CRITICAL**: Page 23 has a 15% vertical gap at 85% from top (likely post-float white space)

### Page 24
- Image: page_images/page_024.png
- **L1 CRITICAL**: Page 24 has 88.4% white space (threshold: 30%)
- **L1 CRITICAL**: Page 24 has a 20% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 24 has a 20% vertical gap at 80% from top (likely post-float white space)

### Page 25
- Image: page_images/page_025.png
- **L1 CRITICAL**: Page 25 has 95.6% white space (threshold: 30%)
- **L1 CRITICAL**: Page 25 has a 22% vertical gap at 0% from top (likely post-float white space)
- **L1 WARNING**: Page 25 has a 14% gap at 47% from top
- **L1 CRITICAL**: Page 25 has a 20% vertical gap at 80% from top (likely post-float white space)

### Page 26
- Image: page_images/page_026.png
- **L1 CRITICAL**: Page 26 has 96.0% white space (threshold: 30%)
- **L1 CRITICAL**: Page 26 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 26 has a 16% vertical gap at 84% from top (likely post-float white space)

### Page 27
- Image: page_images/page_027.png
- **L1 CRITICAL**: Page 27 has 96.4% white space (threshold: 30%)
- **L1 CRITICAL**: Page 27 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 27 has a 15% vertical gap at 85% from top (likely post-float white space)

### Page 28
- Image: page_images/page_028.png
- **L1 CRITICAL**: Page 28 has 95.8% white space (threshold: 30%)
- **L1 CRITICAL**: Page 28 has a 16% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 28 has a 20% vertical gap at 80% from top (likely post-float white space)

### Page 29
- Image: page_images/page_029.png
- **L1 CRITICAL**: Page 29 has 95.7% white space (threshold: 30%)
- **L1 CRITICAL**: Page 29 has a 17% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 29 has a 15% vertical gap at 85% from top (likely post-float white space)

### Page 30
- Image: page_images/page_030.png
- **L1 CRITICAL**: Page 30 has 98.0% white space (threshold: 30%)
- **L1 CRITICAL**: Page 30 has a 16% vertical gap at 0% from top (likely post-float white space)
- **L1 CRITICAL**: Page 30 has a 48% vertical gap at 52% from top (likely post-float white space)

## Automated Recommendations
- Page 1: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 2: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 3: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 4: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 5: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 6: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 7: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 8: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 9: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 10: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 11: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 12: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 13: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 14: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 15: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 16: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 17: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 18: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 19: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 20: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 21: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 22: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 23: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 24: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 25: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 26: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 27: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 28: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 29: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.
- Page 30: Reduce white space by adjusting float placement ([htbp]) or reducing figure size.