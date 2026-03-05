# Reproducibility Checklist — Dam Safety AI

## Code
- [ ] All dependencies listed in `codes/requirements.txt`
- [ ] Setup instructions in `codes/README.md`
- [ ] Code runs on a clean environment without modification
- [ ] Random seeds fixed for deterministic output

## Data
- [ ] All data necessary to reproduce results is available
- [ ] Data format and source documented in `data/README.md`
- [ ] Data license specified

## Results
- [ ] `results/reproduce.sh` exists and regenerates all figures/tables
- [ ] Script exits with code 0 on success
- [ ] Generated outputs match paper's figures and tables
- [ ] Random seeds documented in code

## Process Log
- [ ] `process-log/README.md` describes research workflow
- [ ] AI session logs included in `process-log/ai-sessions/`
- [ ] Human decisions documented in `process-log/human-decisions/`
- [ ] All AI tools and versions disclosed

## Licensing
- [ ] Paper: CC-BY 4.0
- [ ] Code: MIT License
- [ ] Data: CC-BY 4.0 or CC0
