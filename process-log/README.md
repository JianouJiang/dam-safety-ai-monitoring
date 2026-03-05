# Process Log — Dam Safety AI Paper

## Research Workflow Overview

This paper was produced using the Paper Factory autonomous agent system, following AIDER journal's open-process requirements.

### AI Tools Used
- **Claude Opus 4.6** (Anthropic) — Primary Worker agent: code generation, simulation execution, LaTeX writing, figure generation
- **Claude Opus 4.6** — Judge agent (Charlie Munger persona): critical scientific review, anti-shortcut enforcement
- **Claude Opus 4.6** — Statistician agent (R.A. Fisher persona): statistical rigor review
- **Claude Opus 4.6** — Editor agent (William Strunk Jr. persona): writing quality, LaTeX compliance
- **GPT-5.2** (OpenAI Codex) — Illustrator agent (Edward Tufte persona): figure quality review
- **Qwen 2.5 7B** (local Ollama) — Zero-token coordination between agents

### Agent Workflow
The paper was produced through an iterative loop:
```
Worker → Judge → Worker → Statistician → Worker → Editor → Worker → Illustrator → repeat
```

Each cycle produces reviews that the Worker addresses in the next iteration. The process continues until all reviewer scores reach ≥ 8/10.

### Human Decisions
All significant human interventions are logged in `human-decisions/decisions.md`.

### AI Session Logs
Full agent session logs are stored in `ai-sessions/` and in `../logs/` (runtime logs from each agent).
