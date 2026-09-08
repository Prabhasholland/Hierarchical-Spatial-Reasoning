# ARC Reasoning Agent

> A research project for the **ARC Prize 2026 Paper Track**, targeting the ARC-AGI-2 benchmark.

## Research Objective

This project aims to develop an **original reasoning system** for the Abstraction and Reasoning Corpus (ARC), with a focus on:

1. **Understanding** the fundamental reasoning challenges in ARC tasks
2. **Building** measurable baselines and conducting reproducible experiments
3. **Developing** a novel or under-explored approach to ARC reasoning
4. **Submitting** to the ARC-AGI-2 Kaggle competition and documenting our approach for the Paper Track

We explicitly prioritize **scientific rigor over leaderboard rank**. Our goal is to produce a clear, honest research contribution that advances understanding of abstract reasoning in AI systems.

## What is ARC?

The **Abstraction and Reasoning Corpus** (ARC), created by François Chollet, is a benchmark for measuring fluid intelligence — the ability to solve novel problems using only basic cognitive priors (objectness, geometry, counting, topology). Each task provides a few input→output grid examples, and the solver must infer the transformation rule and apply it to unseen test inputs.

- **ARC-AGI-2**: Static grid-based reasoning (1,000+ tasks, JSON format)
- **ARC-AGI-3**: Interactive turn-based reasoning (new in 2026)
- **Paper Track**: Documentation and theory of approaches ($450K prize pool)

See [`docs/ARC_RESEARCH.md`](docs/ARC_RESEARCH.md) for our full research report.

## Project Structure

```
arc-reasoning-agent/
├── docs/                    # Research documentation
│   ├── ARC_RESEARCH.md      # Comprehensive research report
│   └── RESEARCH_ROADMAP.md  # Phased research plan
├── data/                    # ARC datasets (downloaded separately)
├── src/                     # Source code
│   ├── data/                # Data loading and visualization
│   ├── solvers/             # Solver implementations
│   └── evaluation/          # Evaluation metrics and harness
├── tests/                   # Unit tests
├── experiments/             # Experiment configurations and logs
├── notebooks/               # Jupyter notebooks for analysis
├── results/                 # Experiment results
├── scripts/                 # Utility scripts
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Getting Started

### Prerequisites

- Python 3.10+
- Git

### Setup

```bash
# Clone this repository
git clone <repo-url>
cd arc-reasoning-agent

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download ARC datasets
python scripts/download_data.py
```

### Running Tests

```bash
pytest tests/ -v
```

## Research Roadmap

| Phase | Timeline | Goal |
|:--|:--|:--|
| **Phase 0: Foundation** | Aug 21–28 | Data pipeline, visualization, evaluation harness |
| **Phase 1: Baselines** | Aug 28 – Sep 11 | Establish performance floors; LLM program synthesis |
| **Phase 2: Improvements** | Sep 11 – Oct 2 | Targeted improvements based on failure analysis |
| **Phase 3: Integration** | Oct 2 – Oct 16 | Unified solver; identify research contribution |
| **Phase 4: Submission** | Oct 16 – Oct 30 | Kaggle code submission |
| **Phase 5: Paper** | Oct 30 – Nov 9 | Paper Track writeup |

See [`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md) for the full plan.

## Current Status

- [x] Research complete — ARC ecosystem, approaches, and gap analysis
- [x] Competition target selected — ARC-AGI-2 + Paper Track
- [x] Project structure created
- [ ] Data pipeline implementation
- [ ] Baseline solvers
- [ ] First experiments

## Key Decisions

| Decision | Rationale |
|:--|:--|
| Target ARC-AGI-2 | Better data availability, accessibility, and research maturity than ARC-AGI-3 |
| Paper Track focus | Rewards theory and understanding, not just accuracy |
| Start with LLM program synthesis baseline | Proven approach (~40-50% on ARC-AGI-1); good experimental platform |
| Open-weight models first | Avoid cost/dependency on commercial APIs |

## License

This project will be open-sourced under a permissive license (MIT-0 or CC0) to meet ARC Prize eligibility requirements.

## References

- [ARC Prize 2026](https://arcprize.org)
- [ARC-AGI-2 Dataset](https://github.com/arcprize/ARC-AGI-2)
- [Chollet, "On the Measure of Intelligence" (2019)](https://arxiv.org/abs/1911.01547)
