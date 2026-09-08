# ARC Prize 2026 — Research Roadmap

> **Target**: ARC-AGI-2 (primary) + Paper Track (submission)  
> **Deadline**: November 9, 2026  
> **Current Date**: August 21, 2026  
> **Remaining Time**: ~11 weeks

---

## Phase 0: Foundation (Week 1 — Aug 21–28)
**Goal**: Understand the problem space and establish infrastructure.

### Tasks
- [x] Research ARC Prize 2026 ecosystem
- [x] Compare ARC-AGI-2 vs ARC-AGI-3
- [x] Survey existing approaches and prior art
- [x] Choose target competition (ARC-AGI-2 + Paper Track)
- [x] Create project structure
- [ ] Download ARC-AGI-1 and ARC-AGI-2 datasets
- [ ] Implement data loader and visualizer
- [ ] Manually inspect 50+ tasks across all categories
- [ ] Create task taxonomy and category annotations
- [ ] Set up evaluation harness (exact-match accuracy)
- [ ] Create validation split from training set (e.g., 800 train / 200 val)

### Deliverables
- `docs/ARC_RESEARCH.md` — Research findings ✅
- `docs/RESEARCH_ROADMAP.md` — This roadmap ✅
- Working data pipeline
- Task visualization tools
- Evaluation framework

---

## Phase 1: Baselines (Weeks 2–3 — Aug 28 – Sep 11)
**Goal**: Establish performance floors and understand failure patterns.

### Baseline Solvers
- [ ] **Trivial baselines**: Identity (return input), constant (most common color), random
- [ ] **Heuristic baselines**: Color substitution, grid copy, size-preserving transforms
- [ ] **LLM direct prompting**: Zero-shot and few-shot prompting (GPT-4o, Claude, Gemini)
- [ ] **LLM program synthesis (Greenblatt-style)**: Generate Python programs, execute, verify

### Analysis
- [ ] Categorize tasks by difficulty (solve rate across baselines)
- [ ] Identify "easy" tasks (solved by simple heuristics)
- [ ] Identify "hard" tasks (unsolved by any baseline)
- [ ] Analyze failure modes per category (object, spatial, color, etc.)
- [ ] Compute baseline ceiling (union of all baseline solve sets)

### Experiments
1. **EXP-01**: LLM program synthesis — vary model, number of candidates (k), temperature
2. **EXP-02**: Prompting strategies — grid representation format (JSON, ASCII art, natural language)
3. **EXP-03**: Few-shot example selection — which training examples help most?

### Deliverables
- Baseline accuracy numbers on training set validation split
- Per-category failure analysis
- Experiment logs in `experiments/`
- Results in `results/`

---

## Phase 2: Targeted Improvements (Weeks 4–6 — Sep 11 – Oct 2)
**Goal**: Develop and test targeted improvements to address baseline weaknesses.

### Research Directions (select 2–3 based on Phase 1 findings)

#### Direction A: Enhanced Program Synthesis
- [ ] Design a richer DSL for ARC transformations
- [ ] Implement constraint-based search (backward reasoning)
- [ ] Build a "program repair" loop using execution feedback
- [ ] Experiment with evolutionary program synthesis

#### Direction B: Object-Centric Perception
- [ ] Implement grid segmentation (connected components, color regions)
- [ ] Build object graph representations
- [ ] Test graph-based reasoning for object manipulation tasks
- [ ] Evaluate impact on spatial and object reasoning tasks

#### Direction C: Structured Spatial Representations
- [ ] Design 2D-aware encodings (beyond tokenized flat grids)
- [ ] Implement symmetry detection and completion
- [ ] Build geometric transformation primitives
- [ ] Test on symmetry and transformation task categories

#### Direction D: Hierarchical Decomposition
- [ ] Develop sub-task decomposition strategies
- [ ] Build a library of reusable sub-routines
- [ ] Test compositional assembly on multi-step tasks
- [ ] Measure impact on compositional reasoning failures

### Experiments
4. **EXP-04**: DSL expressiveness — which primitives solve the most tasks?
5. **EXP-05**: Object segmentation accuracy — how well can we extract objects?
6. **EXP-06**: Representation format ablation — grid encoding impact on solve rate
7. **EXP-07**: Refinement loop depth — how many iterations of refine improve results?

### Deliverables
- Implementation of 2–3 targeted improvements
- Comparative experiment results
- Updated accuracy numbers
- Analysis of which improvements help which task categories

---

## Phase 3: Integration & Novel Method (Weeks 7–8 — Oct 2 – Oct 16)
**Goal**: Combine successful components into a unified solver and identify our research contribution.

### Tasks
- [ ] Integrate best-performing components from Phase 2
- [ ] Design ensemble/routing strategy (which solver for which task?)
- [ ] Implement task difficulty estimation (meta-learning)
- [ ] Optimize for compute budget (fit within 9-hour Kaggle limit)
- [ ] Run comprehensive evaluation on full training + eval sets
- [ ] Identify and articulate our research contribution
- [ ] Begin literature review for novelty verification

### Experiments
8. **EXP-08**: Ensemble vs. single solver — does combining methods help?
9. **EXP-09**: Task routing — can we predict which solver will succeed?
10. **EXP-10**: Compute budget allocation — optimal distribution across tasks

### Deliverables
- Integrated solver system
- Performance on public evaluation set
- Clear articulation of research contribution
- Draft of theoretical framework

---

## Phase 4: Kaggle Submission (Weeks 9–10 — Oct 16 – Oct 30)
**Goal**: Package and submit to ARC-AGI-2 Kaggle competition.

### Tasks
- [ ] Package solver as Kaggle notebook
- [ ] Test in Kaggle environment (no internet, compute limits)
- [ ] Optimize for 9-hour time budget
- [ ] Submit initial code submission
- [ ] Iterate based on leaderboard feedback
- [ ] Finalize code submission before Oct 26 team deadline

### Deliverables
- Working Kaggle submission
- Leaderboard score
- Submission notebook

---

## Phase 5: Paper Track Writeup (Weeks 10–11 — Oct 30 – Nov 9)
**Goal**: Write and submit Paper Track documentation.

### Paper Structure (aligned with evaluation criteria)

1. **Introduction**: Problem statement, motivation, our approach
2. **Background**: ARC benchmark, core knowledge priors, prior work
3. **Method**: Our architecture, algorithms, and design decisions
4. **Theory**: Why our approach works; what principles underlie it
5. **Experiments**: Reproducible results, ablation studies
6. **Analysis**: Per-category performance, failure analysis, insights
7. **Discussion**: Universality, limitations, future work
8. **Conclusion**: Summary of contributions

### Evaluation Criteria Alignment

| Criterion | Strategy |
|:--|:--|
| **Accuracy** | Maximize leaderboard score via compute-efficient methods |
| **Universality** | Discuss generalization beyond ARC; connect to broader AI |
| **Progress** | Show concrete improvements over baselines; identify path to 85% |
| **Theory** | Provide principled explanation of why our methods work |
| **Completeness** | Thorough documentation with code, experiments, ablations |
| **Novelty** | Clearly distinguish our contribution from prior work |

### Deliverables
- Paper Track writeup (Kaggle format)
- Media gallery
- Public notebook
- Open-sourced code repository

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|:--|:--|:--|:--|
| Time pressure (11 weeks) | High | High | Strict phase gates; cut scope if behind |
| Compute limitations | Medium | High | Prioritize open-weight models; optimize early |
| No novelty found | Medium | High | Focus on rigorous analysis even without new method |
| ARC-AGI-2 leaderboard saturation | Low | Medium | Paper Track rewards theory, not just accuracy |
| Kaggle environment issues | Medium | Medium | Test in Kaggle environment by Week 9 |
| Dataset overfitting | Medium | Medium | Strict val split; report results honestly |

---

## Success Metrics

### Minimum Viable Outcome
- [ ] Reproducible baseline with measured accuracy
- [ ] At least 2 ablation experiments with clear conclusions
- [ ] Paper Track submission with honest results

### Target Outcome
- [ ] Solver achieving >20% on ARC-AGI-2 public eval
- [ ] Novel or under-explored technique with measurable benefit
- [ ] Well-written paper scoring ≥3.5/5 on evaluation rubric

### Stretch Outcome
- [ ] Solver achieving >40% on ARC-AGI-2 public eval
- [ ] Research contribution that advances understanding of ARC reasoning
- [ ] Paper scoring ≥4.5/5 (eligible for Outstanding Papers pool)
- [ ] Preliminary ARC-AGI-3 agent as future work

---

## Decision Log

| Date | Decision | Rationale |
|:--|:--|:--|
| 2026-08-21 | Target ARC-AGI-2 + Paper Track | Better accessibility, data availability, and research maturity |
| 2026-08-21 | Start with LLM program synthesis baseline | Proven approach (~40–50%); good foundation for improvements |
| 2026-08-21 | Defer ARC-AGI-3 to stretch goal | Too new, <1% AI performance, requires separate research track |
