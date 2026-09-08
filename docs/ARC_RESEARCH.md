# ARC Prize 2026 — Research Report

> **Date**: August 2026  
> **Status**: Active research — pre-implementation phase  
> **Authors**: ARC Reasoning Agent Research Team

---

## 1. ARC Prize 2026 Ecosystem Overview

### 1.1 What Is ARC?

The **Abstraction and Reasoning Corpus (ARC)**, created by François Chollet, is a benchmark designed to measure *fluid intelligence* — the ability to solve novel problems without relying on memorized knowledge. Each task provides a small number of input→output grid examples. The solver must infer the underlying transformation rule and apply it to unseen test inputs, producing **pixel-perfect** output grids.

ARC is grounded in **core knowledge priors** — cognitive building blocks universally shared by humans from early childhood (per Elizabeth Spelke's developmental psychology research):

| Core Knowledge Prior | Description |
|:--|:--|
| **Objectness** | Objects are persistent, bounded entities with properties |
| **Goal-directedness** | Recognition of intent and purposeful action |
| **Numbers & Counting** | Elementary number sense, ordering |
| **Basic Geometry** | Symmetry, rotation, translation, shapes |
| **Basic Topology** | Connectivity, inside/outside, containment |

### 1.2 Competition Tracks (2026)

The ARC Prize 2026 consists of **three tracks** with a total prize pool of **$2,000,000**:

#### ARC-AGI-2 — Static Reasoning ($700K)

- **Format**: Static grid-based input→output puzzles (same format as ARC-AGI-1)
- **Goal**: ≥85% accuracy on private evaluation set
- **Prizes**: $275K progress prizes (top 8) + $275K grand prize + $150K accuracy bonus (top 5 crossing 85%)
- **Status**: Becoming "saturated" — some frontier models exceed 90% with high compute

#### ARC-AGI-3 — Interactive Reasoning ($850K)

- **Format**: Interactive, turn-based environments (video-game-like)
- **Goal**: Agent must explore, infer hidden rules, and solve without instructions
- **Scoring**: RHAE (Relative Human Action Efficiency) — penalizes inefficiency; hard cutoff at 5× human actions
- **Prizes**: $700K grand prize (100% score) + $75K top scores + $75K milestones
- **Status**: Most frontier AI systems struggle to cross 1% — massive gap vs. humans

#### Paper Track ($450K)

- **Format**: Written documentation of approach linked to a Kaggle code submission
- **Evaluation**: 0–5 scale across 6 criteria (Accuracy, Universality, Progress, Theory, Completeness, Novelty)
- **Prizes**: $75K top papers + $375K outstanding papers pool (>4.5/5)
- **Deadline**: November 9, 2026
- **Requirement**: Must link to a real code submission in ARC-AGI-2 or ARC-AGI-3

### 1.3 Key Deadlines

| Date | Event |
|:--|:--|
| March 25, 2026 | Competition start |
| June 30, 2026 | Milestone #1 (ARC-AGI-3) |
| September 30, 2026 | Milestone #2 (ARC-AGI-3) |
| October 26, 2026 | Entry / team merger deadline |
| November 2–9, 2026 | Final submission deadline |
| December 4, 2026 | Winners announcement |

### 1.4 Requirements for Prize Eligibility

1. **Open source**: All code and methods under permissive license (CC0, MIT-0, Apache-2.0)
2. **No internet**: Kaggle evaluation runs offline
3. **Reproducibility**: Methods must be reproducible
4. **Paper Track**: Writeup + media gallery + public notebook, linked to code submission

---

## 2. ARC-AGI-2 vs. ARC-AGI-3: Technical Comparison

| Dimension | ARC-AGI-2 | ARC-AGI-3 |
|:--|:--|:--|
| **Format** | Static grid puzzles (JSON) | Interactive turn-based environments |
| **Input** | 3–5 example I/O pairs + test input | Environment frames (JSON, up to 64×64, values 0–15) |
| **Instructions** | Implicit via examples | None — agent must discover rules and goals |
| **Output** | Complete output grid | Sequence of actions (RESET, ACTION1–7) |
| **Scoring** | Exact match accuracy (2 attempts) | RHAE — efficiency² with 5× human cutoff |
| **Grid size** | 1×1 to 30×30 (values 0–9) | Up to 64×64 (values 0–15) |
| **Evaluation** | 110 private tasks | 110 private games |
| **Time limit** | 9-hour notebook | 9-hour notebook |
| **Human baseline** | ~95% (estimated) | 100% by design |
| **AI SOTA** | >90% (high compute) | <1% (frontier models) |
| **Dataset** | 1000 train + 120 public eval | Public game files + starter kit |
| **Paradigm** | Pattern recognition + transformation | Exploration + hypothesis-driven learning |

### Key Insight

ARC-AGI-2 tests whether a model can **correctly apply complex rules** to static visual puzzles. ARC-AGI-3 tests whether a model can **behave like an intelligent agent** — exploring, hypothesizing, and acting in novel environments.

---

## 3. Dataset Structure

### 3.1 ARC-AGI-2 Task Format

```json
{
  "train": [
    {
      "input": [[0, 1, 0], [0, 1, 0], [0, 0, 0]],
      "output": [[0, 2, 0], [0, 2, 0], [0, 0, 0]]
    }
  ],
  "test": [
    {
      "input": [[0, 0, 1], [0, 0, 1], [0, 0, 0]],
      "output": [[0, 0, 2], [0, 0, 2], [0, 0, 0]]
    }
  ]
}
```

- **Grids**: 2D arrays of integers 0–9 (colors)
- **Dimensions**: 1×1 to 30×30
- **Training set**: 1,000 tasks
- **Public evaluation set**: 120 tasks
- **Private sets**: Semi-private (commercial model testing) + fully private (competition)
- **Source**: [github.com/arcprize/ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2)

### 3.2 ARC-AGI-3 Environment Format

- Agent receives **frames** as JSON objects (current game state)
- Grids up to 64×64 with values 0–15
- Agent responds with **actions**: RESET, ACTION1–ACTION7
- Starter kit: [github.com/arcprize/ARC-AGI-3-Kaggle-Starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter)
- Local development via `arc-agi` Python package

---

## 4. Publicly Available Successful Approaches

### 4.1 Taxonomy of ARC Approaches

| Category | Description | Status |
|:--|:--|:--|
| **Discrete Program Search (DSL)** | Enumerate programs in a hand-crafted DSL | Established |
| **LLM-Guided Program Synthesis** | LLM generates candidate Python/DSL programs | Established |
| **Neural/Transductive** | Direct input→output mapping via neural nets | Established |
| **Test-Time Training (TTT)** | Fine-tune per task at inference time | Established |
| **Evolutionary Program Synthesis** | Evolve programs via mutation + selection | Established |
| **Refinement Loops** | Iterative propose→verify→feedback→refine | Established (dominant paradigm) |
| **Neuro-Symbolic Hybrid** | LLM generation + symbolic DSL verification | Established |
| **Tiny Recursive Models** | Small recursive networks with full backprop | Recent (2025 Paper Track winner) |

### 4.2 Notable Solutions

#### Icecuber (2020) — Discrete Program Search

- **Author**: Johan S. Wind
- **Approach**: Greedy enumerative search over a DSL of 142 unary functions (42 base functions)
- **Result**: 20% on private test set
- **Significance**: Demonstrated that structured, compositional search is viable for ARC. Remained competitive baseline for years.

#### Ryan Greenblatt (2024) — LLM-Guided Program Synthesis

- **Approach**: Used GPT-4o to generate thousands of candidate Python programs per task, verified against training pairs, iteratively refined
- **Result**: ~42–50% accuracy
- **Key insight**: Log-linear relationship between number of programs generated (k) and accuracy — more compute → more solutions

#### MindsAI / Tufa Labs (2025) — Test-Time Training Pipeline

- **Approach**: Per-task test-time fine-tuning + augmentation ensembles + tokenizer dropout + pretraining tricks
- **Result**: 3rd place (15.42% on ARC-AGI-2 private leaderboard)
- **Significance**: Showed that heavy engineering at inference time can extract strong performance from adapted models

#### Tiny Recursive Model (TRM) — 2025 Paper Track 1st Place

- **Author**: Alexia Jolicoeur-Martineau
- **Approach**: 7M-parameter recursive neural network with full backpropagation through recursion
- **Mechanism**: Decision-then-revision loop — iteratively updates latent state z and answer y for N steps
- **Key insight**: Algorithmic structure (recursive refinement) matters more than model scale for abstract reasoning
- **Significance**: Demonstrated that "zero-pretraining" tiny architectures can compete by encoding task logic into weights at test time

#### "The Duck" / Tufa Labs (2026) — ARC-AGI-3 Milestone 1 Winner

- **Approach**: Qwen 3.6 27B FP8 with Python REPL interaction; treats each game as interactive coding problem
- **Perception**: Multiple formats (rendered images, ASCII grids, segmented regions)
- **Strategy**: "Infinite play via eviction" — continuously pops oldest messages to manage context window
- **Significance**: First successful approach to ARC-AGI-3; showed LLM+REPL can handle interactive environments

### 4.3 The "Refinement Loop" Paradigm (2025–2026)

The defining development: iterative, per-task program optimization guided by feedback signals.

```
┌─────────────────────────────────────────────┐
│                REFINEMENT LOOP              │
│                                             │
│  1. PROPOSE  → Generate candidate program   │
│  2. EXECUTE  → Run against training pairs   │
│  3. VERIFY   → Check output correctness     │
│  4. FEEDBACK → Extract error signal         │
│  5. REFINE   → Update program/approach      │
│  6. REPEAT   → Until correct or budget out  │
│                                             │
└─────────────────────────────────────────────┘
```

Variants include:
- **Evolutionary**: Mutate and select programs
- **LLM-guided**: Use LLM to generate and debug programs
- **Gradient-based**: Test-time training with backpropagation
- **Hybrid**: Combine multiple feedback mechanisms

---

## 5. Major Categories of ARC Reasoning Problems

### 5.1 Problem Category Taxonomy

| Category | Description | Example Tasks | Estimated Frequency |
|:--|:--|:--|:--|
| **Object Reasoning** | Identify, track, and manipulate discrete objects in grids | Object extraction, movement, copy | Very High |
| **Spatial Reasoning** | Understand relative positions, directions, distances | Alignment, adjacency, path-finding | Very High |
| **Color Reasoning** | Map, swap, or conditionally assign colors | Color replacement, fill by rule | High |
| **Counting** | Count objects, cells, or occurrences to determine behavior | Count-based fill, size-dependent rules | Medium |
| **Symmetry** | Detect, complete, or apply symmetry operations | Mirror, reflect, rotational symmetry | High |
| **Transformation** | Apply geometric transforms to grids or objects | Rotate, scale, translate, flip | High |
| **Compositional Reasoning** | Combine multiple primitive operations into a pipeline | Multi-step transformations | High |
| **Multi-Step Reasoning** | Require sequential application of inferred rules | Iterative growth, cascading effects | Medium-High |
| **Abstraction** | Infer abstract concepts from concrete examples | Pattern generalization, rule induction | Medium-High |
| **Program Synthesis** | The task is effectively to "write a program" that transforms input→output | Complex conditional logic | High |

### 5.2 Cross-Cutting Dimensions

Many ARC tasks span multiple categories simultaneously. Important cross-cutting dimensions include:

- **Foreground/Background separation**: Distinguishing objects from background
- **Grid topology**: Whether the grid wraps, has borders, or contains sub-grids
- **Variable output size**: Output grid dimensions may differ from input
- **Conditional logic**: Rules that apply differently based on context
- **Occlusion and layering**: Objects that overlap or obscure each other
- **Implicit vs. explicit rules**: Whether the transformation is directly observable or requires inference

---

## 6. Weaknesses in Common Approaches — Research Opportunities

### 6.1 Known Weaknesses

| Weakness | Affected Approaches | Potential Research Direction |
|:--|:--|:--|
| **Compositional generalization failure** | All approaches show 2–3× performance drop from ARC-AGI-1 → ARC-AGI-2 | Better compositional representations; modular program construction |
| **Output size prediction** | LLM-based and neural approaches struggle when output dimensions differ from input | Explicit size-prediction module; constraint-based reasoning |
| **Spatial representation** | LLMs lack native 2D spatial reasoning; tokenize grids poorly | Structured spatial encodings; vision transformer adaptations |
| **Brute-force scaling plateau** | Log-linear scaling eventually hits diminishing returns | Smarter search heuristics; abstraction-guided pruning |
| **Few-shot learning inefficiency** | Models need many examples; ARC provides only 3–5 | Meta-learning; analogical reasoning; inductive bias |
| **Knowledge-dependent overfitting** | Large pretrained models apply irrelevant prior knowledge | Smaller, task-specific architectures; zero-pretraining approaches |
| **Verification beyond training pairs** | Programs may overfit to few training examples but fail on test | Stronger program generalization; invariant discovery |
| **Multi-step reasoning chains** | Models struggle with tasks requiring >3 sequential operations | Explicit planning; hierarchical decomposition |
| **Object segmentation** | Identifying individual objects in cluttered grids is error-prone | Object-centric representations; graph-based models |
| **Abstract concept formation** | Forming reusable abstractions from examples is fundamentally hard | Category theory–inspired approaches; analogy engines |

### 6.2 Gap Analysis: Where the Field Stands (August 2026)

**Well-covered areas** (many existing solutions):
- LLM-guided code generation
- Evolutionary program search
- Test-time training
- DSL-based enumerative search

**Under-explored areas** (potential research opportunities):
- Object-centric perception modules (graph neural networks for ARC)
- Explicit spatial reasoning primitives (beyond tokenized grids)
- Hierarchical program decomposition (breaking complex tasks into sub-tasks)
- Constraint propagation and logic programming (Prolog-style inference)
- World-model learning for interactive (ARC-AGI-3) environments
- Curriculum learning and task difficulty estimation
- Analogy-based transfer between similar tasks
- Hybrid architectures that combine perception, reasoning, and synthesis

> [!IMPORTANT]
> **Status**: The items listed as "under-explored" are based on our survey of publicly available research. We have NOT yet verified that these areas lack prior work — this requires deeper literature review before claiming novelty.

---

## 7. Relevant Research (2024–2026)

### 7.1 Key Papers and Resources

| Reference | Year | Key Contribution | Category |
|:--|:--|:--|:--|
| Chollet, "On the Measure of Intelligence" | 2019 | Defines ARC benchmark and intelligence measurement theory | Foundational |
| Icecuber (Johan S. Wind) | 2020 | First competitive DSL-based program search | Established |
| Ryan Greenblatt, LLM program synthesis | 2024 | Demonstrated LLM-guided generation + verification | Established |
| MindsAI, test-time training pipeline | 2025 | Heavy test-time adaptation with augmentation ensembles | Established |
| Tiny Recursive Model (Jolicoeur-Martineau) | 2025 | 7M-param recursive model with full backprop; Paper Track winner | Recent |
| "The Duck" (Tufa Labs) | 2026 | First ARC-AGI-3 milestone winner; LLM + REPL agent | Recent |
| ARC-AGI Living Survey | Ongoing | Comprehensive survey of 80+ approaches | Reference |
| CodeARC benchmark | 2025–2026 | Interactive evaluation via oracle queries and iterative debugging | Reference |

### 7.2 Research Trends

1. **Refinement loops as the dominant paradigm**: Nearly all top systems use some form of iterative propose-verify-refine
2. **Test-time compute over pretraining scale**: Spending more compute at inference (per task) yields better results than larger pretrained models
3. **Program synthesis over direct prediction**: Generating and verifying code is more reliable than neural end-to-end prediction
4. **Efficiency as a constraint**: ARC-AGI-3's RHAE scoring explicitly penalizes brute-force; future benchmarks will likely continue this trend
5. **Small models can compete**: TRM showed that 7M parameters with the right architecture can match much larger models

---

## 8. Classification of Ideas by Evidence Level

### Established (Strong evidence, multiple successful implementations)

- Refinement loops (propose → verify → refine)
- LLM-guided program synthesis with execution feedback
- Test-time training / fine-tuning
- DSL-based enumerative program search
- Augmentation ensembles for robustness

### Promising (Some evidence, limited implementations)

- Recursive/iterative neural architectures (TRM)
- Object-centric perception + symbolic reasoning pipelines
- Constraint-based search (SAT/SMT solvers for ARC)
- Curriculum learning to order tasks by difficulty
- Graph-based representations of ARC grids

### Hypotheses (Theoretically motivated, require literature verification)

- Hierarchical program decomposition with sub-routine libraries may improve compositional generalization
- Analogy-based transfer learning between structurally similar ARC tasks
- Category-theoretic frameworks for representing ARC transformations
- Attention-based spatial reasoning modules that operate directly on 2D grid structure
- Combining forward search (DSL) with backward reasoning (constraint propagation)
- World-model learning for ARC-AGI-3 interactive environments

> [!WARNING]
> **None of these hypotheses have been verified as novel.** Each requires a thorough literature review before being adopted as a research direction.

---

## 9. Recommendation: Which Competition to Target

### Primary Target: **ARC-AGI-2**

**Rationale:**

| Factor | ARC-AGI-2 | ARC-AGI-3 |
|:--|:--|:--|
| **Accessibility** | ✅ Straightforward JSON in/out | ❌ Complex interactive API |
| **Dataset availability** | ✅ 1,120 public tasks (train + eval) | ⚠️ Limited public games |
| **Local development** | ✅ Easy — just load JSON, predict grid | ⚠️ Requires arc-agi package + env setup |
| **Research maturity** | ✅ Well-studied; can build on prior work | ❌ Very new; limited prior work |
| **Paper Track compatibility** | ✅ Direct link to code submission | ✅ Direct link to code submission |
| **Evaluation simplicity** | ✅ Exact match on grids | ❌ RHAE efficiency scoring |
| **Research opportunity** | ✅ Compositional generalization gap | ✅ Fundamental open problem |
| **Risk** | ⚠️ More competitive | ✅ Less competitive but harder |

### Secondary Target: **Paper Track**

The Paper Track is our ultimate submission target, as it rewards *understanding* and *theory* over raw performance. We can submit a paper based on our ARC-AGI-2 code submission regardless of leaderboard position.

### Why Not ARC-AGI-3 Initially?

1. The paradigm is fundamentally different (agentic, interactive) — requires a separate research track
2. Very little prior work to build on
3. The 1% frontier model ceiling suggests this may be premature for a first paper
4. However: ARC-AGI-3 offers the highest novelty potential for future work

### Recommendation Summary

> **Start with ARC-AGI-2 to build strong foundations** → **Submit to Paper Track** → **Explore ARC-AGI-3 as a stretch goal**

---

## 10. First Baseline and Experiment Recommendations

### 10.1 First Baseline: Multi-Strategy Heuristic Solver

A baseline solver that combines simple, interpretable heuristics:

1. **Identity baseline**: Return input as output (measures how many tasks are no-ops)
2. **Majority color fill**: Fill output with the most common color
3. **Grid copy with color substitution**: Apply simple color mapping rules
4. **Input/output size analysis**: Statistical analysis of grid dimension changes

This establishes a floor and helps us understand the task distribution.

### 10.2 First Experiment: LLM-Guided Program Synthesis (Greenblatt-style)

Replicating the core of Ryan Greenblatt's approach:

1. Prompt an LLM to generate Python programs that transform input→output
2. Execute programs against training pairs
3. Keep programs that pass all training examples
4. Apply passing programs to test inputs
5. Measure: solve rate, programs per task, cost, failure modes

This gives us a strong reproducible baseline (~40–50% on ARC-AGI-1) and a platform for experimentation.

### 10.3 Important Risks

| Risk | Severity | Mitigation |
|:--|:--|:--|
| **Compute costs** | High | Start with open-weight models (Qwen, Llama); use local GPU |
| **ARC-AGI-2 saturation** | Medium | Focus on Paper Track novelty, not just leaderboard rank |
| **Deadline pressure** | High | Nov 9, 2026 deadline — ~11 weeks remaining |
| **Overfitting to public eval** | Medium | Reserve a held-out validation split from training set |
| **Scope creep** | Medium | Strict milestone-based roadmap; no feature creep |
| **Prior art overlap** | Medium | Thorough literature review before claiming novelty |
| **Kaggle submission constraints** | Medium | Test early in Kaggle environment; no internet, 9hr limit |

---

## Appendix A: ARC Color Encoding

| Value | Color |
|:--|:--|
| 0 | Black (background) |
| 1 | Blue |
| 2 | Red |
| 3 | Green |
| 4 | Yellow |
| 5 | Grey |
| 6 | Magenta/Fuchsia |
| 7 | Orange |
| 8 | Cyan/Azure |
| 9 | Maroon/Brown |

## Appendix B: Key Resources

| Resource | URL |
|:--|:--|
| ARC-AGI-2 Dataset | https://github.com/arcprize/ARC-AGI-2 |
| ARC-AGI-3 Starter Kit | https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter |
| ARC Prize Official Site | https://arcprize.org |
| ARC-AGI-2 Kaggle Competition | https://www.kaggle.com/competitions/arc-prize-2026 |
| ARC-AGI-2 Human Testing Data | https://huggingface.co/datasets/arcprize/arc_agi_2_human_testing |
| Original ARC Paper (Chollet 2019) | https://arxiv.org/abs/1911.01547 |
