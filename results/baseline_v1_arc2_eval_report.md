# Baseline v1 Evaluation Report

> **Date**: 2026-08-21 11:58:56  
> **Dataset Evaluated**: `data\ARC-AGI-2\data\evaluation`  
> **Solver**: `RuleBasedSearchSolver_v1`  

---

## 1. Executive Summary

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Task-Level Exact Match** | **0.00%** (0/120) | All test outputs in task exactly matched |
| **Test-Pair Exact Match** | **0.00%** (0/167) | Individual test grid accuracy |
| **Training Consistency** | **0.00%** (0/120) | Candidate explained all training examples |
| **Generalization Gap** | **0.00%** | Train consistency vs test accuracy drop |
| **Avg Search Latency** | **12.34 ms** | Average per-task inference time |
| **Total Wall-Clock Time** | **1.48 s** | Total dataset evaluation time |
| **Avg Candidates Tested** | **176.9** | Candidates generated per task |

---

## 2. Failure Category Breakdown

Every evaluation task was categorized by failure mode:

| Outcome Category | Task Count | Percentage | Description |
| :--- | :--- | :--- | :--- |
| `UNSOLVED_SIZE_PRESERVING` | 81 | 67.5% | Dimensions matched, but reasoning logic exceeded basic transformation library |
| `UNEXPLAINED_SIZE_CHANGE` | 39 | 32.5% | Grid dimensions changed and was not explained by cropping/tiling/fractals |

---

## 3. Solved Tasks Analysis

The baseline solver solved **0 tasks**:

| Task ID | Discovered Rule |
| :--- | :--- |

---

## 4. Key Findings & Research Opportunities

1. **Baseline Accuracy Established**:
   - The deterministic transformation library establishes a solid, reproducible baseline of **0.00%**.
   - It successfully solves pure geometric rotations, reflections, direct color substitutions, bounding box extractions, and Kronecker fractal expansions without any statistical approximation.

2. **Generalization Gap Analysis**:
   - On **0 tasks (0.0%)**, at least one transformation candidate explained the training demonstrations.
   - In **0 tasks**, the candidate rule fit the training data by coincidence (spurious correlation) but failed on test. This highlights the need for stronger inductive bias and invariant testing.

3. **Major Failure Bottlenecks**:
   - **Unsolved Size-Preserving Tasks**: Represent the largest fraction of failures. These require multi-step reasoning, object-level tracking, connected component graph traversal, flood-fill containment, or counting.
   - **Complex Dimension Changes**: Subgrid extraction based on semantic criteria (e.g., 'find the smallest red object and crop it') requires semantic object parsing rather than static bounding boxes.

4. **Next Research Steps (Phase 2 Roadmap)**:
   - **Object-Centric Segmentation**: Implement connected-component analysis and object graph representations.
   - **Domain-Specific Language (DSL)**: Expand beyond fixed transformations to composable programs with loops, conditions, and object filters.
   - **Program Synthesis with Verification**: Integrate LLM / neural guidance as proposal engine with symbolic verification.