# Baseline v1 Evaluation Report

> **Date**: 2026-08-21 11:59:16  
> **Dataset Evaluated**: `data\ARC-AGI-2\data\training`  
> **Solver**: `RuleBasedSearchSolver_v1`  

---

## 1. Executive Summary

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Task-Level Exact Match** | **2.60%** (26/1000) | All test outputs in task exactly matched |
| **Test-Pair Exact Match** | **2.51%** (27/1076) | Individual test grid accuracy |
| **Training Consistency** | **2.70%** (27/1000) | Candidate explained all training examples |
| **Generalization Gap** | **0.10%** | Train consistency vs test accuracy drop |
| **Avg Search Latency** | **5.53 ms** | Average per-task inference time |
| **Total Wall-Clock Time** | **5.53 s** | Total dataset evaluation time |
| **Avg Candidates Tested** | **142.8** | Candidates generated per task |

---

## 2. Failure Category Breakdown

Every evaluation task was categorized by failure mode:

| Outcome Category | Task Count | Percentage | Description |
| :--- | :--- | :--- | :--- |
| `UNSOLVED_SIZE_PRESERVING` | 663 | 66.3% | Dimensions matched, but reasoning logic exceeded basic transformation library |
| `UNEXPLAINED_SIZE_CHANGE` | 310 | 31.0% | Grid dimensions changed and was not explained by cropping/tiling/fractals |
| `SOLVED` | 26 | 2.6% | Pixel-perfect match on all test pairs |
| `GENERALIZATION_ERROR` | 1 | 0.1% | Rule explained all training demonstrations but failed on test |

---

## 3. Solved Tasks Analysis

The baseline solver solved **26 tasks**:

| Task ID | Discovered Rule |
| :--- | :--- |
| `007bbfb7` | `FractalTiling(background=0)` |
| `0d3d703e` | `ColorSubstitution(mapping={5: 1, 8: 9, 6: 2, 2: 6, 3: 4, 9: 8, 4: 3, 1: 5})` |
| `1cf80156` | `CropBoundingBox(background=0, pad=0)` |
| `1e0a9b12` | `Gravity_down(direction='down', background=0)` |
| `25ff71a9` | `Translate(dr=1, dc=0, fill_color=0, wrap=True)` |
| `3906de3d` | `Gravity_up(direction='up', background=0)` |
| `3c9b0459` | `Rotate180()` |
| `496994bd` | `SymmetryCompletion_both(axis='both', background=0)` |
| `5b6cbef5` | `FractalTiling(background=0)` |
| `5bd6f4ac` | `CropFixed(r_start=0, r_end=3, c_start=6, c_end=9)` |
| `60c09cac` | `KroneckerScale(scale_r=2, scale_c=2)` |
| `6150a2bd` | `Rotate180()` |
| `67a3c6ac` | `HorizontalFlip()` |
| `68b16354` | `VerticalFlip()` |
| `7468f01a` | `Composite(CropBoundingBox(background=0, pad=0) -> HorizontalFlip())` |
| `74dd1130` | `Transpose()` |
| `9172f3a0` | `KroneckerScale(scale_r=3, scale_c=3)` |
| `9dfd6313` | `Transpose()` |
| `a416b8f3` | `Tile(n_rows=1, n_cols=2)` |
| `b1948b0a` | `ColorReplace(old_color=6, new_color=2)` |
| `c59eb873` | `KroneckerScale(scale_r=2, scale_c=2)` |
| `c8f0f002` | `ColorReplace(old_color=7, new_color=5)` |
| `d10ecb37` | `CropFixed(r_start=0, r_end=2, c_start=0, c_end=2)` |
| `d511f180` | `ColorSwap(color1=5, color2=8)` |
| `ed36ccf7` | `Rotate270()` |
| `f25ffba3` | `SymmetryCompletion_vertical(axis='vertical', background=0)` |

---

## 4. Key Findings & Research Opportunities

1. **Baseline Accuracy Established**:
   - The deterministic transformation library establishes a solid, reproducible baseline of **2.60%**.
   - It successfully solves pure geometric rotations, reflections, direct color substitutions, bounding box extractions, and Kronecker fractal expansions without any statistical approximation.

2. **Generalization Gap Analysis**:
   - On **27 tasks (2.7%)**, at least one transformation candidate explained the training demonstrations.
   - In **1 tasks**, the candidate rule fit the training data by coincidence (spurious correlation) but failed on test. This highlights the need for stronger inductive bias and invariant testing.

3. **Major Failure Bottlenecks**:
   - **Unsolved Size-Preserving Tasks**: Represent the largest fraction of failures. These require multi-step reasoning, object-level tracking, connected component graph traversal, flood-fill containment, or counting.
   - **Complex Dimension Changes**: Subgrid extraction based on semantic criteria (e.g., 'find the smallest red object and crop it') requires semantic object parsing rather than static bounding boxes.

4. **Next Research Steps (Phase 2 Roadmap)**:
   - **Object-Centric Segmentation**: Implement connected-component analysis and object graph representations.
   - **Domain-Specific Language (DSL)**: Expand beyond fixed transformations to composable programs with loops, conditions, and object filters.
   - **Program Synthesis with Verification**: Integrate LLM / neural guidance as proposal engine with symbolic verification.