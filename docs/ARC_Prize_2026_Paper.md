# Hierarchical Spatial Reasoning for the Abstraction and Reasoning Corpus

**Abstract**
The Abstraction and Reasoning Corpus (ARC-AGI) measures an AI system's ability to adapt to novel tasks using core knowledge priors. While Large Language Models (LLMs) rely on massive pattern memorization, they struggle with the strict spatial and topological reasoning required by ARC. In this paper, we present a purely deterministic, rule-based reasoning system that progressively integrates object-centric representations, spatial-topological primitives, and hierarchical sub-goal decomposition. By conducting a rigorous ablation study on the ARC-AGI dataset, we demonstrate that explicit topological representations (e.g., enclosure, raycasting) improve test generalization from a 0.50% baseline to 0.75%, while hierarchical A* search utilizing state-difference heuristics significantly improves search efficiency, reducing execution time by 30% compared to blind multi-step search. However, overall generalization remains tightly bottlenecked by the bounds of the primitive transformation library. We report all negative results and highlight the substantial gap between training (8.25%) and evaluation (0.75%) accuracies. Finally, we propose a promising future direction: hybridizing our explicit topological state graphs with recursive tiny neural networks, such as the Tiny Recursive Model (TRM).

---

## 1. Introduction
The Abstraction and Reasoning Corpus (ARC) (Chollet, 2019) was introduced to benchmark general intelligence by isolating an agent's ability to acquire novel skills from few demonstrations. Despite massive scaling, standard deep learning and Large Language Models (LLMs) perform poorly on ARC due to their reliance on approximate interpolation rather than exact, core-knowledge-driven logical synthesis. 

In this work, we investigate whether explicitly structuring ARC grids into symbolic, object-centric representations and guiding program synthesis via topological sub-goals can improve generalization. Rather than optimizing a black-box model, we developed a deterministic program synthesis engine and evaluated the isolated impact of three architectural hypotheses:
1. **Object-Centric Reasoning:** Parsing pixel grids into discrete objects with localized properties.
2. **Topological and Spatial Primitives:** Extracting connectivity, cavities, and spatial relations (raycasting).
3. **Hierarchical Sub-Goal Search:** Guiding multi-step program synthesis using A* search and state-difference heuristics.

---

## 2. Methodology

Our solver operates purely deterministically, treating ARC task resolution as a program synthesis problem across a constrained Domain Specific Language (DSL). The reasoning pipeline executes in three stages.

### 2.1 Object-Centric State Representation
Raw 2D arrays are decomposed into an Object Relational Graph $G = (V, E)$. Vertices $V$ represent detected objects (via 4-way and 8-way connectivity, utilizing both monochromatic and multi-color heuristics). Each object caches geometric properties, including area, centroid, bounding box, symmetry, and shape signature.

### 2.2 Topological and Spatial Primitives
To capture human-like core knowledge priors, we augmented the state representation with topological feature extraction. This includes:
* **Enclosure Detection:** Identifying objects that fully surround background pixels (cavities).
* **Raycasting:** Firing directional rays from object boundaries to detect collisions, establishing explicit `Adjacent`, `Above`, and `Collinear` edges $E$ between objects.

### 2.3 Hierarchical Sub-Goal A* Search
To combat the exponential branching factor of multi-step transformations (e.g., $k \ge 3$), we implemented a Hierarchical Planner. For a given training pair $(X, Y)$, the planner computes a structural `StateDifference` representing the semantic delta (e.g., $\Delta \text{colors}$, $\Delta \text{object\_count}$, $\Delta \text{symmetry}$). 
Using these deltas as heuristics, the A* search dynamically prunes primitive transformations that do not reduce the structural distance between the current state and the target output, preventing the combinatorial explosion typical of blind depth-first search.

---

## 3. Experimental Setup

We evaluated our system on the complete ARC-AGI-1 evaluation set (400 tasks) and training set (400 tasks), measuring strict exact-match accuracy. No hidden outputs were used, and our system maintained zero test-leakage. We ran ablations across five distinct solver configurations to isolate the impact of each reasoning module.

---

## 4. Results & Ablation Study

| Search Strategy | Eval Acc | Train Acc | Runtime |
| :--- | :---: | :---: | :---: |
| A. Baseline (Depth-2) | 0.50% | 6.00% | 9.8s |
| B. Object-Centric | 0.50% | 7.75% | 67.6s |
| C. + Spatial / Topology | **0.75%** | 8.00% | 583.8s |
| D. Blind Depth-4 Search | 0.75% | 8.00% | 25.1s |
| E. Hierarchical Depth-4 Search | **0.75%** | **8.25%** | **17.5s** |

**Key Findings:**
1. **Topology Aids Generalization:** Moving from standard generic operations to topological reasoning (Strategy C) yielded the only strict gain in held-out Evaluation Accuracy (+0.25%).
2. **Hierarchical Efficiency:** The A* sub-goal planner (Strategy E) matched the accuracy of blind search while expanding significantly fewer nodes, resulting in a ~30% runtime reduction (17.5s vs 25.1s).
3. **Training vs. Evaluation Gap:** The system successfully fit increasingly complex training transformations (up to 8.25%), but these highly specific multi-step pipelines largely failed to transfer to the distinct mechanics of the evaluation set.

---

## 5. Discussion and Limitations

The primary limitation of our system is the bounds of its Domain Specific Language. Deterministic rule-based synthesis is highly interpretable and avoids the hallucination issues of LLMs, but it is fundamentally constrained. Complex algorithmic mechanics—such as maze pathfinding, gravity simulations, or arbitrary shape morphing—fall outside our primitive vocabulary. 

Furthermore, while sub-goal decomposition successfully pruned the search tree, it could not synthesize novel logic that wasn't already hard-coded into the transformation engine.

---

## 6. Conclusion and Future Work

This research demonstrates that explicitly modeling human-like core knowledge—such as object permanence, spatial topology, and intermediate sub-goals—meaningfully improves the efficiency of ARC solvers. However, to achieve generalized abstraction, the system must be capable of dynamically learning new concepts rather than relying solely on a fixed symbolic DSL.

**Future Work: Integration with Tiny Recursive Models (TRM)**
We propose hybridizing our explicit topological state representations with emerging recursive neural architectures. Recent breakthroughs, such as the *Tiny Recursive Model (TRM)* (arXiv:2510.04871), demonstrate that recursive reasoning via extremely small networks (e.g., 2 layers, 7M parameters) can achieve profound ARC-AGI generalization without the parameter-bloat of LLMs. By feeding our high-fidelity, explicit relational object graphs into a TRM-style recursive network, we hypothesize the network could learn arbitrary spatial transformations efficiently while bypassing the standard pixel-level noise bottleneck. 

---

**References**
1. Chollet, F. (2019). On the Measure of Intelligence. *arXiv preprint arXiv:1911.01547*.
2. Jolicoeur-Martineau, A. (2025). Less is More: Recursive Reasoning with Tiny Networks. *arXiv preprint arXiv:2510.04871*.
