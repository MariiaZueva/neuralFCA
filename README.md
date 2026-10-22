# Neural FCA: Interpretable Neural Networks on Concept Lattices

A big research-oriented homework for the **Ordered Sets for Data Analysis** course.

This project merges **neural networks** and **Formal Concept Analysis (FCA)**.

Pay attention on helpful resources with code provided:
- Libraries with FCA methods implemented:
	 https://github.com/EgorDudyrev/FCApy
	 https://github.com/smartFCA/caspailleur 
- example **neural_lib62.py** file with neural_network architecture implemented (you may use and adapt it for your needs)
- toy example **NeuralFCA_Big_hw.ipynb** of working with neural networks based on concept lattices.

There is a growing need for interpretability in AI: standard neural networks are hard to explain, but there is a principled way to build an interpretable network architecture from the **covering relation (Hasse diagram)** of a concept lattice coming from a monotone Galois connection. In such a network, hidden neurons correspond to formal concepts (or disjunctive concepts), and edges between them follow the lattice order. A neuron’s activation can be read as the importance of a set of similar objects described by common attributes, and edges admit a natural reading in terms of concept generality (bottom-up) or conditional probability (top-down).

The main research question:

```text
Can a concept-lattice-structured neural network compete with standard ML models
while offering a more interpretable architecture through concept neurons?
```

## Task Description

You need to choose a classification dataset, binarize it, build a formal context, compute the lattice of disjunctive concepts, construct a Neural FCA network, train it, and compare it with standard baselines. The core pipeline follows Kuznetsov et al. (2017).

Minimum pipeline:

1. Choose an open classification dataset (hundreds of rows, varied feature
   types).
2. Perform EDA and describe a binarization (scaling) strategy.
3. Build a formal context `K = (G, M, I)`.
4. Compute the lattice of **disjunctive concepts** (monotone Galois connection) and its covering relation.
5. Build a k-level lattice diagram (max intent size `k`) and construct the neural network architecture:
   - input layer = binary attributes,
   - hidden layers = concepts linked by the covering relation,
   - output layer = classes.
1. Initialize concept-to-concept weights from conditional probabilities `P(C_j | C_i)` for `C_i ⊆ C_j` (and from concept-to-class conditional probabilities for the output layer).
2. Select top-N concepts (`N ≤ 30`) with a greedy procedure using concept-quality metrics (F1, purity, weighted precision/recall with `α ∈ [0, 1]`).
3. Train with chosen optimizer (for ex., Adam), chosen activation functions and hyperparameters.
4. Evaluate on a held-out test set and compare with standard baselines.

Main improvements expected beyond the baseline:

- better binarization / scaling strategy;
- better concept selection;
- your own modification of the Neural FCA architecture.

## Learning Goals

After completing the project, you should be able to:

1. Apply FCA to build an interpretable neural network from data (context, disjunctive concept lattice, covering relation).
2. Implement a binarization pipeline for mixed-type features to construct a formal context.
3. Use concept-quality metrics (F1, purity, weighted precision/recall) to select the most informative concepts and reduce the network to a bounded number of neurons (`≤ 30`).
4. Compare Neural FCA with standard ML models and document the effect of weight  initialization and activation functions.
5. Interpret concept neurons: their intent, extent, etc..
6. Visualize the resulting network and analyze its interpretability.

## To-do List

### Basic solution

- [ ] Find an open classification dataset (Kaggle, UCI, etc.); define the
      target attribute.
- [ ] Describe a binarization strategy for each feature type (numerical intervals,
      one-hot for categorical, unchanged binaries).
- [ ] Justify chosen thresholds / intervals (distributions, percentiles,
      correlation with target).
- [ ] Perform a stratified train/test split with fixed `random_state`.
- [ ] Implement at least 4 standard baselines (kNN, Naive Bayes, Logistic
      Regression, SVM, Decision Tree, Random Forest, XGBoost).
- [ ] Choose a primary quality measure (F1-score recommended; justify if using
      something else).
- [ ] Build the formal context and the disjunctive concept lattice.
- [ ] Construct a k-level diagram with a max intent size `k`.
- [ ] Build the Neural FCA network with concept neurons and covering-relation
      edges.
- [ ] Initialize weights from conditional probabilities.
- [ ] Perform greedy top-N concept selection (`N ≤ 30`) using F1/purity/weighted
      metric.
- [ ] Train with optimizer, activation functions and hyperparameters.
- [ ] Evaluate and compare with baselines on the same split.
- [ ] Prepare a PDF report and a short presentation.

### Good solution

- [ ] Try several binarization strategies (fixed bins, percentiles, entropy-based
      discretization) and compare them.
- [ ] Compare several concept-selection metrics (F1, purity, delta-stability, weighted
      precision/recall for different `α`).
- [ ] Compare several activation functions (sigmoid, ReLU, GELU, softmax...)
- [ ] Compare weight initialization variants (random vs. conditional-probability
      based).
- [ ] Repeat experiments with at least 3 train/test splits (or k-fold CV) to
      assess stability.
- [ ] Visualize the concept network with intents and edge weights.
- [ ] Analyze 3–5 most important concept neurons with domain-level explanations.

### Excellent solution (any of the following points give an advantage)

- [ ] Propose your own modification of the Neural FCA architecture (e.g. new
      edge structure, concept gating, concept-wise attention, alternative
      initialization).
- [ ] Test the approach on two or more datasets.
- [ ] Study the effect of the number of selected concepts on accuracy and
      interpretability.
- [ ] Compare interpretability of Neural FCA with decision trees and logistic
      regression.
- [ ] Provide a per-object activation path explanation (input → concepts →
      class) for at least one or two test objects.
- [ ] Report training time, number of parameters, and coverage of the training
      set by selected concepts.

## Submission

Your repository should be standalone and follow the general course structure.

It should contain:

- working code;
- one main notebook demonstrating the full pipeline (EDA → binarization →
  context → lattice → network → training → evaluation → visualization);
- a PDF report;
- a short presentation;
- saved examples of concept-level explanations;
- `requirements.txt` or `pyproject.toml`;
- `README.md` or `README_EN.md` with reproduction instructions.

The report should describe classification accuracy, the
architecture and interpretability: which concepts were selected, what they mean,
how the network uses them, and how this compares to standard ML models.

## Recommended Repository Structure

```text
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda_and_binarization.ipynb
│   ├── 02_baselines.ipynb
│   └── 03_neural_fca.ipynb
├── src/
│   ├── data.py
│   ├── scaling.py
│   ├── context.py
│   ├── concept_selection.py
│   ├── network_builder.py
│   ├── train_eval.py
│   ├── baselines.py
│   ├── viz.py
│   └── evaluation.py
├── outputs/
│   ├── lattices/
│   └── explanations/
├── reports/
│   ├── report.pdf
│   └── presentation.pdf
├── requirements.txt
└── README.md
```

## Formal Context and Binarization

A **formal context** is a triple `K = (G, M, I)`:

- `G` — set of objects (rows of the dataset);
- `M` — set of binary attributes;
- `I ⊆ G × M` — incidence relation: `(g, m) ∈ I` means object `g` has attribute
  `m`.

Since FCA requires binary attributes, multi-valued features must be **scaled
(binarized)**:

- **Numerical features** → threshold attributes `x ≤ t` / `x > t`, or interval
  bins (fixed-width, equal-frequency, entropy-based).
- **Categorical features** → one-hot encoding.
- **Binary features** → used unchanged.

The report must justify:

- the number of bins per numerical feature;
- the choice of thresholds (e.g. percentiles, domain meaning);
- handling of rare categories and missing values;
- the final number of binary attributes `|M|`.

**Caution:** coarse binning loses information; too fine binning blows up `|M|`,
increases lattice size, and hurts interpretability. Be aware that CbO-style lattice construction can become prohibitively slow when `|M| > 20–30`, so the choice of lattice algorithm matters (see below).

## Disjunctive Concepts and the Network Architecture

Classical FCA uses an **antitone** Galois connection: larger extent ⇒ smaller
intent. Neural FCA instead uses a **monotone Galois connection**, yielding
**disjunctive concepts**:

```text
if A_1 ⊆ A_2 then B_1 ⊆ B_2
```

This aligns well with the idea of a neuron that can be activated by several alternative attribute sets (a disjunction).

Architecture (Kuznetsov et al., 2017):

- **Input layer**: one neuron per binary attribute `m ∈ M`.
- **Hidden layers**: one neuron per selected concept `C = (A, B)`; layers follow
  the covering relation of the k-level diagram.
- **Output layer**: one neuron per class.
- **Edges**:
  - Concept → concept: edge from `C_i` to `C_j` whenever `C_i ⊆ C_j` in the
    lattice (covering relation).
  - Concept → class: weighted contribution of the concept to each class.

**Weight initialization (semantic, not random):**

```text
w(C_i → C_j) ∝ P(C_j | C_i) = |A_j| / |A_i|          (for C_i ⊆ C_j)
w(C → class c) ∝ purity_c(C) = |{g ∈ A : y_g = c}| / |A|
```

This makes initial weights reflect the concept hierarchy rather than arbitrary
noise, and improves convergence and interpretability.

## Lattice Construction and Concept Selection

The full concept lattice can be exponential in `|M|`, so the project restricts
the structure:

- **k-level diagram**: keep only concepts whose intent size `≤ k`.
- **Top-N selection** (`N ≤ 30`) by a greedy procedure using concept-quality
  metrics:
  - **F1-score** of the concept as a class-c classifier;
  - **purity** `purity_c(C) = |{g ∈ A : y_g = c}| / |A|`;
  - **weighted precision/recall** with `α ∈ [0, 1]`:

```text
score_α(C) = α · precision_c(C) + (1 − α) · recall_c(C)
```

Recommended lattice-building algorithms:

- **Sofia** — usually fast on modest contexts;
- **AddIntent** — alternative;
- **CbO** — default in many libraries, but **very slow** for `|M| > 20–30`; not
  recommended without careful scaling or concept-size limits.

**Constraints to keep the network small and interpretable:**

- max intent size `k`;
- min support of concepts;
- min purity;
- top-N budget on selected concepts.

## Concept Quality Metrics

For a concept `C = (A, B)` and class `c`:

```text
purity_c(C)   = |{g ∈ A : y_g = c}| / |A|
support_c(C)  = |{g ∈ A : y_g = c}| / |G_c|
precision_c(C) = TP / (TP + FP)
recall_c(C)    = TP / (TP + FN)
```

where `TP = |{g ∈ A : y_g = c}|`, `FP = |A| − TP`,
`FN = |G_c| − TP`.

Additional architecture-level metrics:

- number of network parameters;
- training time per epoch;
- training-set coverage by selected concepts (share of objects activating at
  least one selected concept);
- average extent size of selected concepts.

For imbalanced datasets, F1 and per-class precision/recall on the minority class
remain primary; the architecture metrics are supplementary.

## Baseline Models

Recommended baselines (identical splits, identical labels):

- k-Nearest Neighbors;
- Naive Bayes;
- Logistic Regression;
- SVM;
- Decision Tree;
- Random Forest;
- XGBoost / CatBoost.

The comparison must be fair: same train/test split, same metric, and explicit
description of preprocessing differences (e.g. baselines may use standardized
numerical features while Neural FCA uses binarized attributes).

## Experimental Protocol

Minimum protocol:

- one open dataset;
- stratified train/test split with fixed `random_state`;
- primary metric: **F1-score**;
- additional metrics: accuracy, precision, recall, ROC-AUC, confusion matrix;
- at least 4 standard baselines;
- at least one Neural FCA configuration.

Fixed parameters to report:

**Binarization**

- original number of features;
- number of binary attributes after scaling;
- per-feature binning strategy.

**Lattice**

- level `k` (max intent size);
- total number of concepts in the k-level diagram;
- number of selected concepts `N ≤ 30`;
- selection metric and threshold;
- lattice construction algorithm.

**Network**

- number of hidden layers and neurons per layer (before and after selection);
- activation function (sigmoid, ReLU, GELU, softmax);
- weight-initialization scheme (random, conditional-probability, etc.);
- optimizer (Adam), learning rate, number of epochs, loss function;
- training time and parameter count.

Extended protocol:

- repeat experiments with at least 3 train/test splits (or k-fold CV) and report
  means and standard deviations;
- compare binarization strategies;
- compare activation functions;
- compare concept-selection metrics (`F1`, purity, weighted `α`);
- compare architecture variants (including your modification).

## Interpretability Analysis

Interpretability is evaluated separately from numerical quality. The report must
include:

1. **Semantic interpretation of neurons.** Pick 3–5 most significant concept
   neurons (e.g. largest absolute weights to the output layer) and explain, in
   domain terms:
   - the intent `B` (e.g. “glucose in (150, 200]”, “age > 60”);
   - the extent `A` (which objects activate this neuron);
   - the purity with respect to the target class.

2. **Interpretation of weights.** Show examples of concept→concept and
   concept→class weights and explain their conditional-probability meaning
   (e.g. “concept C_i implies class + with probability 0.87”).

3. **Activation path for a specific object.** For 1–2 test objects, trace the
   activation from input attributes through concepts to the output class and
   write a natural-language explanation:
   > “The object has high glucose and high BMI, which activates concept
   > ‘high diabetes risk’; this concept has a large positive weight on the
   > ‘diabetes’ output neuron.”

4. **Effect of concept selection.** Compare networks with different `N` (e.g.
   `N = 4, 10, 20, 30`) and check whether smaller networks keep concept
   meanings clean. In the course example, at `N = 4` each concept was easily
   interpretable; at `N = 10` some concepts became less pure and harder to
   explain.

5. **Network visualization.** Provide a graph of the concept network (nodes =
   concepts with intents, edges = covering relation with weights). Use color
   intensity for weight magnitude.

6. **Comparison with standard interpretable models** (decision tree, logistic
   regression): what advantages does representing rules as a lattice/concept
   structure give?

## Typical Mistakes

- Using fixed global thresholds for numeric features without checking the
  distribution; or too few/many bins producing empty or redundant attributes.
  *Fix:* inspect histograms; use percentiles or entropy-based discretization.
- Using a lattice algorithm unsuitable for `|M| > 20–30` (e.g. CbO default).
  *Fix:* try Sofia or AddIntent and choose the fastest.
- **Overfitting due to too many concepts.** Including the full lattice or too
  many selected concepts reduces generalization. *Fix:* greedy selection, cap
  `N ≤ 30`, choose `N` on validation.
- **Too few epochs.** Neural FCA often needs 2 000–10 000 epochs to converge
  because of the sparse structured architecture. *Fix:* experiment with
  2 000 / 5 000 / 10 000 epochs and monitor the validation curve.
- Running Neural FCA on `> 100` raw features without careful binarization or
  selection → lattice blow-up.
- Reporting only accuracy on an imbalanced dataset.
- Using different train/test splits for different models.
- Claiming interpretability without concrete concept examples.
- Comparing baselines computed on raw features with Neural FCA on binarized
  features without noting the difference.

## Check Questions

1. What is a formal context, formal concept, and concept lattice? How are they
   used as a neural-network architecture here?
2. What is the difference between standard formal concepts (antitone Galois
   connection) and disjunctive concepts (monotone Galois connection)?
3. How are “interesting” concepts selected for the network?
4. How do you interpret a hidden-layer neuron? How do intent and extent relate
   to activation?
5. Which metrics did you use for binary classification? Why is F1 preferable to
   accuracy in some cases?
6. What causes overfitting in Neural FCA, and how can it be avoided?
7. Why is the conditional-probability initialization preferable to random
   initialization?
8. How does the number of selected concepts `N` affect quality and
   interpretability?
9. What role does the level `k` of the lattice diagram play?
10. How do you build a natural-language explanation for a single object’s
    classification?

## Recommended References

1. Kuznetsov, S. O., Makhazhanov, N., Ushakov, M. *On neural network
   architecture based on concept lattices.* ISMIS 2017, Springer, 2017.
2. Kuznetsov, S. O., Makhalova, T. *On interestingness measures of formal
   concepts.* Information Sciences 442, 202–219, 2018.
3. Зуева М. М., Кузнецов С. О. *Индексы интересности для построения нейронных
   сетей на основе решёток понятий.* Автоматика и телемеханика, 2024.
4. Ganter, B., Wille, R. *Formal Concept Analysis: Mathematical Foundations.*
   Springer, 1999.
5. Ganter, B., Kuznetsov, S. O. *Pattern Structures and Their Projections.*
   LNCS, 2001.
