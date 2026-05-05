# References

Curated references relevant to the adaptive heuristic feature selection framework, organised by theme. Sources explored in the context of this project are marked with notes on their relevance.

---

## 1. Instance-wise and Adaptive Feature Selection

**INVASE: Instance-wise Variable Selection using Neural Networks**
Yoon, J., Jordon, J., & van der Schaar, M. (2019).
*International Conference on Learning Representations (ICLR 2019).*
Paper: https://openreview.net/forum?id=BJg_roAcK7
Code: https://github.com/vanderschaarlab/INVASE

> Proposes selecting a different subset of features per data point using an actor-critic framework with three neural networks (selector, predictor, baseline). Demonstrates that feature relevance is not fixed globally but varies by instance. Directly relevant as a motivation and baseline for adaptive/instance-aware selection strategies.

---

**Composite Feature Selection using Deep Ensembles (CompFS)**
Imrie, F., Norcliffe, A., Liò, P., & van der Schaar, M. (2022).
*Advances in Neural Information Processing Systems 35 (NeurIPS 2022).*
Paper: https://arxiv.org/abs/2211.00631
Code: https://github.com/vanderschaarlab/Composite-Feature-Selection

> Addresses the problem that features often act in groups rather than independently (e.g. XOR interactions, multi-gene disease causes). Uses an ensemble of sparse group selectors with a novel regularisation loss to enforce group sparsity and inter-group diversity. Introduces a normalised Jaccard-based group similarity metric (GSim) for evaluation. Relevant for understanding feature interaction structure in diverse datasets and as a complement to flat feature ranking methods.

---

**Learning to Explain: An Information-Theoretic Perspective on Model Interpretation (L2X)**
Chen, J., Song, L., Wainwright, M., & Jordan, M. (2018).
*International Conference on Machine Learning (ICML 2018).*
Paper: https://arxiv.org/abs/1802.07814

> Precursor to INVASE. Introduced instance-wise feature selection but required the user to pre-specify the number of features to select, limiting flexibility. INVASE addresses this limitation directly.

---

## 2. Feature Selection with Statistical Guarantees

**KnockoffGAN: Generating Knockoffs for Feature Selection using Generative Adversarial Networks**
Jordon, J., Yoon, J., & van der Schaar, M. (2019).
*International Conference on Learning Representations (ICLR 2019).*
Paper: https://openreview.net/forum?id=ByeZ5jC5YQ
Code: https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/knockoffgan

> Extends the Model-X knockoff framework by using a GAN to generate valid knockoff features without assumptions on the feature distribution. Enables false discovery rate (FDR) controlled feature selection on arbitrary tabular data. Highly relevant when selection needs to go beyond predictive utility to provide statistical guarantees — particularly for correlated features in diverse datasets.

---

**Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing**
Benjamini, Y., & Hochberg, Y. (1995).
*Journal of the Royal Statistical Society: Series B, 57(1), 289–300.*
Paper: https://doi.org/10.1111/j.2517-6161.1995.tb02031.x

> Foundational paper introducing the FDR framework that underpins knockoff-based selection methods.

---

**Panning for Gold: Model-X Knockoffs for High-dimensional Controlled Variable Selection**
Candès, E., Fan, Y., Janson, L., & Lv, J. (2018).
*Journal of the Royal Statistical Society: Series B, 80(3), 551–577.*
Paper: https://arxiv.org/abs/1610.02351

> Introduces the Model-X knockoff framework used by KnockoffGAN. Relaxes assumptions from the original knockoff construction to work with known but otherwise arbitrary feature distributions.

---

## 3. Synthetic Data Generation and Augmentation

**Synthcity: A Benchmark Framework for Diverse Use Cases of Tabular Synthetic Data**
Qian, Z., Cebere, B.-C., & van der Schaar, M. (2023).
*arXiv.*
Paper: https://arxiv.org/abs/2301.07573
Code: https://github.com/vanderschaarlab/synthcity

> A comprehensive Python library for generating and evaluating synthetic tabular data across a wide range of generators (GANs, VAEs, diffusion models, normalising flows, GNNs, LLM-based). Includes evaluation across statistical fidelity, ML utility, and privacy metrics. Useful for dataset augmentation, benchmarking feature selection pipelines, and generating controlled datasets with known properties.

---

## 4. Data Profiling and Dataset Characterisation

**Data-IQ: Characterizing Subgroups with Heterogeneous Outcomes in Tabular Data**
Seedat, N., Imrie, F., Bellot, A., Bhatt, U., & van der Schaar, M. (2022).
*Advances in Neural Information Processing Systems 35 (NeurIPS 2022).*
Paper: https://arxiv.org/abs/2210.13043
Code: https://github.com/vanderschaarlab/Data-IQ

> Identifies subgroups within tabular datasets that exhibit heterogeneous model behaviour (easy, ambiguous, hard). Relevant to the meta-profiling stage of the heuristic framework — understanding dataset structure before selecting a feature selection strategy.

---

**AutoPrognosis 2.0: Democratizing Diagnostic and Prognostic Modeling in Healthcare**
Imrie, F., Cebere, B., McKinney, E. F., & van der Schaar, M. (2023).
*arXiv.*
Paper: https://arxiv.org/abs/2210.12090
Code: https://github.com/vanderschaarlab/AutoPrognosis

> AutoML framework for tabular data with integrated interpretability. Demonstrates automated pipeline selection (preprocessing, imputation, feature selection, modelling) on diverse datasets. Related to the automated/adaptive framing of this project's framework.

---

## 5. Classical and Embedded Feature Selection Methods

**A Comparative Study of Feature Selection Methods in Machine Learning**
Chandrashekar, G., & Sahin, F. (2014).
*Computers & Electrical Engineering, 40(1), 16–28.*
Paper: https://doi.org/10.1016/j.compeleceng.2013.11.024

> Broad survey of filter, wrapper, and embedded feature selection methods. Useful as background for the three-family heuristic strategy implemented in this framework.

---

**Regression Shrinkage and Selection via the Lasso**
Tibshirani, R. (1996).
*Journal of the Royal Statistical Society: Series B, 58(1), 267–288.*
Paper: https://doi.org/10.1111/j.2517-6161.1996.tb02080.x

> Foundational paper for LASSO, the basis of the embedded/L1 logistic regression strategy in the heuristic selector.

---

**Recursive Feature Elimination (RFE)**
Guyon, I., Weston, J., Barnhill, S., & Vapnik, V. (2002).
*Machine Learning, 46(1–3), 389–422.*
Paper: https://doi.org/10.1023/A:1012487302797

> Original RFE paper underpinning the wrapper strategy in the heuristic selector.

---

## 6. The van der Schaar Lab Ecosystem

The following repositories from the van der Schaar Lab (University of Cambridge) are relevant to this project either as baselines, tools, or conceptual neighbours:

| Repository | Description | Link |
|---|---|---|
| INVASE | Instance-wise variable selection | https://github.com/vanderschaarlab/INVASE |
| Composite-Feature-Selection | Group feature selection with deep ensembles | https://github.com/vanderschaarlab/Composite-Feature-Selection |
| synthcity | Synthetic tabular data generation and evaluation | https://github.com/vanderschaarlab/synthcity |
| AutoPrognosis | AutoML for tabular data with interpretability | https://github.com/vanderschaarlab/AutoPrognosis |
| Data-IQ | Dataset subgroup characterisation | https://github.com/vanderschaarlab/Data-IQ |
| HyperImpute | Automated missing data imputation | https://github.com/vanderschaarlab/HyperImpute |
| Interpretability | Suite of XAI methods (SHAP, LIME, INVASE, SimplEx, Dynamask) | https://github.com/vanderschaarlab/Interpretability |

---

## Notes

Sources are formatted for inline citation as `(Author, Year)` throughout the written materials (`LDS7007M - 240254665.docx`, `Publication-Draft.docx`). Full BibTeX entries are available on request.
