---
layout: home
permalink: index.html
repository-name: e20-4yp-Impact-of-Hybrid-Structured-Random-Projections-on-GNN-Performance
title: Impact of Hybrid Structured Random Projections on GNN Performance
---

# Impact of Hybrid Structured Random Projections on GNN Performance

#### Team

- E/20/133, Tharusha Haththella, [e20133@eng.pdn.ac.lk](mailto:e20133@eng.pdn.ac.lk)
- E/20/305, Sachindu Premasiri, [e20305@eng.pdn.ac.lk](mailto:e20305@eng.pdn.ac.lk)
- E/20/381, Nimesha Somathilaka, [e20381@eng.pdn.ac.lk](mailto:e20381@eng.pdn.ac.lk)

#### Supervisor

- Dr. Eng. Sampath Deegalla, [sampath@eng.pdn.ac.lk](mailto:sampath@eng.pdn.ac.lk)

#### Table of content

1. [Abstract](#abstract)
2. [Related works](#related-works)
3. [Methodology](#methodology)
4. [Experiment Setup and Implementation](#experiment-setup-and-implementation)
5. [Results and Analysis](#results-and-analysis)
6. [Conclusion](#conclusion)
7. [Publications](#publications)
8. [Links](#links)

---

## Abstract

Graph representation learning has become an important approach for analyzing complex networked data in tasks such as node classification and link prediction. However, many graph embedding methods rely on dense random projections, which can introduce higher computational overhead and reduced efficiency when scaling to large graphs. This project investigates whether hybrid structured random projections can provide a better balance between efficiency and predictive performance.

The work is centered around the FastRP graph embedding framework and compares three projection strategies: a Gaussian baseline, a striped structured sparse projection, and a hybrid projection that combines structural information with node feature information. Experiments are conducted on benchmark datasets including BlogCatalog, Flickr, Cora, and ogbn-arxiv. Hyperparameters are optimized using Optuna, and downstream performance is evaluated mainly using Macro-F1 and classification accuracy. The results indicate that hybrid and structured projections can remain competitive with, and in some cases outperform, conventional dense Gaussian projections while offering a more scalable direction for graph representation learning.

## Related works

Graph embedding methods transform graph nodes into low-dimensional vector representations while preserving structural and semantic relationships. Among these methods, FastRP is a lightweight and scalable embedding approach that generates node representations using random projections and iterative neighborhood aggregation. It is especially attractive for large graphs because it avoids the heavy training cost of deeper graph neural network architectures.

Despite these advantages, dense Gaussian random projections can still impose computational and memory overhead. This has motivated interest in structured sparse random projections, where the projection matrix is designed to be computationally lighter and more efficient.

In addition, recent graph learning research has shown the value of combining graph structure with node feature information. While structure-only embeddings are effective in many settings, feature-rich datasets may benefit significantly from hybrid methods. This project therefore investigates both structured sparse projections and a hybrid structure-feature projection strategy within the same FastRP-based pipeline.

## Methodology

The proposed methodology studies how the design of the projection matrix affects graph embedding quality and downstream performance.

The overall workflow is as follows:

1. Load the graph structure and, when available, node features.
2. Construct a projection matrix using one of the following strategies:
   - **Gaussian Projection** – dense random baseline.
   - **Striped Structured Projection** – sparse structured matrix designed to reduce computational overhead.
   - **Hybrid Projection** – combines structural embedding generation with node feature information.
3. Generate node embeddings through iterative neighborhood aggregation.
4. Use the embeddings for downstream graph learning tasks such as:
   - Node classification
   - Link prediction
5. Evaluate performance using appropriate metrics.

The main objective is to determine whether structured sparse projections can preserve or improve the representational quality of dense Gaussian projections while reducing overhead. A secondary objective is to evaluate whether hybridizing structural and feature information leads to stronger predictive performance on feature-rich datasets.

## Experiment Setup and Implementation

The implementation is carried out in Python using graph learning and scientific computing libraries. The project includes notebook-based experimentation and modular code for data loading, embedding generation, tuning, and downstream evaluation.

### Datasets

The experiments are conducted on multiple benchmark datasets with different graph sizes and characteristics:

- **BlogCatalog** – social network dataset used for node classification
- **Flickr** – large social/content-sharing graph used for node classification
- **Cora** – citation network benchmark
- **ogbn-arxiv** – large-scale citation graph from the Open Graph Benchmark
- Additional citation-style datasets for link prediction evaluation

### Experimental pipeline

The evaluation pipeline includes:

- Graph preprocessing and loading
- Sparse or dense projection matrix construction
- Embedding generation using FastRP-style propagation
- Hyperparameter tuning using **Optuna**
- Node classification using downstream classifiers
- Link prediction evaluation using train/validation/test edge splits and negative sampling

### Tuned parameters

Key tunable parameters include:

- Embedding dimension
- Window size
- Normalization
- Structured grouping parameter `g`

This setup enables fair comparison between the Gaussian baseline, striped structured projection, and hybrid projection strategies.

## Results and Analysis

The experimental results show that the effectiveness of a projection method depends on both the dataset and the availability of node features.

### 1. Node classification on Cora

| Model | Score |
|---|---:|
| Gaussian Baseline | 0.7232 |
| Striped Structured Projection | 0.7356 |
| Hybrid Projection | 0.7541 |

For Cora, both structured and hybrid approaches outperform the Gaussian baseline. The hybrid model achieves the highest score, showing that integrating feature information with structural propagation can improve embedding quality.

### 2. Large-scale node classification on ogbn-arxiv

| Model | Train Accuracy | Validation Accuracy | Test Accuracy |
|---|---:|---:|---:|
| Gaussian Baseline | 0.5149 | 0.5080 | 0.5211 |
| Striped HRP | 0.5115 | 0.5063 | 0.5184 |
| Hybrid HRP | 0.7468 | 0.6970 | 0.6891 |

On ogbn-arxiv, the hybrid model clearly outperforms both the Gaussian and striped structure-only variants. This suggests that hybrid structure-feature embeddings are especially beneficial on large graphs with informative node attributes.

### 3. Flickr observation

| Model | Macro-F1 |
|---|---:|
| Gaussian Baseline | 0.2030 |
| Striped Structured Projection | 0.1961 |

In this particular Flickr experiment, the structured sparse variant performed slightly below the Gaussian baseline. This indicates that structured sparsity is promising but can be sensitive to dataset properties and hyperparameter choices.

### 4. General observations

- The **Hybrid projection** shows the strongest overall performance in the current experiments.
- The **Striped structured projection** can be competitive and sometimes outperform the Gaussian baseline.
- The **Gaussian projection** remains a strong baseline for comparison.
- The results suggest that carefully designed hybrid structured projections can improve predictive quality while supporting scalability.

### 5. Link prediction

A complete link prediction pipeline was also implemented using edge splitting, negative sampling, and Optuna-based tuning. This extends the project beyond node classification and demonstrates the applicability of the proposed projection strategies across multiple graph learning tasks.

## Conclusion

This project investigated whether structured and hybrid random projection strategies can improve the efficiency-performance trade-off in graph embedding methods. The findings show that projection design has a measurable effect on downstream graph learning quality.

The hybrid structure-feature projection consistently produced the strongest results in the current experiments, especially on feature-rich datasets such as Cora and ogbn-arxiv. Structured sparse projections also demonstrated promise as a scalable alternative to dense Gaussian projections, although their performance depends on the dataset and tuning configuration.

Overall, the project shows that hybrid structured random projections are a promising direction for scalable graph representation learning and provide a solid basis for future exploration in efficient graph embedding design.

## Publications

<!-- 1. [Semester 7 report](./) -->
<!-- 2. [Semester 7 slides](./) -->
<!-- 3. [Semester 8 report](./) -->
<!-- 4. [Semester 8 slides](./) -->
<!-- 5. Author 1, Author 2 and Author 3, "Research paper title", [PDF](./) -->

## Links

- [Project Repository](https://github.com/cepdnaclk/e20-4yp-Impact-of-Hybrid-Structured-Random-Projections-on-GNN-Performance)
- [Project Page](https://cepdnaclk.github.io/e20-4yp-Impact-of-Hybrid-Structured-Random-Projections-on-GNN-Performance)
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)
