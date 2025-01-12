# Hybrid Recommender System (SVD + NCF)

This repository contains a **C++ implementation** of a **hybrid recommender system** that combines **Singular Value Decomposition (SVD)** and **Neural Collaborative Filtering (NCF)**. It was originally developed as a **CMP2003 "Data Structures & Algorithms"** project on HackerRank, where it achieved **first place**. 

> **Note**: This local version manually splits a sample dataset into `training_data.csv` and `test_data.csv`. The actual competition dataset had ~30,000 rows, and the official test data remains hidden by the competition organizers.

---

## Table of Contents
1. [Introduction](#introduction)  
2. [Key Features](#key-features)  
3. [Performance Summary](#performance-summary)  
4. [Technical Overview](#technical-overview)  
    - [SVD](#svd)  
    - [NCF](#ncf)  
    - [IBCF (Disabled)](#ibcf-disabled)  
5. [Hyperparameters](#hyperparameters)  
6. [Usage](#usage)  
7. [Further Observations](#further-observations)  
8. [Future Work](#future-work)  

---

## Introduction

This hybrid recommender system was developed for a HackerRank competition in CMP2003 "Data Structures & Algorithms." Initially, it utilized three components:
1. **Singular Value Decomposition (SVD)**  
2. **Neural Collaborative Filtering (NCF)**  
3. **Item-Based Collaborative Filtering (IBCF)**  

Subsequent **performance analyses** revealed that **IBCF** introduced significant computational overhead with only marginal improvement in accuracy. It was consequently **disabled** in favor of a **faster** hybrid approach combining only SVD and NCF.

---

## Key Features
- **Hybrid Model**: Merges **SVD** and **NCF** for better accuracy and speed.
- **Lightweight NCF**: Uses a small embedding dimension (3), reducing computational overhead.
- **Scalable**: Successfully handles datasets of ~30,000 rows.
- **High Performance**: 
  - **RMSE**: ~0.9041  
  - **Runtime**: ~0.0793–0.0799 seconds on the competition environment.
- **Configurable Hyperparameters**: Easily adjust factors, dimensions, learning rates, etc.

---

## Performance Summary

- The final hybrid system **disabled IBCF** for improved runtime.
- **RMSE**: ~0.9041 on the competition data.
- **Runtime**: ~0.0799 seconds for test-set predictions.
- Disabling IBCF resulted in a **marginal drop** in accuracy but a **significant** runtime improvement.

---

## Technical Overview

### SVD
1. Learns **latent factors** for users and items.
2. Incorporates **user bias** and **item bias**, along with the **global mean** rating.
3. Suited for capturing **linear** relationships in user-item interactions.

### NCF
1. Learns **neural embeddings** for users and items.
2. Captures **non-linear** relationships.
3. Weights the final prediction alongside SVD results.

### IBCF (Disabled)
- Initially used to compute **cosine similarity** between items.
- Found to be **computationally expensive** for larger datasets.
- Disabled to prioritize **speed**.

---

## Hyperparameters

| Parameter            | Value    | Description                                                    |
|----------------------|----------|----------------------------------------------------------------|
| `SVD_FACTORS`        | 5        | Number of latent factors for SVD                               |
| `NCF_EMBEDDING_DIM`  | 3        | Embedding dimension for NCF                                    |
| `SVD_LR`             | 0.0178f  | Learning rate for SVD                                         |
| `NCF_LR`             | 0.04f    | Learning rate for NCF                                         |
| `REG`                | 0.052f   | Regularization coefficient (L2)                                |
| `EPOCHS`             | 44       | Number of epochs                                               |
| `WEIGHTSVD`          | 0.7f     | Weight of SVD predictions in the final hybrid prediction       |
| `WEIGHTNCF`          | 0.3f     | Weight of NCF predictions in the final hybrid prediction       |
| `WEIGHTIBCF`         | 0        | Weight for IBCF (set to 0 to disable)                          |
| `IBCF_TOP_K`         | 10       | Top similar items considered if IBCF is enabled                |

---

## Usage

1. **Clone or Download** this repository.
2. **Prepare Data**:
   - Provide two CSV files:
     - `training_data.csv` with columns: `userId, itemId, rating`
     - `test_data.csv` with columns: `userId, itemId`
   - Make sure to adjust these paths as needed if they differ from the default.
3. **Build** (using a C++17 compiler or later):
    ```bash
   g++ -std=c++17 main.cpp -o recommender
    ```
4. **Run** the executable: 
    ```bash
   ./recommender
    ```
5. After training, the program reads from `test_data.csv` and writes predictions into:
    ```bash
    predicted_ratings.csv
    ```
    In the form:
    ```bash
    userId,itemId,predictedRating
    ```
6. **Inspect Output:** The final predictions are clamped between 0 and 5.

## Further Observations

1. **Embedding Dimension**: A dimension of 3 for NCF was found sufficient to capture essential interaction features while maintaining speed.
2. **Hyperparameter Tuning**: Experiments revealed a need to balance model complexity (e.g., increasing SVD factors or embedding dimensions) and performance; higher complexity can lead to overfitting or slower runtime.
3. **IBCF Exclusion**: Disabling IBCF lowered accuracy slightly but drastically improved runtime, justifying its removal for this project.

Thank you for exploring this hybrid recommender system. Feel free to contribute or open issues for discussions or improvements!

## License 
This project is licensed under the MIT License. See the LICENSE file for details.

