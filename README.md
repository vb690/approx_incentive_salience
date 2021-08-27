## Approximating Attributed Incentive Salience in Large Scale Scenarios
### A Representation Learning Approach Based on Artificial Neural Networks

## Motivation

## Data

Due to commerical sensitivity and  [data protection regulations](https://en.wikipedia.org/wiki/General_Data_Protection_Regulation) we are not allowed to pubblicly release the data employed in the present work.  

We will now provide an illustrative example on the data format expected by this project.

| user_id | session_order | absence | session_played_time | session_time | activity | maximum_sessions | context |
|:-------:|:-------------:|:-------:|:-------------------:|:------------:|:--------:|:----------------:|:-------:|
|   XXX   |       1       |    10   |          30         |      35      |    14    |         6        |   lis   |
|   XXX   |       2       |    20   |          5          |      15      |     1    |         6        |   lis   |
|   XXX   |       3       |    10   |          6          |      12      |     3    |         6        |   lis   |
|   YYY   |       1       |    40   |          4          |       7      |    43    |         2        |   hms   |
|   YYY   |       2       |    50   |          35         |      38      |    12    |         2        |   hms   |
|   ZZZ   |       1       |    12   |          21         |      21      |     3    |         2        |   jc3   |
|   ZZZ   |       2       |    13   |          8          |       9      |    26    |         2        |   jc3   |

## Features

### Model for Saliency Estimation and Prediction of Future Interactions Intensity

### Architecture

<p align="center">
  <img width="800" height="400" src="https://github.com/vb690/approx_incentive_salience/blob/main/readme_pic/rnn_architecture.png">
</p>

### Hyper-parameters Tuning

### Embedding Extraction

## How to Use

### Scripts
1. Data Preparation
2. Models Optimization
3. Models Comparison
4. Embedding Extraction
    * Dimensionality Reduction
    * Alligned Dimesnionality Reduction
    * Data Container

### Notebooks
1. [Exploratory Data Analysis](https://htmlpreview.github.io/?https://github.com/vb690/approx_incentive_salience/blob/main/notebooks_html/EDA_analysis.html)
1. [Models Performance Analysis](https://htmlpreview.github.io/?https://github.com/vb690/approx_incentive_salience/blob/main/notebooks_html/model_performance_analysis.html)
2. Embeddings
    * [UMAP Analyses](https://htmlpreview.github.io/?https://github.com/vb690/approx_incentive_salience/blob/main/notebooks_html/embedding_analysis.html)
    * [Activations Analyses](https://htmlpreview.github.io/?https://github.com/vb690/approx_incentive_salience/blob/main/notebooks_html/activation_analysis.html) 
    * [Temporal UMAP Analyses](https://htmlpreview.github.io/?https://github.com/vb690/approx_incentive_salience/blob/main/notebooks_html/umap_traces_visualization.html)
3. [Partitioning and Behavioural Profiles Extraction](https://htmlpreview.github.io/?https://github.com/vb690/approx_incentive_salience/blob/main/notebooks_html/cluster_analysis.html)

## Results

### Perfromance Analysis

### Representation Analysis

### Partition Analysis

