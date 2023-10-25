# K-Means Clustering with Automatic k Selection

This Python code performs K-Means clustering on synthetic datasets and automatically selects the optimal number of clusters (k) using the elbow method. It also compares the results with scikit-learn's K-Means implementation.

## Features

- Generates synthetic datasets (blobs or moon) with customizable parameters.
- Implements K-Means clustering with automatic k selection.
- Visualizes the clustering results and cost function.
- Compares the results with scikit-learn's K-Means.

## Prerequisites

- Python 3.x
- Install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```

## Command-Line Arguments
The script supports the following command-line arguments for customization:

--n_samples (int, default: 1000): Specify the number of samples for the dataset. Must be an integer value greater than 0.

--dataset (str, default: 'blobs'): Specify the dataset type. Choose from convex 'blobs' or  non-coonvex'moon'.

--centers (int, default: 3): Specify the number of centers for the 'blobs' dataset. Must be an integer value greater than 0, maximum 12.

--noise (float, default: 0.1): Specify the noise for the dataset when using the 'moon' dataset. Must be a float value between 0 and 1.

--shuffle (bool, default: True): Specify whether to shuffle the dataset. Use 'True' or 'False' as a Boolean value.

--cluster_std (float, default: 0.3): Specify the cluster standard deviation for the 'blobs' dataset. Must be a float value between 0 and 1.

--k (int, default: 0): Specify the number of clusters manually if you don't want to use the automatic elbow method. Must be an integer value greater than 0.

Examples:

```bash
python kmeans_auto_k.py --n_samples 2000 --dataset moon --centers 4
```

```bash
python kmeans_auto_k.py --n_samples 1000 --cluster_std 0.35
```
