<p align="center">
  <h1 align="center">Label-Trustworthiness & Label-Continuity</h1>
</p>

**Label Trustworthiness and Continuity (Label-T&C)** is a novel measures for evaluating the reliability of cluster structure preservation in dimensionality reduction (DR) embeddings, relying on class labels. 
It addresses the shortcomings of traditional evaluation methods using class labels, which assesses the how well the classes form clusters (i.e., *cluster-label matching*; CLM) in the embeddings using clustering validation measures (CVM). Label-T&C, on the other hand, evaluates CLM in both high-dimensional and the original space, thus more accurately measure the reliability of DR emebeddings. 

Label-T quantifies the distortion caused by class compression, with a lower score indicating that points of different classes are closer in the embedding compared to the original data. Label-C evaluates distortion related to class stretching, where a lower score signifies that points of different classes are more stretched in the embedding compared to the original data.

Currently, Label-T&C is developed as a standalone python library.

## Installation & Usage

Due to anonymity issue, Label-T&C will be served via `pip` after the academic paper appears in the peer-reviewed journal or conference!!
Currently, you can clone the repository and directly call the function 

```python
import sys
sys.path.append("PATH/TO/REPOSITORY/src/")

from ltnc import ltnc
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA

# Load the Iris dataset
iris = datasets.load_iris()
raw_data = iris.data
labels = iris.target

# Perform PCA to reduce the dimensionality of the data
pca = PCA(n_components=2)
embedding = pca.fit_transform(raw_data)

# Initialize the LabelTNC class with your data
label_tnc = ltnc.LabelTNC(raw_data, embedding, labels, cvm="btw_ch")

# Run the algorithm and get the results
results = label_tnc.run()

# Access the Label-Trustworthiness (LT) and Label-Continuity (LC) scores
lt_score = results["lt"]
lc_score = results["lc"]

print("Label-Trustworthiness (LT):", lt_score)
print("Label-Continuity (LC):", lc_score)
```

`raw` is the original (raw) high-dimensional data which used to generate multidimensional projections. It should be a 2D array (or a 2D np array) with shape `(n_samples, n_dim)` where `n_samples` denotes the number of data points in dataset and `n_dim` is the original size of dimensionality (number of features). `emb` is the projected (embedded) data of `raw` (i.e., MDP result). It should be a 2D array (or a 2D np array) with shape `(n_samples, n_reduced_dim)` where `n_reduced_dim` denotes the dimensionality of projection. `labels` should be a 1d array with length `n_samples` which holds the categorical information of class labels.


## API

### `LabelTNC` class in `src/ltnc/ltnc.py`

#### `__init__(self, raw, emb, labels, cvm="btw_ch")`

Initializes the Label-TNC class.

**Parameters:**

- `raw`: numpy.ndarray, shape (n, d)
    - Original data.
- `emb`: numpy.ndarray, shape (m, d) where m < n
    - Embedding of the original data.
- `labels`: numpy.ndarray, shape (n,)
    - Labels of the original data.
- `cvm`: str, optional, default: "btw_ch"
    - Cluster validation measure to use. Currently supported: "btw_ch" (Between-dataset Calinski-Harabasz Index), "dsc" (Distance Consistency).

#### `run(self)`

Runs the algorithm and returns the score of Label-Trustworthiness (LT) and Label-Continuity (LC).

**Returns:**

- A dictionary with the following keys:
    - `lt`: Label-Trustworthiness score.
    - `lc`: Label-Continuity score.
    - `f1`: F1 score of Label-T and Label-C.
    - `raw_mat`: Original data's label-pairwise CVM matrix.
    - `emb_mat`: Embedding data's label-pairwise CVM matrix.
    - `lt_mat`: Label-pairwise Label-Trustworthiness matrix
    - `lc_mat`: Label-pairwise Label-Continuity matrix



