<p align="center">
  <h1 align="center">Label-Trustworthiness & Label-Continuity</h1>
</p>

**Label Trustworthiness and Continuity (Label-T&C)** is a novel measures for evaluating the reliability of cluster structure preservation in dimensionality reduction (DR) embeddings, relying on class labels. 
It addresses the shortcomings of traditional evaluation methods using class labels, which assesses the how well the classes form clusters (i.e., *cluster-label matching*; CLM) in the embeddings using clustering validation measures (CVM). Label-T&C, on the other hand, evaluates CLM in both high-dimensional and the original space, thus more accurately measure the reliability of DR emebeddings. 

Label-T quantifies the distortion caused by class compression, with a lower score indicating that points of different classes are closer in the embedding compared to the original data. Label-C evaluates distortion related to class stretching, where a lower score signifies that points of different classes are more stretched in the embedding compared to the original data.

Currently, Label-T&C is developed as a standalone python library.

### Installation & Usage

Due to anonymity issue, Label-T&C will be served via `pip` after the academic paper appears in the peer-reviewed journal or conference!!
Currently, you can clone the repository and directly call the function 

```python
import sys
sys.path.append("PATH/TO/REPOSITORY/")

from ltnc import ltnc

ltnc_obj = ltnc.LabelTNC(
  raw=raw, emb=emb, labels=labels, cvm="btw_ch"
)
result = ltnc_obj.run()

```
`raw` is the original (raw) high-dimensional data which used to generate multidimensional projections. It should be a 2D array (or a 2D np array) with shape `(n_samples, n_dim)` where `n_samples` denotes the number of data points in dataset and `n_dim` is the original size of dimensionality (number of features). `emb` is the projected (embedded) data of `raw` (i.e., MDP result). It should be a 2D array (or a 2D np array) with shape `(n_samples, n_reduced_dim)` where `n_reduced_dim` denotes the dimensionality of projection. `labels` should be a 1d array with length `n_samples` which holds the categorical information of class labels.


### API
