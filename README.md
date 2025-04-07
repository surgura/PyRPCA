# Robust principal component analysis for Python

This package provides algorithms to solve the Robust Principal Component Analysis (RPCA) problem, as presented by Candès et al.[[1]](#candes2011).
Currently, a single algorithm is implemented: it solves the Principal Component Pursuit (PCP) convex relaxation of RPCA from the same paper, using the Inexact Augmented Lagrange Multiplier (IALM) method from Lin et al.[[2]](#lin2011)[[3]](#lin2013).

## Example
```python
from pyrpca import rpca_pcp_ialm

# given an m x n data matrix
data = ...

# decide on sparsity factor.
# this parameter is also commonly known as 'lambda'.
sparsity_factor = 1.0 / numpy.sqrt(max(data.shape))

# run the ialm algorithm.
low_rank, sparse = rpca_pcp_ialm(data, sparsity_factor)
```

## Installing
```shell
pip install pyrpca
```

## Feature requests and contributing
Pull requests and feature requests are welcome. The current version is minimal and suits my personal needs, but feel free to make suggestions. Even if the repository looks inactive, I will still respond :)

## References
1. <a name="candes2011"></a> [Emmanuel J. Candès, Xiaodong Li, Yi Ma, John Wright. Robust principal component analysis? Association for Computing Machinery 2011.](https://doi.org/10.1145/1970392.1970395) (preprint on [arXiv](https://doi.org/10.48550/arXiv.0912.3599))
2. <a name="lin2011"></a> [Zhouchen Lin, Risheng Liu, Zhixun Su. Linearized Alternating Direction Method with Adaptive Penalty for Low-Rank Representation. arXiv 2011.](https://doi.org/10.48550/arXiv.1109.0367)
3. <a name="lin2013"></a> [Zhouchen Lin, Minming Chen, Yi Ma. The Augmented Lagrange Multiplier Method for Exact Recovery of Corrupted Low-Rank Matrices, V3. arXiv 2013.](https://doi.org/10.48550/arXiv.1009.5055)

## Acknowledgements  
Appreciation is due to various other Python implementations of RPCA that served as inspiration for this project. Below is a non-exhaustive list:

- https://github.com/2020leon/rpca  
- https://github.com/dganguli/robust-pca  
- https://github.com/weilinear/PyRPCA  
- https://github.com/loiccoyle/RPCA
