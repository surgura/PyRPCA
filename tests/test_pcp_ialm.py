import numpy as np
from pyrpca import rpca_pcp_ialm
from numpy.linalg import norm
from scipy.sparse import random as sparse_random


def test_rpca_separates_low_rank_and_sparse():
    np.random.seed(0)
    m, n, rank = 500, 400, 5

    # create low-rank matrix A
    u = np.random.randn(m, rank)
    v = np.random.randn(rank, n)
    low_rank = u @ v

    # create sparse matrix E
    sparse = sparse_random(
        m, n, density=0.1, format="csr", data_rvs=np.random.randn
    ).toarray()

    # create observation matrix
    observations = low_rank + sparse

    # Run RPCA
    low_rank_recovered, sparse_recovered = rpca_pcp_ialm(
        observations,
        sparsity_factor=1.0 / np.sqrt(max(observations.shape)),
    )

    # check that the reconstruction is close
    reconstruction_error = norm(
        observations - (low_rank_recovered + sparse_recovered), ord="fro"
    ) / norm(observations, ord="fro")
    assert reconstruction_error < 1e-6, (
        f"Reconstruction error too high: {reconstruction_error}"
    )

    # check that recovered matrices are low rank and sparse
    approx_rank = np.linalg.matrix_rank(low_rank_recovered, tol=1e-3)
    sparsity = np.count_nonzero(sparse_recovered) / sparse_recovered.size

    assert approx_rank <= rank + 2, f"Recovered A not low rank: {approx_rank}"
    assert sparsity < 0.2, f"Recovered E not sparse: sparsity={sparsity}"
