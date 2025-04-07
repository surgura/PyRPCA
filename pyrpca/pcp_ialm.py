import numpy.typing as npt
from typing import Tuple
import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd


def rpca_pcp_ialm(
    observations: npt.ArrayLike,
    sparsity_factor: float,
    max_iter: int = 1000,
    mu: float | None = None,
    mu_upper_bound: float | None = None,
    rho: float = 1.5,
    tol: float = 1e-7,
    verbose: bool = True,
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    """
    Solve the Principal Component Pursuit (PCP) convex relaxation of Robust PCA using the Inexact Augmented Lagrange Multiplier (IALM) method.

    See README for algorithmic details and references.

    Mu is updated every loop by multiplying it by `rho` until reaching `mu_upper_bound`.

    Parameters:
        observations: The m x n input matrix to decompose ('D' in the IALM paper).
        sparsity_factor: Weight on the sparse term in the objective ('lambda' in the IALM paper).
        max_iter: Maximum number of iterations to perform.
        mu: Initial value for the penalty parameter. If None, defaults to 1/spectral norm of observations.
        mu_upper_bound: Maximum allowed value for `mu`. If None, defaults to `mu * 1e7`.
        rho: Multiplicative factor to increase `mu` in each iteration.
        tol: Tolerance for stopping criterion (relative Frobenius norm of the residual).
        verbose: If True, print status and debug information during optimization.

    Returns:
        low_rank_component: The recovered low-rank matrix ('A' in the IALM paper).
        sparse_component: The recovered sparse matrix ('E' in the IALM paper).
    """
    if mu is None:
        mu = float(1.25 / norm(observations, ord=2))
    if mu_upper_bound is None:
        mu_upper_bound = mu * 1e7

    norm_fro_obs = norm(observations, ord="fro")

    dual = observations / np.maximum(
        norm(observations, ord=2), norm(observations, ord=np.inf) / sparsity_factor
    )
    sparse = np.zeros_like(observations)

    i = 0
    while True:
        # compute next iteration of a
        u, s, v = svd(observations - sparse + 1.0 / mu * dual, full_matrices=False)
        s_thresholded = np.maximum(s - 1.0 / mu, 0)
        low_rank = (u * s_thresholded) @ v

        # compute next iteration of e
        residual_for_sparse = observations - low_rank + 1.0 / mu * dual
        sparse = np.sign(residual_for_sparse) * np.maximum(
            np.abs(residual_for_sparse) - sparsity_factor / mu, 0
        )

        # calculate error
        residual = observations - low_rank - sparse
        err = norm(residual, ord="fro") / norm_fro_obs

        i += 1

        if verbose:
            print(f"iter {i:<4} | err {err:<25} | mu {mu:<25}")

        if err < tol:
            if verbose:
                print("Finished optimization. Error smaller than tolerance.")
            break
        if i == max_iter:
            if verbose:
                print("Finized optimization. Max iterations reached.")
            break

        # update dual and mu
        dual = dual + mu * (residual)
        mu = min(mu * rho, mu_upper_bound)
    return low_rank, sparse
