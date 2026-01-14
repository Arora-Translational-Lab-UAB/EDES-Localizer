from __future__ import annotations
import torch

def robust_pca(M: torch.Tensor, device: str, max_iter: int = 1000, lambda_: float | None = None, tol: float = 1e-6):
    """Robust PCA (PCP) via inexact ALM, matching your notebook structure."""
    m, n = M.shape
    if lambda_ is None:
        lambda_ = 1 / (max(m, n) ** 0.5)

    M = M.to(device)
    L = torch.zeros_like(M, device=device)
    S = torch.zeros_like(M, device=device)

    norm_M = torch.norm(M, p="fro")
    Y = M / max(torch.norm(M, p=2), torch.norm(M, p=float("inf")) / lambda_)
    mu = 1.25 / torch.norm(M, p=2)
    mu_bar = mu * 1e7
    rho = 1.5

    for _ in range(max_iter):
        U, sigma, Vh = torch.linalg.svd(M - S + (1.0 / mu) * Y, full_matrices=False)
        sigma_thresh = torch.clamp(sigma - 1.0 / mu, min=0)
        L = U @ torch.diag(sigma_thresh) @ Vh

        temp = M - L + (1.0 / mu) * Y
        S = torch.sign(temp) * torch.clamp(torch.abs(temp) - lambda_ / mu, min=0)

        Z = M - L - S
        Y = Y + mu * Z

        err = torch.norm(Z, p="fro") / (norm_M + 1e-8)
        if err < tol:
            break
        mu = min(mu * rho, mu_bar)

    return L, S
