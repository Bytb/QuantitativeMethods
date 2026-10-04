import numpy as np
import matplotlib.pyplot as plt


def Covariance_PCA(cov_matrix, n_components=2):
    """
    Decompose an asset covariance matrix and plot:
      1. Each asset's loading on the top 2 or 3 principal components.
      2. The percentage of total variance explained by every component.

    Returns
    -------
    eigenvalues : np.ndarray
        Eigenvalues in descending order.
    eigenvectors : np.ndarray
        Eigenvectors in descending order, stored as columns.
        eigenvectors[:, 0] is PC1.
    variance_explained : np.ndarray
        Percentage of total variance explained by each component.
    """
    C = np.asarray(cov_matrix, dtype=float)

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("cov_matrix must be square.")

    n_assets = C.shape[0]
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3.")
    if n_assets < n_components:
        raise ValueError(
            f"Need at least {n_components} assets for this plot."
        )
    if not np.all(np.isfinite(C)):
        raise ValueError("cov_matrix contains missing or infinite values.")
    if not np.allclose(C, C.T, rtol=1e-8, atol=1e-10):
        raise ValueError("cov_matrix must be symmetric.")

    asset_names = (
        [str(name) for name in cov_matrix.columns]
        if hasattr(cov_matrix, "columns")
        else [f"Asset {i + 1}" for i in range(n_assets)]
    )

    # Covariance-matrix PCA: eigenvectors are returned as COLUMNS.
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # A valid covariance matrix cannot have materially negative eigenvalues.
    tolerance = 1e-8 * max(1.0, np.max(np.abs(eigenvalues)))
    if eigenvalues[-1] < -tolerance:
        raise ValueError(
            "cov_matrix has a negative eigenvalue and may not be "
            "a valid covariance matrix."
        )

    # Remove tiny negative values caused by floating-point rounding.
    eigenvalues = np.clip(eigenvalues, 0, None)

    total_variance = eigenvalues.sum()
    if total_variance == 0:
        raise ValueError("cov_matrix has zero total variance.")

    variance_explained = 100 * eigenvalues / total_variance
    asset_loadings = eigenvectors[:, :n_components]

    fig = plt.figure(figsize=(16, 6))

    # Plot assets by their entries in the leading eigenvectors.
    if n_components == 2:
        ax = fig.add_subplot(1, 2, 1)
        ax.scatter(
            asset_loadings[:, 0],
            asset_loadings[:, 1],
            s=65,
            color="steelblue"
        )

        for i, name in enumerate(asset_names):
            ax.annotate(
                name,
                (asset_loadings[i, 0], asset_loadings[i, 1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9
            )

        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlabel(f"PC1 loading ({variance_explained[0]:.1f}%)")
        ax.set_ylabel(f"PC2 loading ({variance_explained[1]:.1f}%)")

    else:
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax.scatter(
            asset_loadings[:, 0],
            asset_loadings[:, 1],
            asset_loadings[:, 2],
            s=65,
            color="steelblue"
        )

        for i, name in enumerate(asset_names):
            ax.text(
                asset_loadings[i, 0],
                asset_loadings[i, 1],
                asset_loadings[i, 2],
                f" {name}",
                fontsize=9
            )

        ax.set_xlabel(f"PC1 loading ({variance_explained[0]:.1f}%)")
        ax.set_ylabel(f"PC2 loading ({variance_explained[1]:.1f}%)")
        ax.set_zlabel(f"PC3 loading ({variance_explained[2]:.1f}%)")

    ax.set_title(f"Asset Loadings on the Top {n_components} PCs")

    # Plot the variance share for ALL eigenvectors.
    ax_bar = fig.add_subplot(1, 2, 2)
    component_numbers = np.arange(1, n_assets + 1)

    ax_bar.bar(component_numbers, variance_explained, color="steelblue")
    ax_bar.set_xlabel("Principal component")
    ax_bar.set_ylabel("Variance explained (%)")
    ax_bar.set_title("Variance Explained by Each Component")
    ax_bar.set_xticks(component_numbers)
    ax_bar.set_xticklabels(
        [f"PC{i}" for i in component_numbers],
        rotation=90 if n_assets > 12 else 0
    )
    ax_bar.set_ylim(0, max(variance_explained) * 1.1)

    plt.tight_layout()
    plt.show()

    return eigenvalues, eigenvectors, variance_explained