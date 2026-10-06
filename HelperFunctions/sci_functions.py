import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from inspect import signature


def return_normality(data, input_type="prices", plot_qq=True,
                     plot_distribution=True, significance_level=5.0, bins=50):
    """Test each ticker's returns for normality and optionally plot diagnostics.

    data : pandas.DataFrame
        Numeric columns, one per ticker, in chronological row order.
    input_type : {"prices", "pct_change"}
        Prices are converted with pct_change(fill_method=None) * 100.
        Supplied percent changes are used as-is (percentage points, e.g. 1 = 1%).
    plot_qq, plot_distribution : bool
        Independently enable the Q–Q plot and histogram for each ticker.
    significance_level : float
        AD rejection threshold in percent: 15, 10, 5, 2.5, or 1.

    Returns a DataFrame indexed by ticker with sample counts, AD statistics,
    critical values (older SciPy) or p-values (newer SciPy), rejection decisions,
    and status. Missing/nonfinite returns
    are excluded independently per ticker. Insufficient or constant samples
    have no test decision. The null is normality with fitted mean and variance;
    failure to reject does not establish normality. AD assumes independent
    observations. Q–Q bounds are approximate pointwise 95% visual guides.
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("data must be a nonempty pandas DataFrame.")
    if not data.columns.is_unique:
        raise ValueError("Ticker column names must be unique.")
    if input_type not in ("prices", "pct_change"):
        raise ValueError("input_type must be 'prices' or 'pct_change'.")
    if significance_level not in (15, 10, 5, 2.5, 1):
        raise ValueError("significance_level must be 15, 10, 5, 2.5, or 1 percent.")
    if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in data.dtypes):
        raise TypeError("All ticker columns must be numeric.")
    values = data.astype(float)
    if input_type == "prices":
        if not data.index.is_monotonic_increasing:
            raise ValueError("Price rows must be in chronological order.")
        if (values <= 0).any().any() or np.isinf(values.to_numpy()).any():
            raise ValueError("Prices must be positive and finite, or missing.")
        values = values.pct_change(fill_method=None) * 100

    rows = []
    for ticker in values.columns:
        sample = values[ticker].to_numpy()
        sample = sample[np.isfinite(sample)]
        n = len(sample)
        row = dict(ticker=ticker, n_obs=n, statistic=np.nan,
                   significance_level=significance_level, critical_value=np.nan,
                   p_value=np.nan,
                   reject_normality=pd.NA, status="insufficient data")
        if n < 3:
            rows.append(row)
            continue
        std = sample.std(ddof=1)
        if std == 0:
            row["status"] = "constant returns"
            rows.append(row)
            continue
        if "method" in signature(stats.anderson).parameters:
            result = stats.anderson(sample, dist="norm", method="interpolate")
            row.update(p_value=float(result.pvalue),
                       reject_normality=bool(result.pvalue < significance_level / 100))
        else:
            result = stats.anderson(sample, dist="norm")
            level_index = np.flatnonzero(
                np.isclose(result.significance_level, significance_level)
            )[0]
            critical = float(result.critical_values[level_index])
            row.update(critical_value=critical,
                       reject_normality=bool(result.statistic > critical))
        row.update(statistic=float(result.statistic), status="ok")
        rows.append(row)

        if plot_distribution:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(sample, bins=bins, density=True, edgecolor="white",
                    color="#65aaa8", alpha=0.7, label="Observed returns")
            mean = sample.mean()
            fitted_std = sample.std(ddof=0)
            x = np.linspace(min(sample.min(), mean - 4 * fitted_std),
                            max(sample.max(), mean + 4 * fitted_std), 500)
            ax.plot(x, stats.norm.pdf(x, loc=mean, scale=fitted_std),
                    color="#b87873", linewidth=2, label="Fitted normal distribution")
            ax.set(title=f"{ticker} — Distribution of Percent Changes",
                   xlabel="Percent change (%)", ylabel="Probability density")
            ax.legend(frameon=False)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            plt.show()
        if plot_qq:
            observed = np.sort((sample - sample.mean()) / std)
            order = np.arange(1, n + 1)
            theoretical = stats.norm.ppf((order - 0.5) / n)
            slope, intercept = np.polyfit(theoretical, observed, 1)
            lower = intercept + slope * stats.norm.ppf(
                stats.beta.ppf(0.025, order, n + 1 - order))
            upper = intercept + slope * stats.norm.ppf(
                stats.beta.ppf(0.975, order, n + 1 - order))
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.scatter(theoretical, observed, marker="+", color="#65aaa8",
                       s=28, linewidths=0.9, label="Standardized returns")
            ax.plot(theoretical, intercept + slope * theoretical,
                    color="#b87873", linewidth=1.3, label="Fitted reference line")
            ax.plot(theoretical, lower, color="#b87873", linewidth=1.3,
                    label="Approximate 95% pointwise bounds")
            ax.plot(theoretical, upper, color="#b87873", linewidth=1.3)
            ax.axhline(0, color="#aaaaaa", linewidth=1.3)
            ax.axvline(0, color="#aaaaaa", linewidth=1.3)
            decision = "Reject normality" if row["reject_normality"] else "Do not reject normality"
            ax.set(title=f"{ticker} — Quantile–Quantile Plot\n"
                         f"AD = {result.statistic:.3f}; {decision} at {significance_level:g}%",
                   xlabel="Theoretical Quantiles",
                   ylabel="Sample Quantiles (standardized returns)")
            ax.grid(True, color="#eeeeee", alpha=0.6)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(length=0)
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            plt.show()
    summary = pd.DataFrame(rows).set_index("ticker")
    summary["reject_normality"] = summary["reject_normality"].astype("boolean")
    return summary


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
