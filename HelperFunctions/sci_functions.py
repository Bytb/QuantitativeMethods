import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from inspect import signature
from numbers import Integral


def _event_bar_spans(events, bar_index=None):
    """Validate events and map endpoints to observed hourly candle starts."""
    if not isinstance(events, pd.DataFrame) or events.empty:
        raise ValueError('events must be a nonempty DataFrame')
    if not events.index.is_unique or events.index.hasnans:
        raise ValueError('Event identifiers must be unique and nonmissing')
    if not {'entry_time', 't1'}.issubset(events.columns):
        raise ValueError('events must contain entry_time and t1')
    bars = events.attrs.get('bar_index') if bar_index is None else bar_index
    if isinstance(bars, tuple):
        bars = pd.DatetimeIndex(bars)
    if not isinstance(bars, pd.DatetimeIndex) or bars.empty:
        raise ValueError('Supply an observed bar_index or use fresh create_events output')
    if bars.hasnans or not bars.is_unique or not bars.is_monotonic_increasing:
        raise ValueError('bar_index must be unique, increasing, and nonmissing')
    endpoints = []
    for field in ('entry_time', 't1'):
        times = pd.DatetimeIndex(events[field])
        if times.hasnans or times.tz != bars.tz:
            raise ValueError(f'{field} must be nonmissing and use the candle timezone')
        positions = bars.searchsorted(times, side='right') - 1
        safe = np.maximum(positions, 0)
        # Bars are start-labeled. Do not map a session gap to a stale candle.
        if ((positions < 0) | (times >= bars.take(safe) + pd.Timedelta(hours=1))).any():
            raise ValueError(f'{field} contains a timestamp outside observed hourly candles')
        endpoints.append(positions)
    if (pd.DatetimeIndex(events['t1']) < pd.DatetimeIndex(events['entry_time'])).any():
        raise ValueError('Event end times must not precede start times')
    starts, ends = endpoints
    difference = np.zeros(len(bars) + 1, dtype=np.int64)
    np.add.at(difference, starts, 1)
    np.add.at(difference, ends + 1, -1)
    covered = np.cumsum(difference[:-1]) > 0
    compressed = np.cumsum(covered) - 1
    return bars[covered], compressed[starts], compressed[ends]


def indicator_matrix(events, bar_index=None):
    """Sparse binary matrix: observed candles by event ID, endpoints inclusive.

    events is create_events output; optional bar_index overrides its stored
    candle index. Timestamps within a candle map to its start label. Only
    candles covered by at least one event are retained. Invalid events error.
    """
    from scipy.sparse import csc_matrix
    bars, starts, ends = _event_bar_spans(events, bar_index)
    lengths = ends - starts + 1
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    rows = np.concatenate([np.arange(start, end + 1)
                           for start, end in zip(starts, ends)])
    matrix = csc_matrix((np.ones(len(rows), dtype=np.uint8), rows, offsets),
                        shape=(len(bars), len(events)))
    return pd.DataFrame.sparse.from_spmatrix(matrix, index=bars, columns=events.index)


def average_uniqueness(ind_matrix):
    """Mean reciprocal concurrency over each event's active candles only.

    Duplicate columns represent separate sampled copies and count separately.
    """
    from scipy.sparse import csc_matrix
    if not isinstance(ind_matrix, pd.DataFrame) or ind_matrix.empty:
        raise ValueError('ind_matrix must be a nonempty binary DataFrame')
    if all(isinstance(dtype, pd.SparseDtype) and dtype.fill_value == 0
           for dtype in ind_matrix.dtypes):
        matrix = ind_matrix.sparse.to_coo().tocsc().astype(float)
    else:
        matrix = csc_matrix(ind_matrix.to_numpy(dtype=float))
    if not np.isin(matrix.data, [0, 1]).all():
        raise ValueError('Indicator entries must be zero or one')
    lengths = np.asarray(matrix.sum(axis=0)).ravel()
    if (lengths == 0).any():
        raise ValueError('Every event must cover at least one candle')
    concurrency = np.asarray(matrix.sum(axis=1)).ravel()
    reciprocal = np.divide(1., concurrency, out=np.zeros_like(concurrency),
                           where=concurrency > 0)
    return pd.Series(np.asarray(matrix.T @ reciprocal).ravel() / lengths,
                     index=ind_matrix.columns, name='average_uniqueness')


def sequential_bootstrap(events, sample_length=None, seed=None, *,
                         bar_index=None, diagnostics=False):
    """Draw event IDs with replacement, favoring conditional uniqueness.

    Main input is msft_events (create_events output). Defaults to len(events)
    draws. seed controls a local NumPy generator. Return a list in draw order.
    With diagnostics=True return (draws, diagnostics_dict), including the
    sparse indicator matrix, original uniqueness, final sampled concurrency,
    and per-draw selected probability/uniqueness. No event or bar cap is used.

    Candidate scores are mean(1 / (selected concurrency + 1)) on their
    inclusive candle spans. Prefix sums avoid rebuilding a matrix per draw.
    """
    bars, starts, ends = _event_bar_spans(events, bar_index)
    if sample_length is None:
        sample_length = len(events)
    if (not isinstance(sample_length, Integral) or isinstance(sample_length, bool)
            or sample_length < 0):
        raise ValueError('sample_length must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    concurrency = np.zeros(len(bars), dtype=np.int64)
    lengths = ends - starts + 1
    selected, records = [], []
    for draw in range(sample_length):
        prefix = np.concatenate(([0.], np.cumsum(1. / (concurrency + 1))))
        scores = (prefix[ends + 1] - prefix[starts]) / lengths
        probabilities = scores / scores.sum()
        position = int(rng.choice(len(events), p=probabilities))
        event_id = events.index[position]
        selected.append(event_id)
        if diagnostics:
            records.append(dict(draw=draw, event_id=event_id,
                                candidate_uniqueness=scores[position],
                                probability=probabilities[position]))
        concurrency[starts[position]:ends[position] + 1] += 1
    if not diagnostics:
        return selected
    matrix = indicator_matrix(events, bar_index)
    return selected, {
        'indicator_matrix': matrix,
        'average_uniqueness': average_uniqueness(matrix),
        'sample_concurrency': pd.Series(concurrency, index=bars, name='concurrency'),
        'draws': pd.DataFrame(records, columns=[
            'draw', 'event_id', 'candidate_uniqueness', 'probability']),
    }


def sample_weights(events, close, labels, *, decay=0.5,
                   class_weight='balanced', normalize_final=False):
    """Return an event-indexed diagnostic table including final_weight.

    events: cleaned, completed single-ticker events with entry_time and t1.
    close: full observed hourly closing-price Series (or one-column frame).
    labels: binary meta-label Series, or get_labels frame containing bin.
    Labels must match event IDs exactly; their order may differ.

    Concurrency and uniqueness use inclusive entry/exit candle coverage.
    Return attribution excludes the entry candle's incoming log return.
    Decay uses cumulative uniqueness in stable entry-time order, -1 < decay
    <= 1. Balanced factors use input label counts, not weighted counts.
    class_weight=None disables class factors. normalize_final optionally
    rescales the product to mean 1. Compute on training subsets only.
    Pass final_weight as model sample_weight; leave model class_weight unset.
    Invalid prices/events/labels or all-zero effective weights raise errors.
    """
    if isinstance(close, pd.DataFrame) and close.shape[1] == 1:
        close = close.iloc[:, 0]
    if not isinstance(close, pd.Series) or close.empty:
        raise ValueError('close must be a nonempty Series or one-column DataFrame')
    prices = close.to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError('Closing prices must be positive and finite')
    bars, starts, ends = _event_bar_spans(events, close.index)
    if 'ticker' in events and events['ticker'].nunique(dropna=False) != 1:
        raise ValueError('Weight one ticker at a time with its closing prices')
    if 'status' in events and not events['status'].eq('complete').all():
        raise ValueError('All input events must be completed')
    if isinstance(labels, pd.DataFrame) and 'bin' in labels:
        labels = labels['bin']
    if (not isinstance(labels, pd.Series) or not labels.index.is_unique
            or len(labels) != len(events)
            or not events.index.isin(labels.index).all()):
        raise ValueError('labels must match the input event IDs exactly')
    labels = labels.reindex(events.index)
    if not labels.isin([0, 1]).all():
        raise ValueError('Meta-labels must be nonmissing 0 or 1')
    if not np.isfinite(decay) or not -1 < decay <= 1:
        raise ValueError('decay must satisfy -1 < decay <= 1')
    if class_weight not in ('balanced', None):
        raise ValueError("class_weight must be 'balanced' or None")
    if not isinstance(normalize_final, (bool, np.bool_)):
        raise ValueError('normalize_final must be boolean')

    difference = np.zeros(len(bars) + 1, dtype=np.int64)
    np.add.at(difference, starts, 1)
    np.add.at(difference, ends + 1, -1)
    concurrency = np.cumsum(difference[:-1])
    uniqueness_prefix = np.r_[0., np.cumsum(1. / concurrency)]
    uniqueness = ((uniqueness_prefix[ends + 1] - uniqueness_prefix[starts])
                  / (ends - starts + 1))
    # Compute returns BEFORE restricting to covered bars: preserve predecessors.
    returns = np.r_[0., np.diff(np.log(prices))]
    covered_returns = returns[close.index.get_indexer(bars)]
    return_prefix = np.r_[0., np.cumsum(covered_returns / concurrency)]
    attributed = return_prefix[ends + 1] - return_prefix[starts + 1]
    absolute = np.abs(attributed)
    if absolute.sum() == 0:
        raise ValueError('All events have zero attributed return')
    return_weight = len(events) * absolute / absolute.sum()

    order = np.argsort(starts, kind='stable')
    cumulative = np.cumsum(uniqueness[order])
    fraction = cumulative / cumulative[-1]
    ordered_decay = (decay + (1 - decay) * fraction if decay >= 0
                     else np.maximum(0., (decay + fraction) / (1 + decay)))
    time_factor = np.empty(len(events))
    time_factor[order] = ordered_decay
    counts = labels.value_counts()
    class_factor = (labels.map(len(events) / (len(counts) * counts)).to_numpy()
                    if class_weight == 'balanced' else np.ones(len(events)))
    combined = return_weight * time_factor * class_factor
    if combined.sum() == 0:
        raise ValueError('All combined weights are zero after decay')
    final = combined * len(events) / combined.sum() if normalize_final else combined
    return pd.DataFrame({
        'label': labels, 'attributed_return': attributed,
        'absolute_attributed_return': absolute, 'average_uniqueness': uniqueness,
        'return_weight': return_weight, 'time_factor': time_factor,
        'class_factor': class_factor, 'combined_weight': combined,
        'final_weight': final,
    }, index=events.index)


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
