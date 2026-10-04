import numpy as np
import pandas as pd


def calculate_residual_metrics(stock_returns, benchmark_returns):
    """
    Calculate beta, residual returns, residual return variance,
    and residual return volatility for each stock relative to a benchmark.

    Parameters
    ----------
    stock_returns : pd.DataFrame
        DataFrame of stock returns.
        Rows should be dates and columns should be ticker symbols.

    benchmark_returns : pd.Series
        Series of benchmark returns indexed by date.

    Returns
    -------
    summary : pd.DataFrame
        One row per stock containing:
        - Alpha
        - Beta
        - Residual Variance
        - Residual Volatility

    residual_returns : pd.DataFrame
        Residual return time series for each stock.
    """

    # Align stock and benchmark returns by date
    data = stock_returns.join(
        benchmark_returns.rename("Benchmark"),
        how="inner"
    ).dropna()

    benchmark = data["Benchmark"]
    stocks = data.drop(columns="Benchmark")

    # Benchmark variance
    benchmark_variance = benchmark.var()

    summary_rows = {}
    residual_returns = pd.DataFrame(index=data.index)

    for ticker in stocks.columns:

        stock = stocks[ticker]

        # Beta = Cov(stock, benchmark) / Var(benchmark)
        beta = stock.cov(benchmark) / benchmark_variance

        # Alpha from the standard market model
        alpha = stock.mean() - beta * benchmark.mean()

        # Predicted return from benchmark exposure
        predicted_return = alpha + beta * benchmark

        # Residual return
        residual = stock - predicted_return

        # Store residual return series
        residual_returns[ticker] = residual

        # Residual variance and volatility
        residual_variance = residual.var()
        residual_volatility = residual.std()

        summary_rows[ticker] = {
            "Alpha": alpha,
            "Beta": beta,
            "Residual Variance": residual_variance,
            "Residual Volatility": residual_volatility
        }

    summary = pd.DataFrame.from_dict(
        summary_rows,
        orient="index"
    )

    summary.index.name = "Ticker"

    return summary, residual_returns