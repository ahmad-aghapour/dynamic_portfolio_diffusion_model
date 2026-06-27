from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .generation import generate_next_month_scenarios


def solve_markowitz_long_only(mu, cov, risk_aversion=5.0, cov_jitter=1e-6, weight_cap: Optional[float] = None):
    """
    Solve long-only Markowitz allocation.

    Objective:
        min_w -mu'w + risk_aversion / 2 * w'Cov w

    Constraints:
        w_i >= 0, sum_i w_i = 1, and optionally w_i <= weight_cap.
    """
    mu = np.asarray(mu, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    n = len(mu)

    cov = 0.5 * (cov + cov.T)
    cov = cov + cov_jitter * np.eye(n)

    upper = 1.0 if weight_cap is None else float(weight_cap)

    try:
        import cvxpy as cp

        w = cp.Variable(n)
        objective = cp.Minimize(-mu @ w + (risk_aversion / 2.0) * cp.quad_form(w, cov))
        constraints = [w >= 0, cp.sum(w) == 1, w <= upper]
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            problem.solve(solver=cp.SCS, verbose=False)

        if w.value is None:
            raise ValueError("CVXPY failed to find solution.")
        weights = np.asarray(w.value).reshape(-1)

    except Exception as exc:
        print("CVXPY failed. Using scipy fallback:", exc)
        from scipy.optimize import minimize

        def objective_fn(w):
            return -mu @ w + (risk_aversion / 2.0) * (w @ cov @ w)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, upper) for _ in range(n)]
        x0 = np.ones(n) / n

        result = minimize(objective_fn, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = result.x if result.success else x0
        if not result.success:
            print("Scipy optimizer failed:", result.message)

    weights = np.maximum(weights, 0.0)
    if weight_cap is not None:
        weights = np.minimum(weights, weight_cap)
    weights = weights / weights.sum()
    return weights


def run_generative_markowitz_backtest(
    model,
    monthly_returns_df,
    test_target_df,
    mean,
    std,
    cfg,
    num_scenarios=500,
    risk_aversion=5.0,
    eta=1.0,
    cov_shrinkage=0.05,
    weight_cap: Optional[float] = None,
):
    """
    Rolling backtest using generated next-month scenarios.

    For every test date:
      1. take the previous context_length real monthly returns;
      2. standardize using training mean/std;
      3. generate next-month scenarios;
      4. convert scenarios back to raw return scale;
      5. compute scenario mean/covariance;
      6. solve long-only Markowitz;
      7. apply weights to the actual next-month return.

    The returned dictionary includes scenario tensors for downstream RL use.
    """
    model.eval()

    full_raw = monthly_returns_df.values.astype(np.float32)
    full_dates = monthly_returns_df.index
    test_dates = test_target_df.index

    mean_vec = mean.reshape(-1)
    std_vec = std.reshape(-1)

    n_assets = monthly_returns_df.shape[1]
    equal_weight = np.ones(n_assets) / n_assets

    dates = []
    gm_returns = []
    ew_returns = []
    gm_weights = []
    scenario_means = []
    scenario_covs = []

    scenarios_raw_list = []
    scenarios_std_list = []
    actual_next_returns_list = []
    history_windows_raw_list = []
    history_windows_std_list = []

    for date in test_dates:
        t = full_dates.get_loc(date)
        start = t - cfg.context_length
        end = t
        if start < 0:
            continue

        history_raw = full_raw[start:end]
        actual_next_raw = full_raw[t]
        history_standardized = (history_raw - mean_vec) / std_vec

        scenarios_std = generate_next_month_scenarios(
            model=model,
            history_window_standardized=history_standardized,
            num_scenarios=num_scenarios,
            eta=eta,
            clip_x0=None,
        )
        scenarios_raw = scenarios_std.numpy() * std_vec + mean_vec

        scenarios_std_list.append(scenarios_std.numpy().copy())
        scenarios_raw_list.append(scenarios_raw.copy())
        actual_next_returns_list.append(actual_next_raw.copy())
        history_windows_raw_list.append(history_raw.copy())
        history_windows_std_list.append(history_standardized.copy())

        mu_hat = scenarios_raw.mean(axis=0)
        cov_hat = np.cov(scenarios_raw.T)

        diag_cov = np.diag(np.diag(cov_hat))
        cov_hat = (1.0 - cov_shrinkage) * cov_hat + cov_shrinkage * diag_cov

        w_gm = solve_markowitz_long_only(
            mu=mu_hat,
            cov=cov_hat,
            risk_aversion=risk_aversion,
            cov_jitter=1e-6,
            weight_cap=weight_cap,
        )

        r_gm = float(w_gm @ actual_next_raw)
        r_ew = float(equal_weight @ actual_next_raw)

        dates.append(date)
        gm_returns.append(r_gm)
        ew_returns.append(r_ew)
        gm_weights.append(w_gm)
        scenario_means.append(mu_hat)
        scenario_covs.append(cov_hat)

    return {
        "dates": pd.DatetimeIndex(dates),
        "gm_returns": np.asarray(gm_returns),
        "ew_returns": np.asarray(ew_returns),
        "gm_weights": np.asarray(gm_weights),
        "scenario_means": np.asarray(scenario_means),
        "scenario_covs": np.asarray(scenario_covs),
        "scenarios_raw": np.asarray(scenarios_raw_list),
        "scenarios_standardized": np.asarray(scenarios_std_list),
        "actual_next_returns": np.asarray(actual_next_returns_list),
        "history_windows_raw": np.asarray(history_windows_raw_list),
        "history_windows_standardized": np.asarray(history_windows_std_list),
    }


def make_backtest_dataframe(backtest_results):
    """Build return and wealth dataframe from backtest_results."""
    bt_df = pd.DataFrame(
        {
            "Generative_Markowitz_Return": backtest_results["gm_returns"],
            "Equal_Weight_Return": backtest_results["ew_returns"],
        },
        index=backtest_results["dates"],
    )
    bt_df["Generative_Markowitz_Wealth"] = (1.0 + bt_df["Generative_Markowitz_Return"]).cumprod()
    bt_df["Equal_Weight_Wealth"] = (1.0 + bt_df["Equal_Weight_Return"]).cumprod()
    return bt_df


def max_drawdown(wealth):
    wealth = np.asarray(wealth, dtype=np.float64)
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1.0
    return drawdown.min()


def performance_stats(monthly_returns, wealth=None, periods_per_year=12):
    r = np.asarray(monthly_returns, dtype=np.float64)
    if wealth is None:
        wealth = np.cumprod(1.0 + r)

    n_months = len(r)
    n_years = n_months / periods_per_year
    final_wealth = wealth[-1]
    annual_return = final_wealth ** (1.0 / n_years) - 1.0
    annual_volatility = r.std(ddof=0) * np.sqrt(periods_per_year)
    sharpe = annual_return / annual_volatility if annual_volatility > 1e-12 else np.nan

    downside_returns = r[r < 0]
    if len(downside_returns) > 1:
        downside_volatility = downside_returns.std(ddof=0) * np.sqrt(periods_per_year)
        sortino = annual_return / downside_volatility if downside_volatility > 1e-12 else np.nan
    else:
        downside_volatility = np.nan
        sortino = np.nan

    return {
        "Months": n_months,
        "Final Wealth": final_wealth,
        "Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_drawdown(wealth),
        "Monthly Mean": r.mean(),
        "Monthly Std": r.std(ddof=0),
        "Best Month": r.max(),
        "Worst Month": r.min(),
    }


def compute_rebalance_turnover(weights, returns):
    """
    Compute monthly rebalance turnover from target weights and realized returns.

    turnover[t] is sum(abs(target_weight_t - drifted_weight_before_trade_t)).
    """
    weights = np.asarray(weights, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    T, _ = weights.shape
    turnover = np.zeros(T)
    turnover[0] = np.sum(np.abs(weights[0]))

    for t in range(1, T):
        prev_w = weights[t - 1]
        prev_r = returns[t - 1]
        prev_port_ret = float(prev_w @ prev_r)
        drifted_w = prev_w * (1.0 + prev_r) / (1.0 + prev_port_ret)
        turnover[t] = np.sum(np.abs(weights[t] - drifted_w))
    return turnover


def build_weight_dataframe(backtest_results, asset_names):
    return pd.DataFrame(backtest_results["gm_weights"], index=backtest_results["dates"], columns=asset_names)


def build_average_weights_dataframe(weights_df):
    n_assets = weights_df.shape[1]
    return pd.DataFrame(
        {
            "Industry": list(weights_df.columns),
            "Average_GM_Weight": weights_df.mean(axis=0).values,
            "Equal_Weight": np.ones(n_assets) / n_assets,
        }
    ).sort_values("Average_GM_Weight", ascending=False)


def save_scenario_dataset_for_rl(backtest_results, asset_names, output_dir="outputs", prefix="ff49_diffusion"):
    """
    Save generated scenario dataset for reinforcement learning.

    Outputs:
      - {prefix}_scenarios_for_rl.npz: compact tensors
      - {prefix}_scenarios_for_rl_panel.csv: one row per date x scenario
      - {prefix}_actual_next_returns_for_rl.csv: realized next-month returns
      - {prefix}_state_windows_for_rl.csv: previous context window per date
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_dates = backtest_results["dates"]
    asset_names = list(asset_names)
    scenarios_raw = backtest_results["scenarios_raw"]
    scenarios_std = backtest_results["scenarios_standardized"]
    actual_next_returns = backtest_results["actual_next_returns"]
    history_raw = backtest_results["history_windows_raw"]
    history_std = backtest_results["history_windows_standardized"]
    gm_weights = backtest_results["gm_weights"]

    npz_path = output_dir / f"{prefix}_scenarios_for_rl.npz"
    np.savez_compressed(
        npz_path,
        scenarios_raw=scenarios_raw,
        scenarios_standardized=scenarios_std,
        dates=scenario_dates.astype(str).to_numpy(),
        asset_names=np.asarray(asset_names),
        actual_next_returns=actual_next_returns,
        history_windows_raw=history_raw,
        history_windows_standardized=history_std,
        gm_weights=gm_weights,
        scenario_means=backtest_results["scenario_means"],
        scenario_covs=backtest_results["scenario_covs"],
    )

    T, S, N = scenarios_raw.shape
    scenario_panel_df = pd.DataFrame(scenarios_raw.reshape(T * S, N), columns=asset_names)
    scenario_panel_df.insert(0, "scenario_id", np.tile(np.arange(S), T))
    scenario_panel_df.insert(0, "date", np.repeat(scenario_dates.astype(str).to_numpy(), S))
    panel_path = output_dir / f"{prefix}_scenarios_for_rl_panel.csv"
    scenario_panel_df.to_csv(panel_path, index=False)

    actual_next_returns_df = pd.DataFrame(actual_next_returns, index=scenario_dates, columns=asset_names)
    actual_path = output_dir / f"{prefix}_actual_next_returns_for_rl.csv"
    actual_next_returns_df.to_csv(actual_path)

    state_rows = []
    context_length = history_raw.shape[1]
    for t, date in enumerate(scenario_dates):
        for lag in range(context_length):
            row = {"date": date, "lag": lag - context_length + 1}
            for j, asset in enumerate(asset_names):
                row[asset] = history_raw[t, lag, j]
            state_rows.append(row)
    state_windows_df = pd.DataFrame(state_rows)
    states_path = output_dir / f"{prefix}_state_windows_for_rl.csv"
    state_windows_df.to_csv(states_path, index=False)

    return {
        "npz": npz_path,
        "scenario_panel_csv": panel_path,
        "actual_returns_csv": actual_path,
        "state_windows_csv": states_path,
    }

def _scenario_mean_cov(scenarios_t, cov_shrinkage=0.0, ridge=1e-6):
    scenarios_t = np.asarray(scenarios_t, dtype=np.float64)
    mu = scenarios_t.mean(axis=0)
    cov = np.cov(scenarios_t, rowvar=False)
    cov = 0.5 * (cov + cov.T)

    if cov_shrinkage > 0.0:
        diag_cov = np.diag(np.diag(cov))
        cov = (1.0 - cov_shrinkage) * cov + cov_shrinkage * diag_cov

    cov = cov + ridge * np.eye(cov.shape[0])
    return mu, cov


def _apply_weight_cap_and_normalize(weights, weight_cap=None):
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 0.0)

    if weight_cap is None:
        total = weights.sum()
        return weights / total if total > 0.0 else np.ones_like(weights) / len(weights)

    cap = float(weight_cap)
    n = len(weights)
    if cap * n < 1.0 - 1e-12:
        raise ValueError("weight_cap is too small to allow weights to sum to 1.")

    weights = weights / weights.sum() if weights.sum() > 0.0 else np.ones(n) / n
    for _ in range(100):
        over = weights > cap
        if not np.any(over):
            break
        weights[over] = cap
        remaining = 1.0 - weights[over].sum()
        under = ~over
        if not np.any(under):
            break
        under_sum = weights[under].sum()
        weights[under] = remaining / under.sum() if under_sum <= 0.0 else weights[under] * remaining / under_sum

    weights = np.minimum(weights, cap)
    return weights / weights.sum()


def diffusion_inverse_vol_tilt_weights(cov, tilt_strength=0.50, weight_cap=None):
    """
    Blend equal weight with inverse generated volatility weights.

    tilt_strength=0 gives equal weight; tilt_strength=1 gives pure inverse-vol.
    """
    cov = np.asarray(cov, dtype=np.float64)
    n = cov.shape[0]
    equal_weight = np.ones(n) / n
    vol = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    inv_vol = 1.0 / vol
    inv_vol = inv_vol / inv_vol.sum()
    weights = (1.0 - tilt_strength) * equal_weight + tilt_strength * inv_vol
    return _apply_weight_cap_and_normalize(weights, weight_cap=weight_cap)


def run_portfolio_variants_from_scenarios(
    backtest_results,
    monthly_returns_df,
    risk_aversion=20.0,
    cov_shrinkage=0.50,
    historical_lookback=120,
    mean_shrinkage=0.75,
    weight_cap=0.10,
    risk_tilt_strength=0.50,
):
    """
    Reuse generated scenarios to test robust portfolio variants.

    Variants:
      - equal_weight
      - generated_markowitz_capped
      - generated_markowitz_mean_shrunk
      - equal_weight_diffusion_risk_tilt

    mean_shrinkage is the weight on historical mean in the shrunk mean:
        mu = (1 - mean_shrinkage) * generated_mu + mean_shrinkage * historical_mu
    """
    from .evaluation import historical_mean_covariance_for_dates

    scenarios = np.asarray(backtest_results["scenarios_raw"], dtype=np.float64)
    actual = np.asarray(backtest_results["actual_next_returns"], dtype=np.float64)
    dates = pd.DatetimeIndex(backtest_results["dates"])
    T, _, N = scenarios.shape

    hist_mean, _, _ = historical_mean_covariance_for_dates(
        monthly_returns_df=monthly_returns_df,
        dates=dates,
        lookback=historical_lookback,
        cov_shrinkage=0.0,
    )

    weights_by_strategy = {
        "equal_weight": np.zeros((T, N)),
        "generated_markowitz_capped": np.zeros((T, N)),
        "generated_markowitz_mean_shrunk": np.zeros((T, N)),
        "equal_weight_diffusion_risk_tilt": np.zeros((T, N)),
    }
    returns_by_strategy = {name: np.zeros(T) for name in weights_by_strategy}

    equal_weight = np.ones(N) / N

    for t in range(T):
        mu_gen, cov_gen = _scenario_mean_cov(
            scenarios[t],
            cov_shrinkage=cov_shrinkage,
            ridge=1e-6,
        )
        mu_shrunk = (1.0 - mean_shrinkage) * mu_gen + mean_shrinkage * hist_mean[t]

        strategy_weights = {
            "equal_weight": equal_weight,
            "generated_markowitz_capped": solve_markowitz_long_only(
                mu=mu_gen,
                cov=cov_gen,
                risk_aversion=risk_aversion,
                cov_jitter=1e-6,
                weight_cap=weight_cap,
            ),
            "generated_markowitz_mean_shrunk": solve_markowitz_long_only(
                mu=mu_shrunk,
                cov=cov_gen,
                risk_aversion=risk_aversion,
                cov_jitter=1e-6,
                weight_cap=weight_cap,
            ),
            "equal_weight_diffusion_risk_tilt": diffusion_inverse_vol_tilt_weights(
                cov=cov_gen,
                tilt_strength=risk_tilt_strength,
                weight_cap=weight_cap,
            ),
        }

        for name, weights in strategy_weights.items():
            weights_by_strategy[name][t] = weights
            returns_by_strategy[name][t] = float(weights @ actual[t])

    rows = []
    for name, returns in returns_by_strategy.items():
        stats = performance_stats(returns)
        stats["Strategy"] = name
        stats["Average Turnover"] = compute_rebalance_turnover(
            weights_by_strategy[name],
            actual,
        ).mean()
        stats["Average Max Weight"] = weights_by_strategy[name].max(axis=1).mean()
        stats["Average Effective N"] = np.mean(
            1.0 / np.sum(weights_by_strategy[name] ** 2, axis=1)
        )
        rows.append(stats)

    stats_df = pd.DataFrame(rows).set_index("Strategy")
    returns_df = pd.DataFrame(returns_by_strategy, index=dates)

    return {
        "stats": stats_df,
        "returns": returns_df,
        "weights": weights_by_strategy,
    }

