# Portfolio Diffusion Model

This repository organizes the notebook code for a conditional recurrent diffusion model that generates monthly industry-return scenarios and evaluates them with a long-only Markowitz portfolio rule.

The notebook is now only the experiment driver. Reusable code lives in `portfolio_diffusion/`.

## Repository layout

```text
portfolio_diffusion/
  config.py        # DDPMConfig
  data.py          # FF49 loader, rolling-window dataset, train/valid split helpers
  models.py        # RNN conditioner + 1D U-Net DDPM
  training.py      # training loop, EMA, schedulers, early stopping
  generation.py    # scenario generation helpers
  portfolio.py     # Markowitz, backtest, stats, RL scenario saving
  plotting.py      # wealth/weights/turnover plots
  simulation.py    # synthetic ARMA simulator
  evaluation.py    # mean/std/ACF1/correlation diagnostics
notebooks/
  ff49_diffusion_pipeline.ipynb
  arma_synthetic_experiment.ipynb
outputs/           # created artifacts; ignored by git
```

## Data

Download `49_Industry_Portfolios.csv` from Kenneth French's data library and place it in:

```text
data/49_Industry_Portfolios.csv
```

The loader uses the `Average Value Weighted Returns -- Monthly` section, starts at 1929-01-01, converts percent returns to decimal returns, and drops incomplete industry columns rather than dropping rows.

## Install

```bash
pip install -r requirements.txt
```

`cvxpy` is optional but recommended. If it is unavailable, the Markowitz solver falls back to `scipy.optimize.minimize`.

## Run

Open and run the real-data FF49 notebook:

```text
notebooks/ff49_diffusion_pipeline.ipynb
```

Or run the synthetic ARMA experiment notebook:

```text
notebooks/arma_synthetic_experiment.ipynb
```

Both notebooks train the diffusion model, run rolling generated-scenario Markowitz backtests, plot wealth trajectories, compute performance statistics, and save generated scenario tensors plus RNN conditioning features for RL. The ARMA notebook also computes generated-path diagnostics: Mean MAE, Std MAE, ACF1 MAE, and correlation MAE.

## RL scenario output

The main RL file is:

```text
outputs/ff49_diffusion_scenarios_for_rl.npz
```

It contains:

```python
scenarios_raw                 # [T_test, num_scenarios, num_assets]
scenarios_standardized        # [T_test, num_scenarios, num_assets]
dates                         # [T_test]
asset_names                   # [num_assets]
actual_next_returns           # [T_test, num_assets]
history_windows_raw           # [T_test, context_length, num_assets]
history_windows_standardized  # [T_test, context_length, num_assets]
rnn_features                  # [T_test, num_scenarios, rnn_hidden_dim]
gm_weights                    # [T_test, num_assets]
scenario_means                # [T_test, num_assets]
scenario_covs                 # [T_test, num_assets, num_assets]
```

For each time step `t` in RL:

```python
state_t = history_windows_raw[t]
scenario_set_t = scenarios_raw[t]
rnn_feature_set_t = rnn_features[t]
realized_return_t = actual_next_returns[t]
```

## Notes

The current training uses repo-style sequence-to-sequence conditioning: each 12-month window contributes a target for every RNN hidden state, producing targets with shape `[batch, context_length, prediction_length, num_assets]`. The diffusion loss flattens `batch * context_length` during training.

## Relation to the reference repo

The reference implementation uses the diffusion model's RNN output as the scenario feature for TD3. In its real-data TD3 code, the observation is built by concatenating `feature[time]`, wealth, and the multiplier. Its scenario generator fills `feature` from the diffusion model's sampled RNN features. This project now saves the same kind of scaled RNN conditioning feature as `rnn_features`.
