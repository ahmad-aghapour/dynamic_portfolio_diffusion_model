# ARMA notebook fix

This project now includes `notebooks/arma_original_exact_reproduction_v2.ipynb`.

The notebook uses the original successful ARMA settings:

- `T=8000`, `num_assets=5`, `sigma=0.01`, `corr=0.30`, no drift
- `context_length=24`, `prediction_length=1`
- LSTM hidden `32`, layers `2`, dropout `0.1`
- U-Net base channels `32`, depth `1`
- `diffusion_steps=1000`
- `batch_size=256`, `epochs=300`
- diagnostic generation `horizon=100`, `num_samples=2000`

It also has assertion cells that fail if these settings are changed accidentally.
