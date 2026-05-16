# Paper checkpoints — WTA-ICL

Original trained Winner-Takes-All (WTA) ICL checkpoints behind the paper
results, preserved here for reproducibility and post-training analysis.

## Contents

`WTA_params_nodes_8+12/<n_nodes>_<rho_all>_<seed>/`, each containing:

| File | What it is |
|------|------------|
| `model.pt` | trained weights (`state_dict`): `W`, `log_K`, `log_beta`, `B`, `W_mask` |
| `results.pkl` | dict: `results`, `history`, `params`, `execution_time` |
| `job_*.sh` | original SLURM script (UChicago Midway, `caslake` partition) |
| `training_batch.out` / `.err` | training logs |

## Runs

| Dir | n_nodes | rho_all | seed | Novel-class ICL | In-dist |
|-----|---------|---------|------|-----------------|---------|
| `8_1.0_30`  | 8  | 1.0 | 30 | 97.8%  | 99.2%  |
| `12_1.0_20` | 12 | 1.0 | 20 | 99.6%  | 100.0% |

Model config (in every `results.pkl['params']`):
`K=128, L=128, D=4, N=4, B=1, epsilon=0.001, R0=2.0, softplus activations,
method=soft, temperature=0.1, epochs=1000, lr=0.0025`.

## Post-training analysis

The `W` matrix inside `model.pt` is the learned chemical-reaction-network
topology — the object of interest for topology analysis. To load a checkpoint:

```python
import pickle, sys; sys.path.insert(0, 'ICL')
from models.wta_icl import load_model

run = 'ICL/paper_checkpoints/WTA_params_nodes_8+12/8_1.0_30/'
params = pickle.load(open(run + 'results.pkl', 'rb'))['params']
model = load_model(params, run)        # eval mode, weights loaded
W = model.W.detach()                   # (n_nodes, (N+1)*D) reaction weights
```

To re-verify a checkpoint reproduces its reported metrics:

```bash
python ICL/verify_checkpoints.py ICL/paper_checkpoints/WTA_params_nodes_8+12/8_1.0_30
```
