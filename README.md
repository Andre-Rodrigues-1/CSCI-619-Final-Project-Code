# Learning-to-Optimize Economic Dispatch

This repository is a self-contained computational study of **learning a fast
economic-dispatch proxy** for a network-constrained power system. It pairs
operations-research structure with a learned objective: a full DC economic
dispatch (ED) model supplies reference decisions, while a much smaller
copperplate LP is learned to imitate those generator dispatches.

The project is aimed at the practical question common to optimization and
quantitative research: **when is it useful to replace an expensive structured
decision problem with a faster approximation, and how should that
approximation be evaluated?**

## Why This Is Interesting

Large dispatch models must be solved repeatedly for changing demand scenarios.
The full DC ED problem captures network flows and transmission constraints but
is relatively expensive. A copperplate ED ignores those network constraints and
is much faster, but its dispatch is a poor behavioral approximation of the DC
solution.

Rather than directly regress generator outputs, this project learns
scenario-dependent generator cost coefficients and keeps an optimization model
in the inference loop:

1. Generate perturbed demand scenarios from a PGLib reference case.
2. Solve the full DC ED oracle to obtain a target dispatch.
3. Predict nonnegative generator costs from scenario features.
4. Solve a reduced copperplate LP with those learned costs.
5. Train the cost model with a Fenchel-Young structured loss so that the
   reduced optimizer's decisions approach the DC oracle's decisions.

This design preserves an interpretable optimization layer at inference and
makes the speed--accuracy trade-off explicit.

## Experimental Setting

| Item | Configuration |
| --- | --- |
| Benchmark | PGLib `pglib_opf_case300_ieee.m` |
| System size | 300 buses, 69 generators, 411 branches |
| Scenario design | Pascal-style global demand scaling and nodal log-normal load noise |
| Total scenarios | 5,000 |
| Split | 3,500 train / 500 validation / 1,000 held-out test |
| Hard model | Network-constrained DC economic dispatch |
| Fast model | Generator-level copperplate LP with energy balance, capacity, and reserve proxy |
| Learning target | DC-oracle generator dispatch |
| Training loss | Fenchel-Young loss |
| Model selection | Validation dispatch L1; early stopping with patience 3 |

The split is fixed and seeded. Training shuffles scenario order within an epoch
to make stochastic optimization less sensitive to scenario ordering; scenarios
themselves are independent synthetic dispatch instances rather than a temporal
forecasting sequence.

## Method

For scenario features `x`, the model predicts a positive vector of generator
costs `c_theta(x)`. The reduced optimizer returns

```text
y_hat = argmin_{y in Y_easy(x)} c_theta(x)^T y
```

where `Y_easy(x)` is the feasible set of the copperplate dispatch LP. Given a
DC-oracle dispatch `y*`, the unregularized Fenchel-Young loss used here is

```text
L_FY(c_theta(x), y*) = c_theta(x)^T y* - min_{y in Y_easy(x)} c_theta(x)^T y
```

with the dispatch-related slack terms included consistently in the evaluated
objective. A subgradient with respect to the learned costs is `y* - y_hat`.
The implementation maps features to costs through a linear model followed by
softplus, guaranteeing positive predicted costs.

This follows the learning-through-optimization perspective of Parmentier et
al. and is complementary to neural feasibility-repair approaches such as
end-to-end learning and repair for ED.

## Included Result

The checked-in `final_outputs` directory contains the final-epoch run from the
configuration above. On the 1,000-scenario held-out test set:

| Method | Mean generator L1 to DC oracle (MW) | Change vs. financial baseline | Mean solve time (s/scenario) | Relative to DC oracle |
| --- | ---: | ---: | ---: | ---: |
| Full DC oracle | 0.0 | -- | 0.202 | 1.00x |
| Financial copperplate baseline | 6,053.1 | -- | 0.0549 | 3.68x faster |
| Learned copperplate LtO | 3,891.0 | 35.7% lower L1 | 0.0529 | 3.82x faster |

The learned model materially improves imitation of the DC dispatch while
preserving the reduced model's runtime. The appropriate primary result is
**dispatch agreement and inference time**, not a comparison of raw objective
values: the copperplate model omits transmission constraints and therefore is
not feasible for the full DC network. It is a proxy for screening, warm starts,
or approximate decision support, not a replacement for a network-feasible
production dispatcher.

## Reproduce

### Prerequisites

- Python 3.10 or newer
- `numpy`, `pandas`, and `scipy`
- `gurobipy` with a working Gurobi license for the reported runtime

Run from this directory:

```bash
../../.venv/bin/python run_final_results.py \
  --hours 5000 \
  --train-hours 3500 \
  --validation-hours 500 \
  --epochs 20 \
  --solver gurobi \
  --solver-threads 6 \
  --jobs 6 \
  --output-dir final_outputs/ieee300_copperplate
```

The runner evaluates two predetermined variants:

- `copperplate_final_epoch`: reports the final epoch after the fixed 20-epoch budget.
- `copperplate_early_stop`: selects the model with the best validation
  generator-dispatch L1 and stops after three non-improving epochs.

To validate installation without the full experiment, use a small run and the
open-source solver:

```bash
../../.venv/bin/python run_final_results.py \
  --hours 20 \
  --train-hours 12 \
  --validation-hours 4 \
  --epochs 2 \
  --solver scipy \
  --solver-threads 1 \
  --jobs 1 \
  --output-dir final_outputs/smoke
```

## Outputs

```text
final_outputs/ieee300_copperplate/
  final_summary.csv                 # headline accuracy/runtime comparison
  final_detailed_metrics.csv        # per-method metrics for both variants
  copperplate_final_epoch/
    metrics.csv                     # DC, financial baseline, and learned model
    training_history.csv            # train/validation FY and dispatch-L1 curves
  copperplate_early_stop/
    metrics.csv
    training_history.csv
```

Key columns:

- `mean_generation_l1_to_dc_mw`: mean absolute error between proxy and DC-oracle
  generator dispatch vectors; the primary accuracy metric.
- `mean_runtime_seconds`: mean wall-clock solve time per scenario.
- `mean_regret_to_financial_proxy`: change in the reduced-model objective
  relative to the raw-cost copperplate baseline.
- `mean_cost_gap_to_dc_oracle`: descriptive only; it must not be interpreted as
  savings because the reduced model does not enforce DC transmission feasibility.

## Repository Layout

```text
run_final_results.py       experiment entry point and result writer
pglib_lto/parser.py        MATPOWER/PGLib case parsing
pglib_lto/scenario.py      Pascal-style scenario generator
pglib_lto/solvers.py       DC-oracle and copperplate optimization models
pglib_lto/model.py         cost model, FY loss, training, and early stopping
pglib_opf_case300_ieee.m   benchmark instance
```

## Scope and Next Research Steps

This is a deliberately focused, reproducible experiment rather than a claim of
operational feasibility. A production-grade extension should add full-network
feasibility repair or screening, compare multiple PGLib systems and random
seeds, report confidence intervals, and evaluate warm-start value for the DC
solver. Those extensions would distinguish approximation quality from
deployment readiness.

## References

- A. Parmentier et al., *Learning to Optimize with Fenchel-Young Losses*.
- P. Chen, R. Tanneau, and P. Van Hentenryck, *End-to-End Learning and Repair
  for Economic Dispatch*.
- Power Grid Lib, PGLib-OPF benchmark collection.
