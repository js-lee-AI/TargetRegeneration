# Data

`vision/` holds the prompt-level acceptance counts behind every vision number the scripts in
`experiments/` rebuild: 241 evaluations, each on the 99 scored prompts of three domains, all on
NVIDIA RTX A6000 GPUs with the tree verifier unless a run says otherwise.

## vision/runs.csv

One row for each evaluation. A cell of the paper is the set of runs that differ only in `seed`.

| column | values |
|---|---|
| `run` | identifier used in `acceptance.csv` |
| `corpus` | `allava`, `sharegpt4v`; `none` for the released drafter |
| `responses` | `external`, `full` (greedy regeneration), `truncated`, `window` (loss window), `sampled_t1`, `sampled_t07` (sampled regeneration at T=1 and T=0.7), `mix25`, `mix50`, `mix75` (that share of the external responses regenerated); `none` for the released drafter |
| `objective` | `ce`, `kd`, `gate` (KD + gate), `soft` (KD with the soft survival weight), `greedy_labels` (CE toward the target's greedy token); `none` for the released drafter |
| `setting` | `main`; the controls `random_init`, `lower_lr`, `l2_anchor`, `kd64` (64 distilled tokens); `initialization` for the released drafter |
| `gamma` | slot weight decay: 7, or 2 in the gamma control |
| `epochs` | training length; 0.25 and 0.5 are schedules that stop and anneal early |
| `seed` | anchor seed of training, 0 or 1 |
| `order` | data order: 0 is the order of every cell, 1000 and 2000 are the two further orders |
| `temperature` | target temperature at evaluation, 0 or 1 |
| `decoder` | `tree` (63 draft tokens) or `chain` (one path of seven) |

## vision/acceptance.csv

One row for each run, domain and scored prompt.

| column | meaning |
|---|---|
| `run` | the run in `runs.csv` |
| `domain` | `caption` (COCO), `textvqa`, `docvqa` |
| `prompt` | 1 to 99, the position in the fixed evaluation set; position 0 is the warm-up and is not scored |
| `committed` | tokens committed over the prompt's verification rounds, the accepted draft tokens plus the target's own token in each round |
| `rounds` | verification rounds |

A domain's MAT is the sum of `committed` over its prompts divided by the sum of `rounds`, and a
run's MAT is the geometric mean of its three domain MATs. `pipeline/prepare_vision_benchmarks.py`
rebuilds the evaluation prompts from `pipeline/data/vision_manifest.json`.
