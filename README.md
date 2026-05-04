# TML26 — Membership Inference Attack (Task 1)

Reproduce our leaderboard submission (priv TPR @ 5% FPR = **0.069691**, 1st place).

The attack is a rank-average ensemble of two online LiRA attacks on two
shadow-model families with different training recipes.

## Prerequisites

- Saarland HPC cluster (or any Linux + NVIDIA GPU + Docker setup).
- Docker image: `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel`.
- Assignment data: `pub.pt`, `priv.pt`, `model.pt` placed in `data/`.
- Submission API key in `secrets.env` (see `secrets.env.example`).

## Steps

All `condor_submit` lines run inside `~/tml26-mia` on the cluster.

### 1. Set up

```bash
ssh atml_teamXXXXX@conduit.hpc.uni-saarland.de
git clone https://github.com/rhakim9374/tml26-mia.git
cd tml26-mia
mkdir -p data && cd data
wget https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/pub.pt
wget https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/priv.pt
wget https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/model.pt
cd ..
cp secrets.env.example secrets.env   # then paste your TML_API_KEY into the file
```

### 2. Train the two shadow families (~3 hours each, in parallel)

Both families train ResNet-18s on a random 50% subset of the combined
pub+priv pool (28k samples). Including priv in the pool is what makes
online LiRA work on the priv leaderboard.

**Baseline family** (no label smoothing, 100 epochs, 512 shadows):
```bash
for k in $(seq 0 15); do
  condor_submit mia.sub \
    -append "script=scripts/train_shadow.py" \
    -append "args=$k --count 32 --ckpt_prefix shadow" \
    -append "tag=shadow_baseline" -queue 1
done
```

**Recipe-matched family** (label smoothing 0.06, 60 epochs, 1024 shadows):
```bash
for k in $(seq 0 31); do
  condor_submit mia.sub \
    -append "script=scripts/train_shadow.py" \
    -append "args=$k --count 32 --label_smoothing 0.06 --epochs 60 --ckpt_prefix lsv1" \
    -append "tag=shadow_lsv1" -queue 1
done
```

Wait until both finish (each shadow takes a few minutes; 16 + 32 GPU jobs run
in parallel so wall-clock is about 2.5 hours):
```bash
ls checkpoints/shadow_*.pt | grep -v in_idx | wc -l    # → 512
ls checkpoints/lsv1_*.pt   | grep -v in_idx | wc -l    # → 1024
```

### 3. Extract LiRA features for each family (~30 min each)

```bash
condor_submit mia.sub \
  -append "script=scripts/score_online_lira.py" \
  -append "args=--ckpt_prefix shadow" \
  -append "tag=score_baseline" -queue 1

condor_submit mia.sub \
  -append "script=scripts/score_online_lira.py" \
  -append "args=--ckpt_prefix lsv1" \
  -append "tag=score_lsv1" -queue 1
```

Each writes `checkpoints/logit_features_<prefix>/log_lr.npy` plus the
underlying φ and μ/σ arrays. The .out file prints the per-family pub TPR
(expect baseline ≈ 0.0697 and lsv1 ≈ 0.0664).

### 4. Run the rank-average ensemble (~1 minute)

```bash
condor_submit mia.sub \
  -append "script=scripts/score_logit_ensemble.py" \
  -append "tag=ensemble" -queue 1
```

This loads both feature sets, sweeps α around 0.30, picks the rank-mix
with highest pub TPR, and writes `submissions/submission.csv`. Expected
pub TPR ≈ 0.0737.

### 5. Submit

```bash
python -m src.submit --tag final_ensemble
```

Expected leaderboard priv TPR ≈ 0.0697.

## Repository layout

```
src/
  data.py     CombinedPool dataset (pub + priv concatenated) + loaders
  model.py    ResNet-18 (CIFAR-style conv1, 9 classes)
  train.py    train_shadow(): SGD lr=0.1 cosine, optional label smoothing
  eval.py     tpr_at_fpr() metric
  submit.py   CSV validator + leaderboard upload
scripts/
  train_shadow.py          Trains one job's worth of shadow models
  score_online_lira.py     Extracts LiRA log-LR features for a shadow family
  score_logit_ensemble.py  Combines two families' features → submission.csv
mia.sub       HTCondor submit template (shared across all stages)
```
