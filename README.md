# TML26 — Membership Inference Attack (Task 1)

How to reproduce our leaderboard result (priv TPR @ 5% FPR = **0.069691**).

The attack is a rank-average ensemble of two online-LiRA attacks, each on a
shadow-model family trained with a different recipe.

## 1. Set up

```bash
git clone https://github.com/rhakim9374/tml26-mia.git
cd tml26-mia
mkdir -p data && cd data
wget https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/pub.pt
wget https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/priv.pt
wget https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/model.pt
cd ..
cp secrets.env.example secrets.env   # paste your TML_API_KEY into secrets.env
```

All commands below run from the repo root inside the cluster's Docker image
(`pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel`); `mia.sub` handles that.

## 2. Train two shadow families

Baseline (no label smoothing, 100 epochs, 512 shadows):
```bash
for k in $(seq 0 15); do
  condor_submit mia.sub \
    -append "script=scripts/train_shadow.py" \
    -append "args=$k --count 32 --ckpt_prefix shadow" \
    -append "tag=shadow_baseline" -queue 1
done
```

Recipe-matched (label smoothing 0.06, 60 epochs, 1024 shadows):
```bash
for k in $(seq 0 31); do
  condor_submit mia.sub \
    -append "script=scripts/train_shadow.py" \
    -append "args=$k --count 32 --label_smoothing 0.06 --epochs 60 --ckpt_prefix lsv1" \
    -append "tag=shadow_lsv1" -queue 1
done
```

Wait until `ls checkpoints/shadow_*.pt | grep -v in_idx | wc -l` is 512 and
`ls checkpoints/lsv1_*.pt | grep -v in_idx | wc -l` is 1024.

## 3. Extract LiRA features for each family

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

## 4. Ensemble and submit

```bash
condor_submit mia.sub \
  -append "script=scripts/score_logit_ensemble.py" \
  -append "tag=ensemble" -queue 1

python -m src.submit --tag final
```

## Repository layout

```
src/
  data.py     CombinedPool dataset (pub + priv) and loaders
  model.py    ResNet-18 (CIFAR-style conv1, 9 classes)
  train.py    train_shadow(): SGD lr=0.1 cosine, optional label smoothing
  eval.py     tpr_at_fpr() metric
  submit.py   CSV validator + leaderboard upload
scripts/
  train_shadow.py          Trains one Condor job's worth of shadow models
  score_online_lira.py     Extracts LiRA log-LR features for a shadow family
  score_logit_ensemble.py  Combines two families' features → submission.csv
mia.sub       HTCondor submit template (shared across all stages)
```

