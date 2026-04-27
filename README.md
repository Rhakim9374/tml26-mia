# TML26 — Membership Inference Attack (Task 1)

Code for our Trustworthy Machine Learning 2026 Assignment 1 submission.

## Reproducing the leaderboard result

_(Filled in once we have a final submission. Placeholder steps:)_

1. Connect to the Saarland HPC cluster and clone this repo.
   ```bash
   ssh atml_teamXXXXX@conduit2.hpc.uni-saarland.de
   git clone https://github.com/rhakim9374/tml26-mia.git
   cd tml26-mia
   ```
2. Download the assignment data into `data/`:
   ```bash
   mkdir -p data && cd data
   wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/pub.pt"
   wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/priv.pt"
   wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/model.pt"
   wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/task_template.py"
   cd ..
   ```
3. Set your API key:
   ```bash
   cp secrets.env.example secrets.env
   # edit secrets.env and paste your TML_API_KEY
   ```
4. _(TBD: the exact `condor_submit` invocations for the winning run.)_
