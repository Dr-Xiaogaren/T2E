# No Where to Go: Benchmarking Multi-robot Collaboration in Target Trapping Environment

Collaboration is one of the most important factors in multi-robot systems. Considering certain real-world applications and to further promote its development, we propose a new benchmark to evaluate multi-robot collaboration in Target Trapping Environment (T2E). Furthermore, we present and evaluate multiple learning-based baselines in T2E, and provide insights into regimes of multi-robot collaboration. Here is the implementation. 

## 1. Usage

We reproduced algorithms of MAGGPD, MAAC, MAPPO, and IPPO. The `./envs` subfolder contains environment implementations based on [MPE](https://github.com/openai/multiagent-particle-envs.). And the `./offpolicy` and `./onpolicy` subfolders contain the corresponding algorithms implementation.

## 2. Installation & Environment
1. Create an environment using conda
```bash
conda create -n t2e python=3.6
conda activate t2e
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```
2. Install the python package
```bash
git clone https://github.com/Dr-Xiaogaren/T2E.git
cd T2E
pip install -e .
pip install -r requirements.txt
```
Note that there may be redundant packages in `requirements.txt`.

## 3.Train
### 3.1 MAPPO
```
cd onpolicy/scripts
chmod +x ./train_catching_both_mappo.sh
./train_catching_both_mappo.sh
```
### 3.2 IPPO
```
cd onpolicy/scripts
chmod +x ./train_catching_both_ippo.sh
./train_catching_both_ippo.sh
```
### 3.3 MADDPG

```
cd offpolicy/scripts
chmod +x ./train_catching_both_maddpg.sh
./train_catching_both_maddpg.sh
```
### 3.4 MAAC

```
cd offpolicy/scripts
chmod +x ./train_catching_both_maac.sh
./train_catching_both_maac.sh
```
### 3.5 Some Notes for Train
* When setting the number of robots in each group, the total number of robots needs to be adjusted at the same time.
* To set the speed of the robot, one needs to modify it directly in the `mpe/scenarios/simple_catching_expert_both.py` file.
* Each group comes with a heuristic that needs to be set with the --step_mode parameter. The optional range is (***expert_adversary***, ***expert_both***, ***expert_prey***, ***none***). ***expert_adversary*** and ***expert_prey*** mean that only the captor or target use the heuristic respectively. ***expert_both*** and ***none*** mean that both use or neither use heuristics.

## 4. Evaluate
In `eval_catching_both.sh`, set the `--model_dir` to the data directory and `--load_model_ep` to ep number in the name of pretrained models. If visualization is not required, remember to remove `-- save_gifs`. Then just run:
```
cd onpolicy/scripts
chmod +x eval_catching_both.sh
eval_catching_both.sh
```
The visualization function has been modified to `scripts/render_catching_both.sh`

## 5. Acknowledgment
Our code frameworks are mainly implemented based on [MPE](https://github.com/openai/multiagent-particle-envs.), [MAPPO](https://github.com/marlbenchmark/on-policy.), [MAAC](https://github.com/shariqiqbal2810/MAAC.), and [MADDPG](https://github.com/shariqiqbal2810/maddpg-pytorch.). We thank the respective authors for open-sourcing their code.