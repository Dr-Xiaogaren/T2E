#!/bin/sh
env="MPE"
scenario="simple_catching_expert_both"  # simple_speaker_listener # simple_reference
num_landmarks=0
num_agents=5
num_adversaries=4
algo="maddpg"
exp="EnvV4_ExpPrey_NoPreyVolo_full_1V1.4_1v4"
good_agent_speed=1.4
seed=1
step_mode="expert_prey"
echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

echo "seed is ${seed}:"

CUDA_VISIBLE_DEVICES=0 python train/train_catching_both.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 1 --n_rollout_threads 16 --episode_length 200 --num_env_steps 30000000  --use_ReLU --gain 1.0 --lr 1e-5 --critic_lr 3e-5 --step_mode ${step_mode} --wandb_name "joe-14807" --user_name "joe-14807" --good_agent_speed ${good_agent_speed}  --num_adversaries ${num_adversaries}
