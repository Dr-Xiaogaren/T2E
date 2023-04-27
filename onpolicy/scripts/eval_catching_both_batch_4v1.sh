#!/bin/sh
env="MPE"
scenario="simple_catching_expert_both"  # simple_speaker_listener # simple_reference
num_landmarks=0
num_agents=5
num_adversaries=4
algo="rippo"
exp="test_for_EnvV4_ExpPrey_NoPreyVolo_full_1V1.4_1v4"
seed=1
maps_path_ls=('/home/zh/Documents/workspace/scene/val/easy' '/home/zh/Documents/workspace/scene/val/middle' '/home/zh/Documents/workspace/scene/val/hard')
model_dir="/home/zh/Documents/workspace/on-policy/onpolicy/scripts/results/MPE/simple_catching_expert_both/rippo/EnvV4_ExpPrey_NoPreyVolo_full_1V1.4_1v4/wandb/IPPO_EnvV4_ExpPrey_NoPreyVolo_full_1V1.4_1v4/offline-run-20230409_120945-2n0j22rh/files"
load_model_ep=12400
num_test_episode=500
good_agent_speed=1.4
step_mode="expert_prey" # assert mode == expert_adversary or  expert_both or  expert_prey or  none

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"


for maps_path in ${maps_path_ls[@]};
do
    echo "seed is ${seed}:"
    echo "step_mode is ${step_mode}:"
    echo "maps_path is ${maps_path}"
    CUDA_VISIBLE_DEVICES=1 python eval/eval_catching_both_batch.py  --render_episodes 10  --use_wandb False --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 1 --episode_length 500 --num_env_steps 6000000 --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --model_dir ${model_dir} --load_model_ep ${load_model_ep} --num_test_episode ${num_test_episode} --step_mode ${step_mode} --maps_path ${maps_path} --good_agent_speed ${good_agent_speed} --num_adversaries ${num_adversaries}
done