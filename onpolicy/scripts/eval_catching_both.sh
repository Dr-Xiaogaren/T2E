#!/bin/sh
env="MPE"
scenario="simple_catching_expert_both"  # simple_speaker_listener # simple_reference
num_landmarks=0
num_agents=4
algo="rmappo"
exp="debug"
seed_max=1
model_dir="/workspace/on-policy/onpolicy/scripts/results/MPE/simple_catching_expert_both/rmappo/EnvV3_TargetSpeed1.0_FullMap_Optimize_SparseRW_NoPreyVolo_ExpPrey/wandb/run-20221113_162931-mewcuvw8/files"
load_model_ep=9300
num_test_episode=500
step_mode="expert_prey" # assert mode == expert_adversary or  expert_both or  expert_prey or  none

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"


for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    echo "step_mode is ${step_mode}:"
    CUDA_VISIBLE_DEVICES=0 python eval/eval_catching_both.py  --render_episodes 10  --use_wandb False --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 1 --episode_length 200 --num_env_steps 6000000 --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --model_dir ${model_dir} --load_model_ep ${load_model_ep} --num_test_episode ${num_test_episode} --step_mode ${step_mode}
done