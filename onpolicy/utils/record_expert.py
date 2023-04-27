from typing import Dict
import numpy as np
import gym
from gym import spaces
import sys
import copy
from tqdm import tqdm
import os

def generate_expert_traj(args, env, save_path=None, n_episodes=5):
    """
    Train expert controller (if needed) and record expert trajectories.
    .. note::
        only Box and Discrete spaces are supported for now.
    :param env: (gym.Env) The environment, if not defined then it tries to use the model
        environment.
    :param save_path: (str) Path without the extension where the expert dataset will be saved
        (ex: 'expert_cartpole' -> creates 'expert_cartpole.npz').
        If not specified, it will not save, and just return the generated expert trajectories.
        This parameter must be specified for image-based environments.
    :param n_episodes: (int) Number of trajectories (episodes) to record
    :return: (dict) the generated expert trajectories.
    """

    assert env is not None, "You must set the env in the model or pass it to the function."
    num_agent = args.num_agents
    num_bads = args.num_adversaries
    num_goods = args.num_good_agents
    file_name_bads = "chaser_trajectory_1.0_check.npz"
    file_name_goods = "evader_trajectory_1.0_check.npz"

    # Sanity check
    actions_gp1 = []
    actions_gp2 = []
    observations_gp1 = dict()
    observations_gp2 = dict()
    rewards_gp1 = []
    rewards_gp2 = []
    episode_returns_gp1 = np.zeros((n_episodes,))
    episode_returns_gp2 = np.zeros((n_episodes,))
    episode_starts = []

    ep_idx = 0
    obs = env.reset()
    obs_dict = dict()
    for key in args.observation_dict:
            obs_dict[key] = np.array([obs[n][key] for n in range(num_agent)])
            observations_gp1[key] = []
            observations_gp2[key] = []
    
    episode_starts.append(True)
    reward_sum_gp1 = 0.0
    reward_sum_gp2 = 0.0
    idx = 0
    pbar = tqdm(total=n_episodes)
    while ep_idx < n_episodes:

        action = env.controllers()

        for key in args.observation_dict:
            observations_gp1[key].append(obs_dict[key][0:num_bads,...])
            observations_gp2[key].append(obs_dict[key][num_bads:,...])
        
        action_array = np.where(np.array(action)==1)[-1]
        actions_gp1.append(action_array[0:num_bads])
        actions_gp2.append(action_array[num_bads:])

        obs, reward, done, _ = env.step(action, mode="none")

        for key in args.observation_dict:
            obs_dict[key] = np.array([obs[n][key] for n in range(num_agent)])
        
        reward_array = np.array(reward)
        rewards_gp1.append(reward_array[0:num_bads])
        rewards_gp2.append(reward_array[num_bads:])

        episode_starts.append(sum(done) == num_agent)

        reward_sum_gp1 += reward_array[0:num_bads]
        reward_sum_gp2 += reward_array[num_bads:]
        idx += 1

        if sum(done) == num_agent:
            obs = env.reset()
            for key in args.observation_dict:
                obs_dict[key] = np.array([obs[n][key] for n in range(num_agent)])
            episode_returns_gp1[ep_idx] = np.mean(reward_sum_gp1)
            episode_returns_gp2[ep_idx] = np.mean(reward_sum_gp2)
            reward_sum_gp1 = 0.0
            reward_sum_gp2 = 0.0
            ep_idx += 1
            pbar.update(1)

    pbar.close()

    for key in args.observation_dict:
            observations_gp1[key] = np.array(observations_gp1[key])
            observations_gp2[key] = np.array(observations_gp2[key])

    actions_gp1 = np.array(actions_gp1)
    actions_gp2 = np.array(actions_gp2)


    rewards_gp1 = np.array(rewards_gp1)
    rewards_gp2 = np.array(rewards_gp2)
    episode_starts = np.array(episode_starts[:-1])


    numpy_dict_gp1 = {
        'actions': actions_gp1,
        'rewards': rewards_gp1,
        'episode_returns': episode_returns_gp1,
        'episode_starts': episode_starts
    }  # type: Dict[str, np.ndarray]

    numpy_dict_gp2 = {
        'actions': actions_gp2,
        'rewards': rewards_gp2,
        'episode_returns': episode_returns_gp2,
        'episode_starts': episode_starts
    }  # type: Dict[str, np.ndarray]

    numpy_dict_gp1.update(observations_gp1)
    numpy_dict_gp2.update(observations_gp2)

    for key, val in numpy_dict_gp1.items():
        print("group 1:",key, val.shape)
    
    for key, val in numpy_dict_gp2.items():
        print("group 2:",key, val.shape)

    if save_path is not None:
        path_gp1 = os.path.join(save_path, file_name_bads)
        path_gp2 = os.path.join(save_path, file_name_goods)
        np.savez(path_gp1, **numpy_dict_gp1)
        np.savez(path_gp2, **numpy_dict_gp2)

    env.close()

    return numpy_dict_gp1, numpy_dict_gp2


def main():
    import time
    from envs.mpe.environment import MultiAgentEnv, CatchingEnv, CatchingEnvExpert
    from envs.mpe.scenarios import load
    from onpolicy.config import get_config
    parser = get_config()
    args = parser.parse_known_args()[0]
    args.env_name = "MPE"
    args.scenario_name = "simple_catching_expert_both"
    args.num_agents = 4
    scenario = load(args.scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(args)
    # create multiagent environment
    env = CatchingEnvExpert(world, reset_callback=scenario.reset_world, reward_callback=scenario.reward, 
                        observation_callback= scenario.observation, info_callback=  scenario.info, 
                        done_callback=scenario.if_done, post_step_callback=scenario.post_step)
    save_path = "/workspace/tmp/data"
    generate_expert_traj(args, env, save_path, n_episodes=1000)

if __name__=="__main__":
   main()
