import torch
import numpy as np
from collections import defaultdict

from offpolicy.utils.util import check,get_shape_from_obs_space, get_shape_from_act_space

def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])

def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])

def _cast_till_agent(x):
    return x.transpose(1, 0, 2, 3).reshape(-1, *x.shape[2:])

def _flatten_till_agent(T, N, x):
    return x.reshape(T * N, *x.shape[3:])

class SharedReplayBuffer(object):
    def __init__(self, args, num_agents, obs_space, act_space):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self._use_popart = args.use_popart

        max_buffer_size = args.buffer_size
        self.max_buffer_size = max_buffer_size

        self._mixed_obs = False  # for mixed observation   

        obs_shape = get_shape_from_obs_space(obs_space)

        # for mixed observation
        if 'Dict' in obs_shape.__class__.__name__:
            self._mixed_obs = True
            
            self.obs = {}

            for key in obs_shape:
                self.obs[key] = np.zeros((max_buffer_size, self.n_rollout_threads, num_agents, *obs_shape[key].shape), dtype=np.float32)
       
        else: 
            # deal with special attn format   
            if type(obs_shape[-1]) == list:
                obs_shape = obs_shape[:1]

            self.obs = np.zeros((max_buffer_size, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)

        self.rnn_states = np.zeros((max_buffer_size, self.n_rollout_threads, num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros(
            (max_buffer_size, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)

        self.rewards = np.zeros(
            (max_buffer_size, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        self.masks = np.ones((max_buffer_size, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0  # current index to write to (ovewrite oldest data)
        self.filled_i = 0  # index of first empty location in buffer (last index when full)
    
    def buffer_size(self):
        return self.filled_i+1

    def insert(self, obs, rnn_states, rnn_states_critic, actions,
               rewards, masks):

        next_step = (self.step + 1) % (self.max_buffer_size)
        if self._mixed_obs:
            for key in self.obs.keys():
                self.obs[key][next_step] = obs[key].copy()
        else:
            self.obs[next_step] = obs.copy()

        self.rnn_states[next_step] = rnn_states.copy()
        self.rnn_states_critic[next_step] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[next_step] = masks.copy()

        self.step = next_step
        self.filled_i = self.filled_i + 1 if self.filled_i + 1 < self.max_buffer_size else self.max_buffer_size-1

        return self.step

    def recurrent_generator(self,num_batch, batch_size, data_chunk_length):
        _ , n_rollout_threads, num_agents = self.rewards.shape[0:3]
        n_records = n_rollout_threads * (self.filled_i-1)
        data_chunks = n_records // data_chunk_length - 2

        rand = torch.randperm(data_chunks).numpy() # make sure no overflow


        sampler_all = [rand[0:batch_size] for _ in range(num_batch)]

        if self._mixed_obs:
            obs = {}

            for key in self.obs.keys():
                if len(self.obs[key].shape) == 6:
                    obs[key] = self.obs[key][:self.filled_i].transpose(1, 0, 2, 3, 4, 5).reshape(-1, *self.obs[key].shape[2:])
                elif len(self.obs[key].shape) == 5:
                    obs[key] = self.obs[key][:self.filled_i].transpose(1, 0, 2, 3, 4).reshape(-1, *self.obs[key].shape[2:])
                else:
                    obs[key] = _cast_till_agent(self.obs[key][:self.filled_i])
        else:
            if len(self.obs.shape) > 4:
                obs = self.obs[:self.filled_i].transpose(1, 0, 2, 3, 4, 5).reshape(-1, *self.obs.shape[2:])
            else:
                obs = _cast_till_agent(self.obs[:self.filled_i])

        actions = _cast_till_agent(self.actions[:self.filled_i])
        rewards = _cast_till_agent(self.rewards[:self.filled_i])
        masks = _cast_till_agent(self.masks[:self.filled_i])    
        # rnn_states = _cast(self.rnn_states[:-1])
        # rnn_states_critic = _cast(self.rnn_states_critic[:-1])
        rnn_states = self.rnn_states[:self.filled_i].transpose(1, 0, 2, 3, 4).reshape(-1, *self.rnn_states.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:self.filled_i].transpose(1, 0, 2, 3, 4).reshape(-1, *self.rnn_states_critic.shape[2:])

        for sampler in sampler_all:
            if self._mixed_obs:
                obs_batch = defaultdict(list)
                next_obs_batch = defaultdict(list)
            else:
                obs_batch = []
                next_obs_batch = []

            rnn_states_batch = []
            next_rnn_states_batch = []

            rnn_states_critic_batch = []
            next_rnn_states_critic_batch = []

            actions_batch = []
            reward_batch = []

            masks_batch = []
            next_masks_batch = []
            
            for index in sampler:
                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                if self._mixed_obs:
                    for key in obs.keys():
                        obs_batch[key].append(obs[key][ind:ind+data_chunk_length])
                        next_obs_batch[key].append(obs[key][ind+1:ind+data_chunk_length+1])
                else:
                    obs_batch.append(obs[ind:ind+data_chunk_length])
                    next_obs_batch.append(obs[ind+1:ind+data_chunk_length+1])

                actions_batch.append(actions[ind:ind+data_chunk_length])

                reward_batch.append(rewards[ind:ind+data_chunk_length])

                masks_batch.append(masks[ind:ind+data_chunk_length])
                next_masks_batch.append(masks[ind+1:ind+data_chunk_length+1])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                next_rnn_states_batch.append(rnn_states[ind+1])

                rnn_states_critic_batch.append(rnn_states_critic[ind])
                next_rnn_states_critic_batch.append(rnn_states_critic[ind+1])
            
            L, N = data_chunk_length, batch_size*num_agents

            # These are all from_numpys of size (L, N, num_agent, Dim) 
            if self._mixed_obs:
                for key in obs_batch.keys():  
                    obs_batch[key] = np.stack(obs_batch[key], axis=1)
                    next_obs_batch[key] = np.stack(next_obs_batch[key], axis=1)
            else:        
                obs_batch = np.stack(obs_batch, axis=1)
                next_obs_batch = np.stack(next_obs_batch, axis=1)
            actions_batch = np.stack(actions_batch, axis=1)

            reward_batch = np.stack(reward_batch, axis=1)

            masks_batch = np.stack(masks_batch, axis=1)
            next_masks_batch = np.stack(next_masks_batch, axis=1)

            # States is just a (N, num_agent, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            next_rnn_states_batch = np.stack(next_rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])

            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])
            next_rnn_states_critic_batch = np.stack(next_rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            
            # Flatten the (L, N, num_agent, ...) from_numpys to (L * N, ...)
            if self._mixed_obs:
                for key in obs_batch.keys(): 
                    if len(obs_batch[key].shape) == 6:
                        obs_batch[key] = obs_batch[key].reshape(L*N, *obs_batch[key].shape[3:])
                        next_obs_batch[key] = next_obs_batch[key].reshape(L*N, *next_obs_batch[key].shape[3:])
                    elif len(obs_batch[key].shape) == 5:
                        obs_batch[key] = obs_batch[key].reshape(L*N, *obs_batch[key].shape[3:])
                        next_obs_batch[key] = next_obs_batch[key].reshape(L*N, *next_obs_batch[key].shape[3:])
                    else:
                        obs_batch[key] = _flatten_till_agent(L,N, obs_batch[key])
                        next_obs_batch[key] = _flatten_till_agent(L,N, next_obs_batch[key])
            else:
                obs_batch = _flatten_till_agent(L, N, obs_batch)
                next_obs_batch = _flatten_till_agent(L, N, next_obs_batch)
            actions_batch = _flatten_till_agent(L, N, actions_batch)
            reward_batch = _flatten_till_agent(L, N, reward_batch)

            masks_batch = _flatten_till_agent(L, N, masks_batch)
            next_masks_batch = _flatten_till_agent(L, N, next_masks_batch)

            yield obs_batch, rnn_states_batch, rnn_states_critic_batch, masks_batch, \
                next_obs_batch, next_rnn_states_batch, next_rnn_states_critic_batch,next_masks_batch, actions_batch, reward_batch
  
    def get_average_rewards(self):
        if self.filled_i == self.max_buffer_size-1:
            inds = np.arange(self.step - self.episode_length, self.step)
        else:
            inds = np.arange(max(0, self.step - self.episode_length), self.step)
        
        ep_reward = self.rewards[inds,...].mean()*self.episode_length
        return ep_reward