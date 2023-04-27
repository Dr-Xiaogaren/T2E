import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check

class R_MAPPO_ForBC():

    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.data_chunk_length = args.imitation_data_chunk_length
        self.max_grad_norm = args.max_grad_norm 

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self.recurrent_N = args.recurrent_N
        self.hidden_size = args.hidden_size


    def policy_update(self, sample):

        obs_batch, actions_batch, rnn_states_actor, masks = sample
        actions_batch = check(actions_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        action_probs = self.policy.get_actions_probs(obs_batch,
                                                    rnn_states_actor,
                                                    masks)


        policy_loss = nn.CrossEntropyLoss()(action_probs, actions_batch).sum()

        self.policy.actor_optimizer.zero_grad()

        policy_loss.backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        return  policy_loss, actor_grad_norm

    def train(self, dataset):

        dataset.initial_dataloader()
        train_info = {}
        train_info['policy_loss'] = 0
        train_info['actor_grad_norm'] = 0
        for i in range(len(dataset.train_loader)):

            expert_obs, expert_actions, mask = dataset.get_next_batch('train')

            rnn_states_actor = np.zeros((int(len(expert_actions)/self.data_chunk_length),self.recurrent_N,self.hidden_size))
            assert len(rnn_states_actor)*self.data_chunk_length == len(expert_actions)

            actions_onehot = np.squeeze(np.eye(self.policy.act_space.n)[expert_actions], 1)

            mask = 1 - mask
            # sample 
            sample=(expert_obs, actions_onehot, rnn_states_actor, mask)
            # update network
            policy_loss, actor_grad_norm = self.policy_update(sample)

            train_info['policy_loss'] += policy_loss.item()
            train_info['actor_grad_norm'] += actor_grad_norm.item()
        num_updates = len(dataset.train_loader)
        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()

    def prep_rollout(self):
        self.policy.actor.eval()

    def val(self,dataset):
        dataset.initial_dataloader()
        val_info = {}
        val_info['val_loss'] = 0
        for i in range(len(dataset.train_loader)):

            expert_obs, expert_actions, mask = dataset.get_next_batch('val')
            rnn_states_actor = np.zeros((int(len(expert_actions)/self.data_chunk_length),self.recurrent_N,self.hidden_size))
            assert len(rnn_states_actor)*self.data_chunk_length == len(expert_actions)

            actions_onehot = np.squeeze(np.eye(self.policy.act_space.n)[expert_actions], 1)

            mask = 1 - mask
            # sample 
            sample=(expert_obs, actions_onehot, rnn_states_actor, mask)
            obs_batch, actions_batch, rnn_states_actor, masks = sample
            actions_batch = check(actions_batch).to(**self.tpdv)

            # Reshape to do in a single forward pass for all steps
            action_probs = self.policy.get_actions_probs(obs_batch,
                                                        rnn_states_actor,
                                                        masks)           
            # update network
            policy_loss = nn.CrossEntropyLoss()(action_probs, actions_batch).sum()

            val_info['val_loss'] += policy_loss.item()
        num_updates = len(dataset.train_loader)
        for k in val_info.keys():
            val_info[k] /= num_updates
    
        return val_info
