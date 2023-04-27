    
import time
from tokenize import group
import wandb
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter


def _t2n(x):
    return x.detach().cpu().numpy()

class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        self.num_groups = 2
        self.num_goods = config['num_goods'] # 1
        self.num_bads = config['num_bads'] # 3
        self.obs_dict_keys = self.all_args.observation_dict


        # parameters
        self.num_epoch = self.all_args.imitation_num_epoch
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.episode_length = self.all_args.episode_length
        self.use_linear_lr_decay = self.all_args.imitation_use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.imitation_save_interval
        self.log_interval = self.all_args.imitation_log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)


        from onpolicy.algorithms.imitation_learning.imitation_learning import R_MAPPO_ForBC as TrainAlgo
        from onpolicy.algorithms.imitation_learning.algorithm.rMAPPOPolicy import R_MAPPOPolicy_ForBC as Policy
        from onpolicy.utils.dataset import ExpertDataset


        self.policy = []
        # 0 is bads , is goods
        for group_id in range(self.num_groups):
            # policy network
            po = Policy(self.all_args,
                        self.envs.observation_space[self.num_bads-1+group_id],
                        self.envs.action_space[self.num_bads-1+group_id],
                        device = self.device)
            self.policy.append(po)

        if self.model_dir is not None:
            self.restore(self.all_args.load_model_ep)

        self.trainer = []
        self.dataset = []
        dataset_path = [self.all_args.expert_path_gp0, self.all_args.expert_path_gp1]
        for group_id in range(self.num_groups):
            # algorithm
            tr = TrainAlgo(self.all_args, self.policy[group_id], device = self.device)
            self.trainer.append(tr)
            # dataset
            dataset = ExpertDataset(self.all_args, dataset_path[group_id])
            self.dataset.append(dataset)

    def train(self):
        all_train_infos = [] 
        for epoch_id in range(self.num_epoch):
            if self.use_linear_lr_decay:
                for group_id in range(self.num_groups):
                    self.trainer[group_id].policy.lr_decay(epoch_id, self.num_epoch)
            train_infos = []
            val_infos = []
            for group_id in range(self.num_groups):
                self.trainer[group_id].prep_training()
                train_info = self.trainer[group_id].train(self.dataset[group_id])
                val_info = self.trainer[group_id].val(self.dataset[group_id])
                train_infos.append(train_info)
                val_infos.append(val_info) 
            
            if epoch_id % self.log_interval == 0:
                self.log_train(train_infos,epoch_id)

            if epoch_id % self.save_interval == 0:
                self.save(epoch_id)
            print('----------------------------------------------------------------------------')
            print("epoch:",epoch_id,"/",self.num_epoch)
            group_id = 0
            for tr_inf, va_inf in zip(train_infos,val_infos):
                print("Group {}:".format(group_id), end=' ')
                for key,value in tr_inf.items():
                    print(key,":",value, end=' ')
                for key,value in va_inf.items():
                    print(key,":",value, end=' ')
                group_id += 1
                print('')

        all_train_infos.append(train_infos)               
        return all_train_infos

    def save(self, episode):
        for group_id in range(self.num_groups):
            policy_actor = self.trainer[group_id].policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_group" + str(group_id) + "-ep" + str(episode) + ".pt")

    def restore(self, episode):
        for group_id in range(self.num_groups):
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_group' + str(group_id) + "-ep" + str(episode) + '.pt')
            self.policy[group_id].actor.load_state_dict(policy_actor_state_dict)

    def log_train(self, train_infos, total_num_steps): 
        for group_id in range(self.num_groups):
            for k, v in train_infos[group_id].items():
                agent_k = "group%i/" % group_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

