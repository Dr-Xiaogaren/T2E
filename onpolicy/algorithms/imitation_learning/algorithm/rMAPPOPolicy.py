import torch
from onpolicy.algorithms.imitation_learning.algorithm.r_actor_critic import R_CNNActor
from onpolicy.utils.util import update_linear_schedule


class R_MAPPOPolicy_ForBC:


    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.imitation_lr
        self.opti_eps = args.imitation_opti_eps
        self.weight_decay = args.imitation_weight_decay

        self.obs_space = obs_space
        self.act_space = act_space

        self.actor = R_CNNActor(args, self.obs_space, self.act_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
    
    def lr_decay(self, epoch, all_epoch):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, epoch, all_epoch, self.lr)

    def get_actions_probs(self, obs, rnn_states_actor, masks, available_actions=None):

        action_probs = self.actor.get_probs(obs,
                                            rnn_states_actor,
                                            masks,
                                            available_actions
                                            )

        return action_probs

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):

        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
