import torch
from offpolicy.algorithms.r_maddpg.algorithm.r_actor_critic import R_Actor, R_Critic, R_CNNActor, R_CNNCritic, CNNActorAndCritic
from offpolicy.utils.util import update_linear_schedule, soft_update, hard_update
from offpolicy.algorithms.utils.util import check

class R_MADDPGPolicy:
    """
    MADDPG Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param obs_space: (gym.spaces) observation space of the environment.
    :cent_obs_space: (gym.spaces) central observation space of the environment.
    :param act_space: (gym.spaces) action space of the environment.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.tau = args.tau

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor_critic = CNNActorAndCritic(args,self.obs_space,self.share_obs_space, self.act_space, self.device)

        self.target_actor_critic = CNNActorAndCritic(args,self.obs_space,self.share_obs_space, self.act_space, self.device)

        self.actor = self.actor_critic.actor
        self.critic = self.actor_critic.critic

        self.target_actor = self.target_actor_critic.actor
        self.target_critic = self.target_actor_critic.critic

        # sync the target weights
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False, use_target_actor=True, use_target_critic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        :param use_target_actor: (bool) whether to use the target actor network.
        :param use_target_critic: (bool) whether to use the target critic network.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_probs: (torch.Tensor) probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        if use_target_actor:
            actions, action_probs, rnn_states_actor = self.target_actor(obs,
                                                                    rnn_states_actor,
                                                                    masks,
                                                                    available_actions,
                                                                    deterministic)
        else:
            actions, action_probs, rnn_states_actor = self.actor(obs,
                                                                    rnn_states_actor,
                                                                    masks,
                                                                    available_actions,
                                                                    deterministic)
        action_dim = self.act_space.n
        one_hot_action = torch.zeros(actions.shape[0],action_dim, device=self.device).scatter(1,actions,1)
        if use_target_critic:
            values, rnn_states_critic = self.target_critic(obs,one_hot_action,rnn_states_critic,masks)
        else:
            values, rnn_states_critic = self.critic(obs,one_hot_action,rnn_states_critic,masks)
        return values, actions, action_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, obs, joint_action,rnn_states_critic, masks, use_target=True):
        """
        Get value function predictions.
        :param obs (np.ndarray): centralized input to the critic.
        :param joint_action (np.ndarray): joint action input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param use_target: (bool) whether to use the target critic network.

        :return values: (torch.Tensor) value function predictions.
        """
        if use_target:
            values, _ = self.target_critic(obs, joint_action, rnn_states_critic, masks)
        else:
            values, _ = self.critic(obs, joint_action, rnn_states_critic, masks)
        return values

    def evaluate_actions(self,obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False, use_target_actor=True, use_target_critic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        :param use_target_actor: (bool) whether to use the target actor network.
        :param use_target_critic: (bool) whether to use the target critic network.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_probs: (torch.Tensor) probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        if use_target_actor:
            actions, action_probs, rnn_states_actor = self.target_actor(obs,
                                                                    rnn_states_actor,
                                                                    masks,
                                                                    available_actions,
                                                                    deterministic)
        else:
            actions, action_probs, rnn_states_actor = self.actor(obs,
                                                                    rnn_states_actor,
                                                                    masks,
                                                                    available_actions,
                                                                    deterministic)
        action_dim = self.act_space.n
        one_hot_action = torch.nn.functional.gumbel_softmax(action_probs, hard=True)
        if use_target_critic:
            values, rnn_states_critic = self.target_critic(obs,one_hot_action,rnn_states_critic,masks)
        else:
            values, rnn_states_critic = self.critic(obs,one_hot_action,rnn_states_critic,masks)
        return values, actions, action_probs, rnn_states_actor, rnn_states_critic

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return actions: (torch.Tensor) actions to take.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor

    def soft_target_updates(self):
        """Soft update the target networks through a Polyak averaging update."""
        soft_update(self.target_critic, self.critic, self.tau)
        soft_update(self.target_actor, self.actor, self.tau)

    def hard_target_updates(self):
        """Hard update target networks by copying the weights of the live networks."""
        hard_update(self.target_critic, self.critic)
        hard_update(self.target_actor, self.actor)