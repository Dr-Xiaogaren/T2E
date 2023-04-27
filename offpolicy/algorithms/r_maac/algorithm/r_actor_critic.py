from cv2 import merge
import torch
import torch.nn as nn
from offpolicy.algorithms.utils.util import init, check
from offpolicy.algorithms.utils.cnn import CNNBase
from offpolicy.algorithms.utils.mlp import MLPBase
from offpolicy.algorithms.utils.rnn import RNNLayer
from offpolicy.algorithms.utils.act import ACTLayer
from offpolicy.utils.util import get_shape_from_obs_space

class R_CNNCritic(nn.Module):
    """
    The critic network of MAAC.
    :param args: (argparse.Namespace) arguments including hyperparameters and training settings.
    :param obs_space: (gym.Space) observation space of the environment.
    :param cent_obs_space: (gym.Space) central observation space of the environment.
    :param action_space: (gym.Space) action space of the environment.
    :param device: (torch.device) device to put the network on.
    """
    def __init__(self, args, obs_space, cent_obs_space, action_space, device=torch.device("cpu")):
        super(R_CNNCritic, self).__init__()
        self.hidden_size = args.hidden_size

        self._use_orthogonal = args.use_orthogonal
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N

        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        
        obs_shape = get_shape_from_obs_space(obs_space)
        size_for_fc = obs_shape["one-dim"].shape[0]
        self.obs_height = obs_shape["two-dim"].shape[1]
        self.num_channel = obs_shape["two-dim"].shape[0]
        self.size_for_fc = size_for_fc
        self.num_agents = cent_obs_shape["two-dim"].shape[0] // obs_shape["two-dim"].shape[0]
        assert self.num_agents == cent_obs_shape["one-dim"].shape[0] // obs_shape["one-dim"].shape[0]


        self.State_Encoder = MLPBase(args, input_size=size_for_fc, layer_N= 3, hidden_size=self.hidden_size)

        self.CNNbase = CNNBase(args,  self.num_channel, self.obs_height)    

        self.action_dim = action_space.n

        self.SA_Encoder = MLPBase(args, input_size=self.hidden_size+self.action_dim+self.CNNbase.output_size, layer_N= 3, hidden_size=self.hidden_size)

        merge_input_size = (self.num_agents+1)*(self.hidden_size) + self.CNNbase.output_size
        # merge_input_size = self.hidden_size

        self.MergeLayer = MLPBase(args, input_size=merge_input_size, layer_N=3, hidden_size=self.hidden_size)


        if self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_size, self.action_dim))

        self.to(device)
        self.device = device

    def forward(self, obs_batch, action_batch, rnn_states, masks, return_all_value=False):
        """
        Compute actions from the given inputs while inference.
        :param obs_batch: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.
        :param action_batch:(np.ndarray / torch.Tensor) action inputs into network, one-hot format or batch format.
        
        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs_for_cnn = check(obs_batch["two-dim"]).to(**self.tpdv).reshape(-1,self.num_channel,self.obs_height,self.obs_height)
        cent_obs_for_fc = check(obs_batch["one-dim"]).to(**self.tpdv).reshape(-1,self.size_for_fc)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action_batch = check(action_batch).to(**self.tpdv).reshape(-1,action_batch.shape[-1])
        masks = check(masks).to(**self.tpdv)
        # if not one-hot format
        if action_batch.shape[-1] != self.action_dim:
            action_batch = torch.zeros(action_batch.shape[0],self.action_dim, device=self.device).scatter(1,action_batch.to(dtype=torch.int64),1)
        
        state_feature = self.State_Encoder(cent_obs_for_fc)
        critic_features_from_cnn = self.CNNbase(cent_obs_for_cnn)

        state_feature = torch.concat([state_feature, critic_features_from_cnn],dim=-1)

        cent_obs_a = torch.cat([state_feature,action_batch], dim=-1)
        state_action_feature = self.SA_Encoder(cent_obs_a)
        # encode respectively      
        
        
        # critic_features_from_cnn = critic_features_from_cnn.reshape(-1, self.num_agents,critic_features_from_cnn.size(-1)).repeat(1,self.num_agents,1)
        state_action_feature = state_action_feature.reshape(-1,self.num_agents,state_action_feature.size(-1)).repeat(1,self.num_agents,1)


        # critic_features_from_cnn = critic_features_from_cnn.reshape(-1,self.num_agents*critic_features_from_cnn.size(-1))
        state_action_feature = state_action_feature.reshape(-1,self.num_agents*state_action_feature.size(-1))
        # concat and merge
        critic_features = self.MergeLayer(torch.cat([state_feature, state_action_feature], dim=-1))

        if self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        all_values = self.v_out(critic_features)
        int_acs = action_batch.max(dim=1, keepdim=True)[1]
        values  = all_values.gather(1, int_acs)
        
        if return_all_value:
            return values, all_values  
        else:
            return values, rnn_states

class R_CNNActor(nn.Module):
    """
    Actor network for RNN policy.
    :param args: (argparse.Namespace) arguments including hyperparameters and training settings.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) device to put the network on.
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_CNNActor, self).__init__()

        self.hidden_size = args.hidden_size
        self.obs_height = args.obs_map_size
        
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        size_for_fc = obs_shape["one-dim"].shape[0]
        self.obs_height = obs_shape["two-dim"].shape[1]
        self.num_channel = obs_shape["two-dim"].shape[0]
        self.size_for_fc = size_for_fc

        self.FCbase = MLPBase(args, input_size=size_for_fc, layer_N= 3, hidden_size=self.hidden_size)
        self.CNNbase = CNNBase(args,  self.num_channel, self.obs_height)    

        merge_input_size = self.hidden_size + self.CNNbase.output_size

        self.MergeLayer = MLPBase(args, input_size=merge_input_size, layer_N=3, hidden_size=self.hidden_size)

        if self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_probs: (torch.Tensor) probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs_for_cnn = check(obs["two-dim"]).to(**self.tpdv).reshape(-1,self.num_channel,self.obs_height,self.obs_height)
        obs_for_fc = check(obs["one-dim"]).to(**self.tpdv).reshape(-1,self.size_for_fc)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        
        # encode respectively      
        actor_features_from_cnn = self.CNNbase(obs_for_cnn)
        actor_features_from_fc = self.FCbase(obs_for_fc)
        # concat and merge
        actor_features = self.MergeLayer(torch.cat([actor_features_from_fc, actor_features_from_cnn], dim=-1))
        # rnn layer
        if self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        # output layer
        actions, action_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_probs, rnn_states


class CNNActorAndCritic(nn.Module):
    """
     Actor-Critic network for MAAC
    :param args: (argparse.Namespace) arguments including hyperparameters and training settings.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) centralized observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) device to put the network on.
    """
    def __init__(self,args, obs_space, cent_obs_space,action_space, device):
        super(CNNActorAndCritic, self).__init__()
        self.hidden_size = args.hidden_size
        obs_shape = get_shape_from_obs_space(obs_space)
        self.obs_height = obs_shape["two-dim"].shape[1]
        self.num_channel = obs_shape["two-dim"].shape[0]
        size_for_fc = obs_shape["one-dim"].shape[0]
        self.size_for_fc = size_for_fc

        self.actor = R_CNNActor(args, obs_space, action_space, device=device)
        self.critic = R_CNNCritic(args, obs_space, cent_obs_space, action_space, device=device)

        self.to(device)