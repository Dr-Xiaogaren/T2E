import argparse


def get_config():
    
    parser = argparse.ArgumentParser(
        description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str,
                        default='mappo', choices=["rmappo", "mappo", "rippo"])

    parser.add_argument("--experiment_name", type=str, default="check", help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--cuda", action='store_false', default=True, help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True, help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int, default=32,
                        help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for evaluating rollouts")
    parser.add_argument("--n_render_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for rendering rollouts")
    parser.add_argument("--num_env_steps", type=int, default=10e6,
                        help='Number of environment steps to train (default: 10e6)')
    parser.add_argument("--user_name", type=str, default='marl',help="[for wandb usage], to specify user's name for simply collecting training data.")
    parser.add_argument("--use_wandb", action='store_false', default=True, help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.")

    # env parameters
    parser.add_argument("--env_name", type=str, default='StarCraft2', help="specify the name of environment")
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int,
                        default=200, help="Max length for any episode")

    # network parameters
    parser.add_argument("--share_policy", action='store_false',
                        default=True, help='Whether agent share the same policy')
    parser.add_argument("--use_centralized_V", action='store_false',
                        default=True, help="Whether to use centralized V function")
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames", action='store_true',
                        default=False, help="Whether to use stacked_frames")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Dimension of hidden layers for actor/critic networks") 
    parser.add_argument("--layer_N", type=int, default=1,
                        help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", action='store_false',
                        default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart", action='store_true', default=False, help="by default False, use PopArt to normalize rewards.")
    parser.add_argument("--use_valuenorm", action='store_false', default=True, help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization", action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")

    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", action='store_true',
                        default=False, help='Whether to use a naive recurrent policy')
    parser.add_argument("--use_recurrent_policy", action='store_false',
                        default=True, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, default=10,
                        help="Time length of chunks used to train a recurrent_policy")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True, help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True, help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')
    # save parameters
    parser.add_argument("--save_interval", type=int, default=100, help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=5, help="time duration between contiunous twice log printing.")

    # eval parameters
    parser.add_argument("--use_eval", action='store_true', default=False, help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=25, help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=32, help="number of episodes of a single evaluation.")

    # render parameters
    parser.add_argument("--save_gifs", action='store_true', default=False, help="by default, do not save render video. If set, save video.")
    parser.add_argument("--use_render", action='store_true', default=False, help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--render_episodes", type=int, default=5, help="the number of episodes to render a given env")
    parser.add_argument("--ifi", type=float, default=0.1, help="the play interval of each rendered image in saved video.")

    # pretrained parameters
    parser.add_argument("--model_dir", type=str, default=None, help="by default None. set the path to pretrained model.")
    parser.add_argument("--load_model_ep", type=int, default=2360, help="by default None. the trained episode of pretrained model.")

    # environment
    parser.add_argument("--maps_path", type=str, default="./scene/candidate", help="by default None. set the path to the gibson travasable map image directory.")
    parser.add_argument("--trav_map_default_resolution", type=float, default=0.01, help="by default 0.01, the default resolution of the travasable map")
    parser.add_argument("--trav_map_resolution", type=float, default=0.05, help="by default 0.05, the actual resolution of the travasable map")
    parser.add_argument("--num_good_agents", type=int, default=1, help="by default 1, the number of prey")
    parser.add_argument("--num_adversaries", type=int, default=3, help="by default 3, the number of predators")
    parser.add_argument("--max_initial_inner_distance", type=float, default=4, help = "by default 5, the max distance between adversaries and good")
    parser.add_argument("--max_initial_inter_distance", type=float, default=10, help = "by default 20, the max distance between adversaries and good")
    parser.add_argument("--min_initial_distance", type=float, default=1, help = "by default 1, the min distance between all agents")
    parser.add_argument("--max_map_size", type=int, default=1600, help="the maxsize of the loaded traversable map")
    parser.add_argument("--obs_map_size", type=int, default=48, help="the size of observed map")
    parser.add_argument("--observation_dict", type=list, default=["one-dim","two-dim"], help="the keys of observation space")
    parser.add_argument("--step_mode", type=str, default="none", help=" assert mode == expert_adversary or mode == expert_both or mode == expert_prey or mode == none" )
    parser.add_argument("--use_intrinsic_reward", action='store_true', default=False, help="use_intrinsic_reward")
    parser.add_argument("--adversary_speed", type=float,default=1.0, help='Speed of Predator, m/s')
    parser.add_argument("--good_agent_speed", type=float,default=1.0, help='Speed of Prey, m/s')
    parser.add_argument("--use_strict_terminal", action='store_true', default=False, help="if use the restrict terminal condition")
    
    # imitation learning
    parser.add_argument("--expert_path_gp0", type=str, default="/workspace/tmp/data/chaser_trajectory_1.0.npz", help="set the path to the expert trajectory of predators")
    parser.add_argument("--expert_path_gp1", type=str, default="/workspace/tmp/data/evader_trajectory_1.0.npz", help="set the path to the expert trajectory of preys")
    parser.add_argument("--imitation_batch_size", type=int, default=16, help="the batchsize of training")
    parser.add_argument("--imitation_num_epoch", type=int, default=1000, help="the batchsize of training")
    parser.add_argument("--imitation_train_fraction", type=float, default=0.8, help="the train validation split (0 to 1)for pre-training using behavior cloning (BC)")
    parser.add_argument("--imitation_save_interval", type=int, default=50, help=" time duration between contiunous twice models saving in imitation learning")
    parser.add_argument("--imitation_data_chunk_length", type=int, default=100, help="Time length of chunks used to train a recurrent_policy in imitation learning")
    parser.add_argument("--imitation_lr", type=float, default=0.9, help="the learning rate of imitation learning") 
    parser.add_argument("--imitation_opti_eps", type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--imitation_weight_decay", type=float, default=0)
    parser.add_argument("--imitation_use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')
    parser.add_argument("--imitation_log_interval", type=int, default=1, help="the batchsize of training")
    return parser
