import queue
import time
from multiprocessing import Queue, Process

import cv2
from matplotlib.pyplot import axis  # pytype:disable=import-error
import numpy as np
from joblib import Parallel, delayed
import torch.utils.data as data
import torch
from collections import defaultdict

def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])

class ExpertDataset(object):
    """
    Dataset for using behavior cloning or GAIL.

    The structure of the expert dataset is a dict, saved as an ".npz" archive.
    The dictionary contains the keys 'actions', 'episode_returns', 'rewards', 'two-dim', 'one-dim' and 'episode_starts'.
    The corresponding values have data concatenated across episode: the first axis is the timestep,
    the remaining axes index into the data. In case of images, 'obs' contains the relative path to
    the images, to enable space saving from image compression.

    :param expert_path: (str) The path to trajectory data (.npz file). Mutually exclusive with traj_data.
    :param traj_data: (dict) Trajectory data, in format described above. Mutually exclusive with expert_path.
    :param train_fraction: (float) the train validation split (0 to 1)
        for pre-training using behavior cloning (BC)
    :param batch_size: (int) the minibatch size for behavior cloning
    :param traj_limitation: (int) the number of trajectory to use (if -1, load all)
    :param randomize: (bool) if the dataset should be shuffled
    :param verbose: (int) Verbosity
    :param sequential_preprocessing: (bool) Do not use subprocess to preprocess
        the data (slower but use less memory for the CI)
    """

    def __init__(self, args, expert_path=None, traj_data=None, traj_limitation=-1, 
                 randomize=True, verbose=1, sequential_preprocessing=False):
        
        self.batch_size = args.imitation_batch_size
        self.data_chunk_length = args.imitation_data_chunk_length
        self.train_fraction = args.imitation_train_fraction
        sample_batch = self.batch_size*self.data_chunk_length

        if traj_data is not None and expert_path is not None:
            raise ValueError("Cannot specify both 'traj_data' and 'expert_path'")
        if traj_data is None and expert_path is None:
            raise ValueError("Must specify one of 'traj_data' or 'expert_path'")
        if traj_data is None:
            traj_data = np.load(expert_path, allow_pickle=True)

        if verbose > 0:
            for key, val in traj_data.items():
                print(key, val.shape)
        

        # Array of bool where episode_starts[i] = True for each new episode
        episode_starts = traj_data['episode_starts']
        

        traj_limit_idx = traj_data['one-dim'].shape[0]

        if traj_limitation > 0:
            n_episodes = 0
            # Retrieve the index corresponding
            # to the traj_limitation trajectory
            for idx, episode_start in enumerate(episode_starts):
                n_episodes += int(episode_start)
                if n_episodes == (traj_limitation + 1):
                    traj_limit_idx = idx - 1

        traj_limit_idx *= traj_data['one-dim'].shape[1]

        observations = dict()
        for key in args.observation_dict:
            if len(traj_data[key].shape) == 5:
                observations[key] = traj_data[key][:traj_limit_idx].transpose(1, 0, 2, 3, 4).reshape(-1,*traj_data[key].shape[2:])
            if len(traj_data[key].shape) == 3:
                observations[key] = traj_data[key][:traj_limit_idx].transpose(1, 0, 2).reshape(-1,*traj_data[key].shape[2:])

        actions = traj_data['actions'][:traj_limit_idx]
        mask = traj_data['episode_starts'][:traj_limit_idx]

        self.num_batch_train = int(traj_limit_idx*self.train_fraction) // sample_batch
        self.num_batch_val = int(traj_limit_idx*(1-self.train_fraction)) // sample_batch
        # index of each chunk
        rand_train = torch.randperm(self.num_batch_train*self.batch_size).numpy()
        train_sampler = [rand_train[i*self.batch_size:(i+1)*self.batch_size] for i in range(self.num_batch_train)]

        rand_val = torch.randperm(self.num_batch_val*self.batch_size).numpy()
        val_sampler = [rand_val[i*self.batch_size:(i+1)*self.batch_size]+self.num_batch_train*self.batch_size for i in range(self.num_batch_val)]

        # Train/Validation split when using behavior cloning
        train_sampler = train_sampler
        val_sampler = val_sampler

        assert len(train_sampler) > 0, "No sample for the training set"
        assert len(val_sampler) > 0, "No sample for the validation set"

        self.observations = observations
        self.actions = actions.transpose(1, 0).reshape(-1,1)
        self.mask = np.expand_dims(mask,0).repeat(actions.shape[-1],axis=0).reshape(-1,1)

        self.returns = traj_data['episode_returns'][:traj_limit_idx]
        self.avg_ret = sum(self.returns) / len(self.returns)
        self.std_ret = np.std(np.array(self.returns))
        self.verbose = verbose

        for key in args.observation_dict:
            assert len(self.observations[key]) == len(self.actions), "The number of actions and observations differ " \
                                                                "please check your expert dataset"
        self.num_traj = min(traj_limitation, np.sum(episode_starts))
        self.num_transition = len(self.observations)
        self.randomize = randomize
        self.sequential_preprocessing = sequential_preprocessing

        self.train_loader = DataLoader(train_sampler, self.observations, self.actions, self.mask, self.data_chunk_length,self.batch_size,
                                       shuffle=self.randomize, start_process=False,
                                       sequential=sequential_preprocessing)

        self.val_loader = DataLoader(val_sampler, self.observations, self.actions, self.mask, self.data_chunk_length,self.batch_size,
                                       shuffle=self.randomize, start_process=False,
                                       sequential=sequential_preprocessing)

    def __del__(self):
        del self.train_loader, self.val_loader

    def prepare_pickling(self):
        """
        Exit processes in order to pickle the dataset.
        """
        self.train_loader, self.val_loader = None, None
    
    def initial_dataloader(self):

        # index of each chunk
        rand_train = torch.randperm(self.num_batch_train*self.batch_size).numpy()
        train_sampler = [rand_train[i*self.batch_size:(i+1)*self.batch_size] for i in range(self.num_batch_train)]

        rand_val = torch.randperm(self.num_batch_val*self.batch_size).numpy()
        val_sampler = [rand_val[i*self.batch_size:(i+1)*self.batch_size]+self.num_batch_train*self.batch_size for i in range(self.num_batch_val)]

        # Train/Validation split when using behavior cloning
        train_sampler = train_sampler
        val_sampler = val_sampler

        self.train_loader = DataLoader(train_sampler, self.observations, self.actions, self.mask, self.data_chunk_length,self.batch_size,
                                       shuffle=self.randomize, start_process=False,
                                       sequential=self.sequential_preprocessing)
        self.val_loader = DataLoader(val_sampler, self.observations, self.actions, self.mask, self.data_chunk_length,self.batch_size,
                                       shuffle=self.randomize, start_process=False,
                                       sequential=self.sequential_preprocessing)

    def get_next_batch(self, split=None):
        """
        Get the batch from the dataset.

        :param split: (str) the type of data split (can be None, 'train', 'val')
        :return: (np.ndarray, np.ndarray) inputs and labels
        """
        dataloader = {
            'train': self.train_loader,
            'val': self.val_loader
        }[split]

        if dataloader.process is None:
            dataloader.start_process()
        try:
            return next(dataloader)
        except StopIteration:
            dataloader = iter(dataloader)
            return next(dataloader)

    def plot(self):
        """
        Show histogram plotting of the episode returns
        """
        # Isolate dependency since it is only used for plotting and also since
        # different matplotlib backends have further dependencies themselves.
        import matplotlib.pyplot as plt
        plt.hist(self.returns)
        plt.show()


class DataLoader(object):
    """
    A custom dataloader to preprocessing observations (including images)
    and feed them to the network.

    Original code for the dataloader from https://github.com/araffin/robotics-rl-srl
    (MIT licence)
    Authors: Antonin Raffin, René Traoré, Ashley Hill

    :param indices: ([int]) list of observations indices
    :param observations: (np.ndarray) observations or images path
    :param actions: (np.ndarray) actions
    :param batch_size: (int) Number of samples per minibatch
    :param n_workers: (int) number of preprocessing worker (for loading the images)
    :param infinite_loop: (bool) whether to have an iterator that can be reset
    :param max_queue_len: (int) Max number of minibatches that can be preprocessed at the same time
    :param shuffle: (bool) Shuffle the minibatch after each epoch
    :param start_process: (bool) Start the preprocessing process (default: True)
    :param backend: (str) joblib backend (one of 'multiprocessing', 'sequential', 'threading'
        or 'loky' in newest versions)
    :param sequential: (bool) Do not use subprocess to preprocess the data
        (slower but use less memory for the CI)
    :param partial_minibatch: (bool) Allow partial minibatches (minibatches with a number of element
        lesser than the batch_size)
    """

    def __init__(self, indices, observations, actions, mask, data_chunk_length,batch_size, n_workers=1,
                 infinite_loop=True, max_queue_len=1, shuffle=False,
                 start_process=True, backend='threading', sequential=False):
        super(DataLoader, self).__init__()
        self.n_workers = n_workers
        self.infinite_loop = infinite_loop
        self.indices = indices
        self.original_indices = indices.copy()
        self.n_minibatches = len(indices)
        self.batch_size = batch_size
        self.data_chunk_length = data_chunk_length
        self.observations = observations
        self.actions = actions
        self.mask = mask
        self.shuffle = shuffle
        self.queue = Queue(max_queue_len)
        self.process = None
        self.backend = backend
        self.sequential = sequential
        self.start_idx = 0
        if start_process:
            self.start_process()

    def start_process(self):
        """Start preprocessing process"""
        # Skip if in sequential mode
        if self.sequential:
            return
        self.process = Process(target=self._run)
        # Make it a deamon, so it will be deleted at the same time
        # of the main process
        self.process.daemon = True
        self.process.start()


    def sequential_next(self):
        """
        Sequential version of the pre-processing.
        """
        if self.start_idx > len(self.indices):
            raise StopIteration

        obs_batch = defaultdict(list)
        actions_batch = []
        mask_batch = []
        indices = self.indices[self.start_idx]
        for index in indices:
            ind = index * self.data_chunk_length

            for key in self.observations.keys():
                obs_batch[key].append(self.observations[key][ind:ind+self.data_chunk_length])
            actions_batch.append(self.actions[ind:ind+self.data_chunk_length])
            mask_batch.append(self.mask[ind:ind+self.data_chunk_length])

        for key in self.observations.keys():
            obs_batch[key] = np.stack(obs_batch[key], 1)
        actions_batch = np.stack(actions_batch, axis=1)
        mask_batch = np.stack(mask_batch, axis=1)

        # Flatten the (L, N, ...) from_numpys to (L * N, ...)
        L, N = self.data_chunk_length, self.batch_size
        for key in self.observations.keys():
            obs_batch[key] = _flatten(L, N, obs_batch[key])
        actions_batch = _flatten(L, N, actions_batch)
        mask_batch = _flatten(L, N, mask_batch)

        self.start_idx += 1
        return obs_batch, actions_batch, mask_batch

    def _run(self):
        start = True
        with Parallel(n_jobs=self.n_workers, batch_size="auto", backend=self.backend) as parallel:
            while start or self.infinite_loop:
                start = False

                for indices in self.indices:
                    obs_batch = defaultdict(list)
                    actions_batch = []
                    mask_batch = []
                    for index in indices:
                        ind = index * self.data_chunk_length

                        for key in self.observations.keys():
                            obs_batch[key].append(self.observations[key][ind:ind+self.data_chunk_length])
                        actions_batch.append(self.actions[ind:ind+self.data_chunk_length])
                        mask_batch.append(self.mask[ind:ind+self.data_chunk_length])

                    for key in self.observations.keys():
                        obs_batch[key] = np.stack(obs_batch[key], 1)
                    actions_batch = np.stack(actions_batch, axis=1)
                    mask_batch = np.stack(mask_batch, axis=1)

                    # Flatten the (L, N, ...) from_numpys to (L * N, ...)
                    L, N = self.data_chunk_length, self.batch_size
                    for key in self.observations.keys():
                        obs_batch[key] = _flatten(L, N, obs_batch[key])
                    actions_batch = _flatten(L, N, actions_batch)
                    mask_batch = _flatten(L, N, mask_batch)

                    self.queue.put((obs_batch, actions_batch, mask_batch))

                    # Free memory
                    del obs_batch

                self.queue.put(None)

    def __len__(self):
        return self.n_minibatches

    def __iter__(self):
        self.start_idx = 0
        self.indices = self.original_indices.copy()
        return self

    def __next__(self):
        if self.sequential:
            return self.sequential_next()

        if self.process is None:
            raise ValueError("You must call .start_process() before using the dataloader")
        while True:
            try:
                val = self.queue.get_nowait()
                break
            except queue.Empty:
                time.sleep(0.001)
                continue
        if val is None:
            raise StopIteration
        return val

    def __del__(self):
        if self.process is not None:
            self.process.terminate()



def main():
    from onpolicy.config import get_config
    parser = get_config()
    args = parser.parse_known_args()[0]
    args.env_name = "MPE"
    args.scenario_name = "simple_catching_expert_both"
    args.num_agents = 4
    dataset = ExpertDataset(args, args.expert_path_gp0)
    for i in range(len(dataset.train_loader)):
        expert_obs, expert_actions, mask = dataset.get_next_batch('train')
        for key in args.observation_dict:
            print("expert_obs",expert_obs[key].shape)
        print("expert_actions",expert_actions.shape)
        print("mask",mask.shape)

if __name__=="__main__":
   main()
