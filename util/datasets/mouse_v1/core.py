import os
import numpy as np
import torch

from util.datasets import TrajectoryDataset
from .label_functions import label_functions_list


# TODO let users define where data lies
ROOT_DIR = 'util/datasets/mouse_v1/data'
TRAIN_FILE = 'train.npz'
TEST_FILE = 'test.npz'

FRAME_WIDTH_TOP = 1024
FRAME_HEIGHT_TOP = 570


class MouseV1Dataset(TrajectoryDataset):

    name = 'mouse_v1'
    all_label_functions = label_functions_list

    # Default config
    _seq_len = 100
    _state_dim = 28
    _action_dim = 28

    normalize_data = True
    single_agent = False

    def __init__(self, data_config):
        super().__init__(data_config)

    def _load_data(self):
        # Process configs
        if 'normalize_data' in self.config:
            self.normalize_data = self.config['normalize_data']
        if 'single_agent' in self.config:
            self.single_agent = self.config['single_agent']

        # TODO hacky solution
        if 'labels' in self.config:
            for lf_config in self.config['labels']:
                lf_config['data_normalized'] = self.normalize_data

        self.train_states, self.train_actions = self._load_and_preprocess(train=True)
        self.test_states, self.test_actions = self._load_and_preprocess(train=False)

    def _load_and_preprocess(self, train):
        path = os.path.join(ROOT_DIR, TRAIN_FILE if train else TEST_FILE)
        file = np.load(path)
        data = file['data']

        # Subsample timesteps
        data = data[:,::self.subsample]

        # Normalize data
        if self.normalize_data:
            data = normalize(data)

        # Convert to states and actions
        states = data
        actions = states[:,1:] - states[:,:-1]

        # Update dimensions
        self._seq_len = actions.shape[1]
        self._state_dim = states.shape[-1]
        self._action_dim = actions.shape[-1]

        print(states.shape)
        print(actions.shape)

        return torch.Tensor(states), torch.Tensor(actions)

    def save(self):
        pass

def normalize(data):
    """Scale by dimensions of image and mean-shift to center of image."""
    state_dim = data.shape[2]//2
    shift = [int(FRAME_WIDTH_TOP/2), int(FRAME_HEIGHT_TOP/2)] * state_dim
    scale = [int(FRAME_WIDTH_TOP/2), int(FRAME_HEIGHT_TOP/2)] * state_dim
    return np.divide(data-shift, scale)

def unnormalize(data):
    """Undo normalize."""
    state_dim = data.shape[2]//2
    shift = [int(FRAME_WIDTH_TOP/2), int(FRAME_HEIGHT_TOP/2)] * state_dim
    scale = [int(FRAME_WIDTH_TOP/2), int(FRAME_HEIGHT_TOP/2)] * state_dim
    return np.multiply(data, scale) + shift

def _set_figax():
    pass