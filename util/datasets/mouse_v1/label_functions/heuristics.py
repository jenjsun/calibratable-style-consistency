import torch
import numpy as np

from util.datasets import LabelFunction


class AverageSpeed(LabelFunction):

    name = 'average_speed'

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)

    def label_func(self, states, actions, true_label=None):
        vel = actions.view(actions.size(0), -1, 2)
        speed = torch.norm(vel, dim=-1)
        avg_speed = torch.mean(speed, dim=0)
        return torch.mean(avg_speed)

    def plot(self, ax, states, label, width, length):
        return ax


class NoseNoseDistance(LabelFunction):

    name = 'nose_nose_distance'

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)

    def label_func(self, states, actions, true_label=None):
        keypoints = states.view(states.size(0), 2, 7, 2)
        nose_distance = torch.norm(
            keypoints[:, 0, 0, :] - keypoints[:, 1, 0, :], dim=-1)
        return torch.mean(nose_distance)

    def plot(self, ax, states, label, width, length):
        return ax
