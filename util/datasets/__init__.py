from .core import TrajectoryDataset, LabelFunction
from .bball import BBallDataset
from .mouse_v1 import MouseV1Dataset


dataset_dict = {
    'bball' : BBallDataset,
    'mouse_v1' : MouseV1Dataset
}


def load_dataset(data_config):
    dataset_name = data_config['name'].lower()

    if dataset_name in dataset_dict:
        return dataset_dict[dataset_name](data_config)
    else:
        raise NotImplementedError
