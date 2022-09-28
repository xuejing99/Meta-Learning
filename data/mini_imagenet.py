import os
import pickle
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from core.workspace import Registers
from data.data_helper import get_transform


@Registers.dataset.register
class MiniImageNet(Dataset):
    """
    Args:
        root_path (str): root directory for dataset.
        train_file_path (str): file path for train data.
        val_file_path (str): file path for val data.
        test_file_path (str): file path for test data.
        mode (str): "train" or "test".
        preprocess(object): the preprocess operation of images.
    """
    def __init__(self, root_path, file_path=None,
                 mode='train', preprocess=None):
        super(MiniImageNet, self).__init__()
        dataset_file = os.path.join(root_path, file_path)
        assert os.path.isfile(dataset_file)

        with open(dataset_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        images, labels = data['data'], data['labels']

        images = [Image.fromarray(x) for x in images]
        labels = np.array(labels)
        label_key = sorted(np.unique(labels))
        label_map = dict(zip(label_key, range(len(label_key))))
        new_label = np.array([label_map[x] for x in labels])

        self.images = images
        self.labels = new_label
        self.n_classes = len(label_key)
        self.transform = get_transform(preprocess)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.transform(self.images[index])
        label = self.labels[index]
        return image, label


@Registers.dataset.register
class MetaMiniImageNet(MiniImageNet):
    """
    Args:
        root_path (str): root directory for dataset.
        train_file_path (str): file path for train data.
        val_file_path (str): file path for val data.
        test_file_path (str): file path for test data.
        mode (str): "train" or "test".
        preprocess(object): the preprocess operation of images.
        n_batch (int): batch size
        n_episode (int): episode number to train meta-model
        n_way (int): the number of classes to train a meta-model
        n_shot (int): the number of examples of each class to train a meta-model
        n_query (int): the number of samples to test the trained meta-model
    """
    def __init__(self, root_path, file_path=None,
                 mode='train', preprocess=None, n_batch=200,
                 n_episode=4, n_way=5, n_shot=1, n_query=15):
        super(MetaMiniImageNet, self).__init__(root_path, file_path,
                                               mode, preprocess)
        self.n_batch = n_batch
        self.n_episode = n_episode
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query

        self.categories = tuple()
        for cat in range(self.n_classes):
            self.categories += (np.argwhere(self.labels == cat).reshape(-1),)

    def __len__(self):
        return self.n_batch * self.n_episode

    def __getitem__(self, index):
        shot, query = [], []
        cats = np.random.choice(self.n_classes, self.n_way, replace=False)
        for c in cats:
            c_shot, c_query = [], []
            idx_list = np.random.choice(
                self.categories[c], self.n_shot + self.n_query, replace=False)
            shot_idx, query_idx = idx_list[:self.n_shot], idx_list[-self.n_query:]
            for idx in shot_idx:
                c_shot.append(self.transform(self.images[idx]))
            for idx in query_idx:
                c_query.append(self.transform(self.images[idx]))
            shot.append(torch.stack(c_shot))
            query.append(torch.stack(c_query))

        shot = torch.cat(shot, dim=0)  # [n_way * n_shot, C, H, W]
        query = torch.cat(query, dim=0)  # [n_way * n_query, C, H, W]
        cls = torch.arange(self.n_way)[:, None]
        shot_labels = cls.repeat(1, self.n_shot).flatten()  # [n_way * n_shot]
        query_labels = cls.repeat(1, self.n_query).flatten()  # [n_way * n_query]

        return shot, query, shot_labels, query_labels

    def collate_fn(self, batch):
        shot, query, shot_label, query_label = [], [], [], []
        for s, q, sl, ql in batch:
            shot.append(s)
            query.append(q)
            shot_label.append(sl)
            query_label.append(ql)

        shot = torch.stack(shot)  # [n_ep, n_way * n_shot, C, H, W]
        query = torch.stack(query)  # [n_ep, n_way * n_query, C, H, W]
        shot_label = torch.stack(shot_label)  # [n_ep, n_way * n_shot]
        query_label = torch.stack(query_label)  # [n_ep, n_way * n_query]

        return shot, query, shot_label, query_label


