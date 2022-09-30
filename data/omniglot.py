import os
import pickle
import numpy as np
import random
from PIL import Image
import torch
from torch.utils.data import Dataset

from core.workspace import Registers
from data.data_helper import get_transform


@Registers.dataset.register
class Omniglot(Dataset):
    """
    Args:
        root_path (str): root directory for dataset.
        mode (str): "train" or "test".
        preprocess(object): the preprocess operation of images.
    """
    def __init__(self, root_path, file_path, mode='train', preprocess=None):
        super(Omniglot, self).__init__()
        data_folder = os.path.join(root_path, file_path)

        character_folders = [os.path.join(data_folder, family, character)
                             for family in os.listdir(data_folder)
                             if os.path.isdir(os.path.join(data_folder, family))
                             for character in os.listdir(os.path.join(data_folder, family))
                             if os.path.isdir(os.path.join(data_folder, family, character))]
        random.seed(1)
        random.shuffle(character_folders)
        meta_train = 1100
        meta_val = 100

        if mode == "train":
            self.characters = character_folders[: meta_train]
        elif mode == "val":
            self.characters = character_folders[meta_train: meta_train + meta_val]
        else:
            self.characters = character_folders[meta_train + meta_val:]

        labels_images = [(character, os.path.join(character, images))
                         for character in self.characters
                         for images in os.listdir(character)]
        images = [item[1] for item in labels_images]
        labels = [item[0] for item in labels_images]
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
        image = self.transform(mage.open(self.images[index]).convert('RGB'))
        label = self.labels[index]
        return image, label


@Registers.dataset.register
class MetaOmniglot(Omniglot):
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
                 n_episode=1, n_way=5, n_shot=1, n_query=15):
        super(MetaOmniglot, self).__init__(root_path, file_path,
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
                c_shot.append(self.transform(Image.open(self.images[idx]).convert('RGB')))
            for idx in query_idx:
                c_query.append(self.transform(Image.open(self.images[idx]).convert('RGB')))
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