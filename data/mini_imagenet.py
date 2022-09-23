import os
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from core.workspace import Registers


@Registers.dataset.register
class MiniImageNet(Dataset):
    """
    Args:
        root_path (str): root directory for dataset.
        train_file_path (str): file path for train data.
        val_file_path (str): file path for val data.
        test_file_path (str): file path for test data.
        mode (str): "train" or "test".
        image_size (int): image size.
        preprocess(object): the preprocess operation of images.
    """
    def __init__(self, root_path, train_file_path=None,
                 val_file_path=None, test_file_path=None,
                 mode='train', image_size=84,
                 preprocess=None):
        super(MiniImageNet, self).__init__()
        if mode == "train":
            dataset_file = os.path.join(root_path, train_file_path)
        elif mode == "val":
            dataset_file = os.path.join(root_path, val_file_path)
        elif mode == "test":
            dataset_file = os.path.join(root_path, test_file_path)
        assert os.path.isfile(dataset_file)

        with open(dataset_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        images, labels = data['data'], data['labels']

        images = [Image.fromarray(x) for x in images]
        labels = np.array(labels)
        label_key = sorted(np.unique(labels))
        label_map = dict(zip(label_key, range(len(label_key))))
        new_label = np.array([label_map[x] for x in label])

        self.image_size = image_size
        self.images = images
        self.labels = new_label
        self.n_classes = len(label_key)
        self.transform = get_transform(preprocess, self.image_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.transform(self.images[index])
        label = self.labels[index]
        return image, label


@Registers.dataset.register
class MetaMiniImageNet(MiniImageNet):
    def __init__(self, root_path, split='train', image_size=84,
                 normalization=True, transform=None, val_transform=None,
                 n_batch=200, n_episode=4, n_way=5, n_shot=1, n_query=15):
        super(MetaMiniImageNet, self).__init__(root_path, split, image_size,
                                               normalization, transform)
        self.n_batch = n_batch
        self.n_episode = n_episode
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query

        self.catlocs = tuple()
        for cat in range(self.n_classes):
            self.catlocs += (np.argwhere(self.label == cat).reshape(-1),)

        self.val_transform = get_transform(
            val_transform, image_size, self.norm_params)

    def __len__(self):
        return self.n_batch * self.n_episode

    def __getitem__(self, index):
        shot, query = [], []
        cats = np.random.choice(self.n_classes, self.n_way, replace=False)
        for c in cats:
            c_shot, c_query = [], []
            idx_list = np.random.choice(
                self.catlocs[c], self.n_shot + self.n_query, replace=False)
            shot_idx, query_idx = idx_list[:self.n_shot], idx_list[-self.n_query:]
            for idx in shot_idx:
                c_shot.append(self.transform(self.data[idx]))
            for idx in query_idx:
                c_query.append(self.val_transform(self.data[idx]))
            shot.append(torch.stack(c_shot))
            query.append(torch.stack(c_query))

        shot = torch.cat(shot, dim=0)  # [n_way * n_shot, C, H, W]
        query = torch.cat(query, dim=0)  # [n_way * n_query, C, H, W]
        cls = torch.arange(self.n_way)[:, None]
        shot_labels = cls.repeat(1, self.n_shot).flatten()  # [n_way * n_shot]
        query_labels = cls.repeat(1, self.n_query).flatten()  # [n_way * n_query]

        return shot, query, shot_labels, query_labels


