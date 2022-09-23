import os
import pickle
from torch.utils.data import Dataset

from core.workspace import Registers


@Registers.dataset.register
class MiniImageNet(Dataset):
    def __init__(self):
        super(MiniImageNet, self).__init__()
        self.images = []
        self.labels = []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label


@Registers.dataset.register
class MetaMiniImageNet(MiniImageNet):
    def __init__(self, root_path, mode='train', image_size=84,
                 n_batch=200, n_episode=4, n_way=5, n_shot=1, n_query=15):
        super(MetaMiniImageNet, self).__init__(root_path, mode, image_size)
        self.n_batch = n_batch
        self.n_episode = n_episode
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query

        self.categories = tuple()
        for cat in range(self.n_classes):
            self.categories += (np.argwhere(self.label == cat).reshape(-1),)

    def __len__(self):
        return self.n_batch * self.n_episode

    def __getitem__(self, index):
        shot, query = [], []
        cats = np.random.choice(self.n_classes, self.n_way, replace=False)
        for cat in cats:
            cat_shot, c_query = [], []
            idx_list = np.random.choice(
                self.categories[c], self.n_shot+self.n_query, replace=False
            )
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


