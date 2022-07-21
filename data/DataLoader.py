import os
import random
import numpy as np
from sklearn.utils import shuffle
from data.utils import get_images, image_file_to_array, plot_images, image_transform


class OmniglotGenerator(object):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(self, data_folder, num_classes, num_samples_per_class, image_size):
        """
        Args:
            num_classes: int
                Number of classes for classification (N-way)

            num_samples_per_class: int
                Number of samples per class in the support set (K-shot).
                Will generate additional sample for the querry set.
        """
        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class

        self.image_size = image_size
        self.input_dim = np.prod(self.image_size)
        self.output_dim = self.num_classes

        character_folders = [os.path.join(data_folder, family, character)
                             for family in os.listdir(data_folder)
                             if os.path.isdir(os.path.join(data_folder, family))
                             for character in os.listdir(os.path.join(data_folder, family))
                             if os.path.isdir(os.path.join(data_folder, family, character))]

        random.seed(1)
        random.shuffle(character_folders)

        meta_train = 1100
        meta_val = 100

        self.meta_train_characters = character_folders[: meta_train]
        self.meta_val_characters = character_folders[meta_train: meta_train + meta_val]
        self.meta_test_characters = character_folders[meta_train + meta_val:]

    def sample_batch(self, batch_type, batch_size):
        """
        Samples a batch for training, validation, or testing
        Args:
            batch_type: train/val/test
        Returns:
            A a tuple of (1) Image batch and (2) Label batch where
            image batch has shape [B, K, N, 784] and label batch has shape [B, K, N, N]
            where B is batch size, K is number of samples per class, N is number of classes
        """

        if batch_type == "train":
            folders = self.meta_train_characters
        if batch_type == "test":
            folders = self.meta_test_characters
        if batch_type == "val":
            folders = self.meta_val_characters

        all_image_batches = []
        all_label_batches = []

        for batch_idx in range(batch_size):
            # 1. Sample N from folders
            sample_classes = random.sample(folders, self.num_classes)

            # 2. Load K images of each N classes -> Total K x N images
            one_hot_labels = np.identity(self.num_classes)
            labels_images = get_images(sample_classes, one_hot_labels,
                                       nb_samples=self.num_samples_per_class,
                                       shuffle=False)

            train_images, train_labels = [], []
            test_images, test_labels = [], []

            for sample_idx, (labels, images) in enumerate(labels_images):
                # Take the first image of each class (index is 0, N, 2N...) to test_set
                angles = np.random.randint(0, 4, 1)[0] * 90 + (np.random.rand()-0.5)*22.5
                trans = np.random.randint(-10, 11, size=2).tolist()
                if sample_idx % self.num_samples_per_class == 0:
                    test_images.append(image_transform(images, angle=angles, trans=trans, size=self.image_size))
                    test_labels.append(labels)
                else:
                    train_images.append(image_transform(images, angle=angles, trans=trans, size=self.image_size))
                    train_labels.append(labels)

            train_images, train_labels = shuffle(train_images, train_labels)
            test_images, test_labels = shuffle(test_images, test_labels)

            labels = np.vstack(train_labels + test_labels).reshape((-1, self.num_classes, self.num_classes))  # K, N, N
            images = np.vstack(train_images + test_images).reshape(
                (self.num_samples_per_class, self.num_classes, -1))  # K x N x 784

            all_label_batches.append(labels)
            all_image_batches.append(images)

        # 3. Return two numpy array (B, K, N, 784) and one-hot labels (B, K, N, N)
        all_image_batches = np.stack(all_image_batches).astype(np.float32).reshape(batch_size, -1, np.prod(self.image_size))  # Batch x (K x N) x 784
        all_label_batches = np.stack(all_label_batches).astype(np.float32).reshape(batch_size, -1, self.num_classes)  # Batch x (K x N) x 5

        return all_image_batches, all_label_batches


if __name__ == "__main__":
    N, K, B = 5, 3, 4

    ordered_generator = OmniglotGenerator('../datasets/Omniglot/omniglot_resized', N, K + 1, (28, 28))
    batch_imgs, batch_labels = ordered_generator.sample_batch('train', B)
    batch_imgs = batch_imgs.reshape(B, K + 1, N, 28*28)
    batch_labels = batch_labels.reshape(B, K + 1, N, N)

    images, labels = batch_imgs[0], batch_labels[0]
    train_images, train_labels = images[:-1].reshape(-1, 28, 28), labels[:-1].reshape(-1, N)  # K-1, N, N
    test_images, test_labels = images[-1:].reshape(-1, 28, 28), labels[-1:].reshape(-1, N)  # 1, N, N

    plot_images(train_images, train_labels, n_col=N, n_row=K+1, title=f'Train #{1}')
    plot_images(test_images, test_labels, n_col=N, n_row=K+1, title=f'Test #{1}')