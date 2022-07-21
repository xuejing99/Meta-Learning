import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageOps, Image


def get_images(paths, labels, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of category folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per category
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x

    labels_images = [(i, os.path.join(path, image)) \
                     for i, path in zip(labels, paths) \
                     for image in sampler(os.listdir(path))]

    if shuffle is None:
        random.shuffle(labels_images)

    return labels_images


def image_file_to_array(filename, dim_input):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename
        dim_input: Flattened shape of image
    Returns:
        1 channel image
    """
    image = np.asarray(plt.imread(filename))
    # plt.imshow(1 - image, cmap='gray')
    image = np.reshape(image, [dim_input])
    image = (1. - image.astype(np.float32)) / 255.

    return image


def plot_images(imgs, labels, n_col, n_row, title=None):
    plt.figure(figsize=(8, n_row+1))
    n_row = np.ceil(len(imgs) / n_col).astype(int)
    for img_idx, (img, label) in enumerate(zip(imgs, labels)):
        plt.subplot(n_row, n_col, img_idx + 1)
        plt.imshow(img, cmap='gray')
        plt.title(label.argmax())

    if title:
        plt.suptitle(title)
    plt.show()


def image_transform(filename, angle=0., trans=(0., 0.), size=(20, 20)):
    image = Image.open(filename)
    image = ImageOps.invert(image.convert("L")).rotate(angle, translate=trans).resize(size)
    np_image = np.reshape(np.array(image, dtype=np.float32), newshape=(np.prod(size)))
    max_value = np.max(np_image)
    if max_value > 0.:
        np_image = np_image / max_value
    return np_image
