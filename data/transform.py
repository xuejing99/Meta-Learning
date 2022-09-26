import torchvision.transforms as transforms
from core.workspace import Registers


@Registers.transform.register
def resize(img_size):
    return transforms.Resize(img_size)


@Registers.transform.register
def toTensor(params):
    return transforms.ToTensor()


@Registers.transform.register
def normalize(params):
    return transforms.Normalize(**params)
