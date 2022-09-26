import torchvision.transforms as transforms

from core.workspace import Registers


def get_transform(operations):
    op_list = list()
    for op in operations.keys():
        op_list.append(Registers.transform[op](operations[op]))
    return transforms.Compose(op_list)
