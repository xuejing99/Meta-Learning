import torchvision.transforms as transforms


def get_transform(operations, img_size):
    op_list = [transforms.Resize(img_size)]
    # for op in op_list:
    #     if op == ""
    op_list.append(transforms.ToTensor())
    op_list.append(transforms.Normalize(operations))
