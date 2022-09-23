import argparse
import yaml
import data
from torch.utils.data import Dataset

from core.workspace import Registers


def train_step():
    pass  # todo


def model_save():
    pass  # todo


def load_model():
    pass  # todo
    return model


def main(config):
    # 1. parameter initialization
    print(config)

    # 2. construct model
    model = load_model()

    # 3. load dataset
    train_loader = DataLoader()
    val_loader = DataLoader

    # 4. train the model
    for epoch in range(start_epoch, config.epoch + 1):
        model.train()
        for data in tqdm(train_loader, desc='meta-train', leave=False):
            x_shot, x_query, y_shot, y_query = data
            train_step()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='configuration file',
                        default="./configs/MAML/maml_convnet4_miniImagenet.yaml")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = parse_opt()
    configs = yaml.load(open(opt.config, 'r'), Loader=yaml.FullLoader)
    dataset = Registers.dataset["MiniImageNet"](configs, "train")
    print(len(dataset))
    # main(configs)

