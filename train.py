import argparse
import yaml
import data
from tqdm import tqdm
from torch.utils.data import DataLoader

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
    # model = load_model()

    # 3. load dataset
    train_dataset = Registers.dataset["MetaMiniImageNet"](
        root_path=config['TrainDataset']['dataset_dir'],
        train_file_path=config['TrainDataset']['image_anno_dir'],
        mode="train",
        preprocess=config['TrainDataset']['preprocess'],
        n_batch=config['TrainConfig']['batch_size'],
        n_episode=config['TrainConfig']['episode'],
        n_way=config['TrainConfig']['n_way'],
        n_shot=config['TrainConfig']['k_shot'],
        n_query=config['TrainConfig']['query_number'])
    train_loader = DataLoader(train_dataset, config['TrainConfig']['episode'],
                              collate_fn=train_dataset.collate_fn,
                              num_workers=0, pin_memory=True)
    # val_loader = DataLoader()

    # 4. train the model
    start_epoch = 0
    for epoch in range(start_epoch, 10):
        # model.train()
        for data in tqdm(train_loader, desc='meta-train', leave=False):
            x_shot, x_query, y_shot, y_query = data
            train_step()
            print(x_shot, x_query, y_shot, y_query)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='configuration file',
                        default="./configs/MAML/maml_convnet4_miniImagenet.yaml")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = parse_opt()
    configs = yaml.load(open(opt.config, 'r'), Loader=yaml.FullLoader)

    main(configs)

