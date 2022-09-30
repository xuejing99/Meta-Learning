import os
import shutil
import argparse
import yaml
import data
import models
import utils
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import logging

from core.workspace import Registers


def initial_setting(config):
    ckpt_name = config['project_name']
    ckpt_path = os.path.join("./runs", ckpt_name)
    if os.path.exists(ckpt_path):
        shutil.rmtree(ckpt_path)
        os.makedirs(ckpt_path)
    writer = SummaryWriter(os.path.join(ckpt_path, 'tensorboard'))
    yaml.dump(config, open(os.path.join(ckpt_path, 'config.yaml'), 'w'))
    handler = logging.FileHandler(os.path.join(ckpt_path, "log.txt"), encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s  %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return writer, ckpt_path


def load_model(config):
    model_name = config['name']
    model = Registers.model[model_name](
        config["encoder"], config["classifier"],
        config.get("encoder_config"), config.get("classifier_config"))
    return model


def load_dataset(dataset_name, config, mode, batch_size,
                 episode, n_way, k_shot, query_number, ):
    dataset = Registers.dataset[dataset_name](
        root_path=config['dataset_dir'],
        file_path=config['image_anno_dir'],
        mode=mode, preprocess=config['preprocess'])
        # n_batch=batch_size, n_episode=episode,
        # n_way=n_way, n_shot=k_shot, n_query=query_number)
    loader = DataLoader(dataset, episode,
                        collate_fn=dataset.collate_fn,
                        num_workers=0, pin_memory=True)
    logger.info('meta-{} set: {} (x{}), {}'.format(
        mode, dataset[0][0].shape, len(dataset), dataset.n_classes))
    return loader


def train_step(model, x_shot, x_query, y_shot, y_query, config, optimizer):
    logits = model(x_shot, x_query, y_shot, config['TrainConfig']['inner_args'], meta_train=True)
    logits = logits.flatten(0, 1)
    labels = y_query.flatten()
    pred = torch.argmax(logits, dim=-1)
    acc = utils.metrics.compute_acc(pred, labels)
    loss = F.cross_entropy(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    for param in optimizer.param_groups[0]['params']:
        nn.utils.clip_grad_value_(param, 10)
    optimizer.step()
    return loss, acc


def eval_model(model, x_shot, x_query, y_shot, y_query, config):
    logits = model(x_shot, x_query, y_shot, config['TrainConfig']['inner_args'], meta_train=False)
    logits = logits.flatten(0, 1)
    labels = y_query.flatten()

    pred = torch.argmax(logits, dim=-1)
    acc = utils.metrics.compute_acc(pred, labels)
    loss = F.cross_entropy(logits, labels)
    return loss, acc


def main(config):
    # initial setting
    writer, ckpt_path = initial_setting(config['TrainConfig'])
    device = torch.device("cuda:0" if config['device'] == 'cuda' else "cpu")
    aves_keys = ['train-loss', 'train-acc', 'val-loss', 'val-acc']
    
    # construct the model
    model = load_model(config['Model']).to(device)
    logger.info('num params: {}'.format(Registers.metric['params'](model)))

    # load datasets
    train_loader = load_dataset(config['dataset'], config['TrainDataset'],
                                "train", **config['TrainConfig']['outer_args'])
    val_loader = load_dataset(config['dataset'], config['EvalDataset'],
                              "val", **config['TrainConfig']['outer_args'])

    # 4. train the model
    start_epoch = 1
    optimizer = Registers.optimizer[config['TrainConfig']['optimizer']["method"]](
        model.parameters(),
        **config['TrainConfig']['optimizer']["args"])

    for epoch in range(start_epoch, config['TrainConfig']['epoch'] + 1):
        aves = {k: utils.metrics.AverageMeter() for k in aves_keys}
        model.train()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        for data in tqdm(train_loader, desc='meta-train', leave=False):
            x_shot, x_query, y_shot, y_query = data
            x_shot, y_shot = x_shot.to(device), y_shot.to(device)
            x_query, y_query = x_query.to(device), y_query.to(device)
            loss, acc = train_step(model, x_shot, x_query, y_shot, y_query, config, optimizer)
            aves['train-loss'].update(loss.item(), 1)
            aves['train-acc'].update(acc, 1)

        model.eval()
        for data in tqdm(val_loader, desc='meta-val', leave=False):
            x_shot, x_query, y_shot, y_query = data
            x_shot, y_shot = x_shot.to(device), y_shot.to(device)
            x_query, y_query = x_query.to(device), y_query.to(device)
            loss, acc = eval_model(model, x_shot, x_query, y_shot, y_query, config)
            aves['val-loss'].update(loss.item(), 1)
            aves['val-acc'].update(acc, 1)

        for k, avg in aves.items():
            aves[k] = avg.item()
            
        logger.info('epoch {}, meta-train {:.4f}|{:.4f}  |  meta-val {:.4f}|{:.4f}'
                    .format(str(epoch), aves['train-loss'], aves['train-acc'],
                            aves['val-loss'], aves['val-acc']))
        writer.add_scalars('loss', {'meta-train': aves['train-loss']}, epoch)
        writer.add_scalars('acc', {'meta-train': aves['train-acc']}, epoch)
        writer.add_scalars('loss', {'meta-val': aves['val-loss']}, epoch)
        writer.add_scalars('acc', {'meta-val': aves['val-acc']}, epoch)

        torch.save(model, os.path.join(ckpt_path, 'epoch-last.pth'))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='configuration file',
                        default="./configs/MAML/maml_convnet4_omniglot.yaml")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = parse_opt()
    configs = yaml.load(open(opt.config, 'r'), Loader=yaml.FullLoader)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
    logger = logging.getLogger(__name__)

    main(configs)

