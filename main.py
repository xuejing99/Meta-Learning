import datetime
from argparse import ArgumentParser
import numpy as np
import torch
import os

from models.MANN.MANN import MANN
from data.DataLoader import OmniglotGenerator


def build_argparser():
    parser = ArgumentParser()

    parser.add_argument('--mode', default="train")
    parser.add_argument('--restore_training', default=False)
    parser.add_argument('--batch-size',
                        dest='batch_size', help='Batch size (default: %(default)s)',
                        type=int, default=16)
    parser.add_argument('--num-classes',
                        dest='nb_classes', help='Number of classes in each episode (default: %(default)s)',
                        type=int, default=5)
    parser.add_argument('--num-samples',
                        dest='nb_samples_per_class',
                        help='Number of total samples in each episode (default: %(default)s)',
                        type=int, default=10)
    parser.add_argument('--input-height',
                        dest='input_height', help='Input image height (default: %(default)s)',
                        type=int, default=20)
    parser.add_argument('--input-width',
                        dest='input_width', help='Input image width (default: %(default)s)',
                        type=int, default=20)
    parser.add_argument('--num-reads',
                        dest='nb_reads', help='Number of read heads (default: %(default)s)',
                        type=int, default=4)
    parser.add_argument('--controller-size',
                        dest='controller_size', help='Number of hidden units in controller (default: %(default)s)',
                        type=int, default=200)
    parser.add_argument('--memory-locations',
                        dest='memory_locations', help='Number of locations in the memory (default: %(default)s)',
                        type=int, default=128)
    parser.add_argument('--memory-word-size',
                        dest='memory_word_size', help='Size of each word in memory (default: %(default)s)',
                        type=int, default=40)
    parser.add_argument('--num_layers',
                        dest='num_layers', help='Size of each word in memory (default: %(default)s)',
                        type=int, default=1)
    parser.add_argument('--learning-rate',
                        dest='learning_rate', help='Learning Rate (default: %(default)s)',
                        type=float, default=1e-3)
    parser.add_argument('--start_iterations',
                        dest='start_iterations', default=0)
    parser.add_argument('--iterations',
                        dest='iterations', help='Number of iterations for training (default: %(default)s)',
                        type=int, default=100000)
    parser.add_argument('--augment', default=True)
    parser.add_argument('--save-dir', default='./ckpt/')
    parser.add_argument("--log-dir", default="./log/")
    parser.add_argument('--model', default="MANN", help='LSTM or MANN')
    parser.add_argument('--weights', default="ckpt/MANN/model5000.pt", help='Path of trained model')
    parser.add_argument('--reuse', default=False, help='Reuse the pre-train model')

    return parser


def metric_accuracy(args, labels, outputs):
    seq_length = args.nb_classes * args.nb_samples_per_class
    outputs = outputs.reshape(-1, seq_length, args.nb_classes)
    labels = labels.reshape(-1, seq_length, args.nb_classes)
    labels = np.argmax(labels, axis=-1)
    outputs = np.argmax(outputs, axis=-1)
    correct = [0] * seq_length
    total = [0] * seq_length
    for i in range(np.shape(labels)[0]):
        label = labels[i]
        output = outputs[i]
        class_count = {}
        for j in range(seq_length):
            class_count[label[j]] = class_count.get(label[j], 0) + 1
            total[class_count[label[j]]-1] += 1
            if label[j] == output[j]:
                correct[class_count[label[j]]-1] += 1
    return [float(correct[i]) / total[i] if total[i] > 0. else 0. for i in range(args.nb_samples_per_class)]


def train(model: MANN, data_genarator: OmniglotGenerator, args):
    start_iter = args.start_iterations
    max_iter = args.iterations
    csv_write_path = '{}/{}-{}-{}--{}.csv'.format(
        args.log_dir,
        args.model,
        args.nb_classes,
        args.nb_samples_per_class,
        datetime.datetime.now().strftime('%m-%d-%H-%M'))

    print(args)
    print("1st\t2nd\t3rd\t4th\t5th\t6th\t7th\t8th\t9th\t10th\tbatch\tloss")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for step in range(start_iter, max_iter):
        model.train()
        image, label = data_genarator.sample_batch("train", args.batch_size)
        optimizer.zero_grad()
        _, loss, acc = model((image, label))
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            model.eval()
            image, label = data_genarator.sample_batch("val", args.batch_size)
            output, loss, acc = model((image, label))
            accuracy = metric_accuracy(args, label, output.detach().numpy())
            for accu in accuracy:
                print('%.4f' % accu, end='\t')
            print('%d\t%.4f' % (step, loss))

            with open(csv_write_path, 'a') as fh:
                fh.write(str(step) + ", " + ", ".join(['%.4f' % accu for accu in accuracy]+['%.4f' %acc,'%.4f' %loss.item()]) +"\n")

            if step % 5000 == 0 and step > 0:
                torch.save(model.state_dict(),
                           os.path.join(args.save_dir, args.model) + rf'/model{step}.pt')


def test(model: MANN, data_generator: OmniglotGenerator, args):
    print("Test Result\n1st\t2nd\t3rd\t4th\t5th\t6th\t7th\t8th\t9th\t10th\tloss")
    label_list = []
    output_list = []
    loss_list = []
    model.eval()
    for ep in range(20):
        image, label = data_generator.sample_batch("test", args.batch_size)
        output, loss, _ = model((image, label))
        label_list.append(label)
        output_list.append(output.detach().numpy())
        loss_list.append(loss.detach().numpy())
    accuracy = metric_accuracy(args, np.concatenate(label_list, axis=0), np.concatenate(output_list, axis=0))
    for accu in accuracy:
        print('%.4f' % accu, end='\t')
    print(np.mean(loss_list))


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()

    batch_size = args.batch_size
    nb_classes = args.nb_classes
    nb_samples_per_class = args.nb_samples_per_class
    img_size = (args.input_height, args.input_width)
    input_size = args.input_height * args.input_width

    nb_reads = args.nb_reads
    controller_size = args.controller_size
    memory_size = args.memory_locations
    memory_dim = args.memory_word_size
    num_layers = args.num_layers

    learning_rate = args.learning_rate

    model = MANN(learning_rate, input_size, memory_size, memory_dim,
                 controller_size, nb_reads, num_layers, nb_classes,
                 nb_samples_per_class, batch_size, args.model)

    data_generator = OmniglotGenerator(
        data_folder="./datasets/Omniglot/omniglot_resized",
        num_classes=nb_classes,
        num_samples_per_class=nb_samples_per_class,
        image_size=img_size)

    if args.reuse or args.mode == "test":
        model.load_state_dict(torch.load(args.weights))

    if args.mode == "train":
        train(model, data_generator, args)
    elif args.mode == "test":
        test(model, data_generator, args)
