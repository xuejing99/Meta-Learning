import numpy as np
import torch
import torch.nn as nn
from .MANNCell import MANNCell


class MANN(nn.Module):
    def __init__(self, learning_rate=1e-3, input_size=28 * 28,
                 memory_size=128, memory_dim=40, controller_size=200,
                 nb_reads=4, num_layers=1, nb_classes=5,
                 nb_samples_per_class=10, batch_size=16, model="MANN"):

        super().__init__()
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.controller_size = controller_size
        self.num_layers = num_layers
        self.nb_reads = nb_reads
        self.nb_classes = nb_classes
        self.nb_samples_per_class = nb_samples_per_class
        self.batch_size = batch_size
        self.model = model

        def weight_init(m):
            if isinstance(m, nn.Linear):
                # torch.nn.init.xavier_uniform_(m.weight)
                nn.init.uniform_(m.weight, a=-0.1, b=0.1)
                nn.init.uniform_(m.bias, a=-0.1, b=0.1)
                # nn.init.zeros_(m.bias)

        self.cell = MANNCell(lstm_size=self.controller_size,
                             memory_size=self.memory_size,
                             memory_dim=self.memory_dim,
                             nb_reads=self.nb_reads,
                             input_size=self.input_size + self.memory_size * self.nb_reads + self.nb_classes)
        self.classifier = nn.Linear(in_features=self.controller_size + self.memory_dim * self.nb_reads,
                                    out_features=self.nb_classes,
                                    bias=True)
        self.classifier.apply(weight_init)
        self.state = self.cell.get_initial_state(self.batch_size)

    def forward(self, inputs):
        # import pdb
        # pdb.set_trace()
        images, labels = inputs  # [batch_size, K * N, 784]
        offset_target = np.concatenate([np.expand_dims(np.zeros_like(labels[:, 0]), axis=1),
                                        labels[:, :-1]], axis=1)

        # state = self.cell.get_initial_state(self.batch_size)
        inputs = torch.cat([torch.Tensor(images), torch.Tensor(offset_target)], axis=2)
        out = []
        state = self.state
        for i in range(inputs.shape[1]):
            output, state = self.cell(inputs[:, i], state)
            # pdb.set_trace()
            batch_y_pre = self.classifier(output)
            out.append(batch_y_pre)

        self.state = {
            "M": state['M'].detach(),
            "read_vector": state['read_vector'].detach(),
            # "controller_state": (torch.zeros([1, self.controller_size]), torch.zeros([1, self.controller_size])),
            "controller_state": (state['controller_state'][0].detach(), state['controller_state'][1].detach()),
            "wu": state['wu'].detach(),
            "wr": state['wr'].detach(),
        }

        out = torch.stack(out, axis=1)
        softmax = torch.nn.Softmax(2)
        out = softmax(out)
        pred = out.argmax(axis=2).flatten()
        target = torch.Tensor(labels.argmax(axis=-1).flatten())

        out = out.reshape(-1, self.nb_classes)
        labels = torch.Tensor(labels.reshape(-1, self.nb_classes))

        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, labels)
        acc = (1.0 * (pred == target)).mean().item()

        return out, loss, acc
