import numpy as np
import torch
import torch.nn as nn
from .utils import variable_one_hot


class MANNCell(nn.Module):
    def __init__(self, lstm_size, memory_size, memory_dim,
                 nb_reads, input_size, gamma=0.95):
        super().__init__()
        self.lstm_size = lstm_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.nb_reads = nb_reads  # #(read head) == #(write head)
        self.input_size = input_size
        self.gamma = gamma

        self.controller = nn.LSTM(input_size=self.input_size,
                                  hidden_size=self.lstm_size,
                                  batch_first=True)

        parameter_dim_per_head = self.memory_dim * 2 + 1
        parameter_total_dim = parameter_dim_per_head * self.nb_reads  # []
        self.controller2parameter = nn.Linear(in_features=self.lstm_size,
                                              out_features=parameter_total_dim,
                                              bias=True)

        def initialize_weights(model):
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)

        initialize_weights(self.controller)
        nn.init.xavier_uniform_(self.controller2parameter.weight)
        # nn.init.zeros_(self.controller2parameter.bias)
        nn.init.uniform_(self.controller2parameter.bias, a=-0.1, b=0.1)

    def forward(self, input, prev_state):
        M_prev, r_prev, controller_state_prev, wu_prev, wr_prev = \
            prev_state["M"], prev_state["read_vector"], \
            prev_state["controller_state"], \
            prev_state["wu"], prev_state["wr"]

        controller_input = torch.cat([input, wr_prev], -1)
        controller_hidden_t, controller_state_t = self.controller(controller_input, controller_state_prev)

        parameter = self.controller2parameter(controller_hidden_t)

        indices_prev, wlu_prev = self.least_used(wu_prev)

        k = torch.tanh(parameter[:, 0:self.nb_reads * self.memory_dim])
        a = torch.tanh(parameter[:, self.nb_reads * self.memory_dim: 2 * self.nb_reads * self.memory_dim])
        sig_alpha = torch.sigmoid(parameter[:, -self.nb_reads:])

        wr_t = self.read_head_addressing(k, M_prev)
        ww_t = self.write_head_addressing(sig_alpha, wr_prev, wlu_prev)

        wu_t = self.gamma * wu_prev + torch.sum(wr_t, axis=1) + torch.sum(ww_t, axis=1)

        # "Prior to writing to memory, the least used memory location set to zero"
        M_t = M_prev * (1. - torch.nn.functional.one_hot(indices_prev[:, -1], self.memory_size)).unsqueeze(2)
        M_t = M_t + torch.matmul(torch.transpose(ww_t, 1, 2), a.reshape(-1, self.nb_reads, self.memory_dim))
        r_t = torch.matmul(wr_t, M_t).reshape(-1, self.nb_reads * self.memory_dim)

        state = {
            "M": M_t,
            "read_vector": r_t,
            "controller_state": controller_state_t,
            "wu": wu_t,
            "wr": wr_t.reshape(-1, self.nb_reads * self.memory_size),
        }

        NTM_output = torch.cat([controller_hidden_t, r_t], -1)

        return NTM_output, state

    def read_head_addressing(self, k, M_prev, eps=1e-8):
        k = k.reshape(-1, self.nb_reads, self.memory_dim)
        inner_product = torch.matmul(k, torch.transpose(M_prev, 2, 1))
        k_norm = torch.sqrt(torch.sum(torch.square(k), 2).unsqueeze(2))
        M_norm = torch.sqrt(torch.sum(torch.square(M_prev), 2).unsqueeze(1))
        norm_product = k_norm * M_norm
        K = inner_product / (norm_product + eps)
        softmax = torch.nn.Softmax(-1)
        return softmax(K)

    def write_head_addressing(self, sig_alpha, wr_prev, wlu_prev):
        sig_alpha = sig_alpha.unsqueeze(-1)
        wr_prev = wr_prev.reshape(-1, self.nb_reads, self.memory_size)
        return sig_alpha * wr_prev + (1. - sig_alpha) * wlu_prev.unsqueeze(1)

    def least_used(self, w_u):
        _, indices = torch.topk(w_u, k=self.memory_size)
        wlu = indices[:, -self.nb_reads:]
        wlu = torch.sum(torch.nn.functional.one_hot(wlu, self.memory_size), axis=1)
        return indices, wlu

    def get_initial_state(self, batch_size):
        M_0 = torch.Tensor(np.ones([batch_size, self.memory_size, self.memory_dim]) * 1e-6)
        r_0 = torch.zeros([batch_size, self.memory_dim * self.nb_reads])
        controller_state_0 = (torch.zeros([1, self.lstm_size]), torch.zeros([1, self.lstm_size]))
        wu_0 = torch.Tensor(variable_one_hot([batch_size, self.memory_size]))
        wr_0 = torch.Tensor(variable_one_hot([batch_size, self.memory_size * self.nb_reads]))

        initial_state = {
            "M": M_0,
            "read_vector": r_0,
            "controller_state": controller_state_0,
            "wu": wu_0,
            "wr": wr_0,
        }
        return initial_state
