import numpy as np
import torch
import torch.nn as nn


class MANNCell(nn.Module):
    def __init__(self, lstm_size, memory_size, memory_dim,
                 nb_reads, input_size, gamma = 0.95):
        super().__init__()
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.nb_reads = nb_reads  # #(read head) == #(write head)
        self.input_size = input_size
        self.gamma = gamma

        def initialize_weights(model):
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)

        self.controller = nn.LSTM(input_size=self.input_size,
                                  hidden_size=self.lstm_size,
                                  batch_first=True)
        initialize_weights(self.controller)

        self.num_parameters_per_head = self.memory_dim + 1
        self.total_parameter_num = self.num_parameters_per_head * self.nb_reads

        self.controller_output_to_params = nn.Linear(in_features=self.lstm_size,
                                                     out_features=self.total_parameter_num,
                                                     bias=True)
        nn.init.xavier_uniform_(self.controller_output_to_params.weight)
        nn.init.zeros_(self.controller_output_to_params.bias)

    def get_initial_state(self, batch_size):
        one_hot_weight_vector = np.zeros([batch_size, self.memory_size])
        one_hot_weight_vector[..., 0] = 1
        one_hot_weight_vector = torch.Tensor(one_hot_weight_vector)
        h_0 = torch.zeros([1, self.lstm_size])
        c_0 = torch.zeros([1, self.lstm_size])
        initial_state = {
            'controller_state': (h_0, c_0),
            'read_vector_list': [torch.zeros([batch_size, self.memory_dim])
                                 for _ in range(self.nb_reads)],
            'w_r_list': [one_hot_weight_vector for _ in range(self.nb_reads)],
            'w_u': one_hot_weight_vector,
            'M': torch.Tensor(np.ones([batch_size, self.memory_size, self.memory_dim]) * 1e-6)
        }
        return initial_state

    def _least_used(self, w_u):
        _, indices = torch.topk(w_u, k=self.memory_size)
        w_lu = torch.sum(torch.nn.functional.one_hot(indices[:, -self.nb_reads:],
                                                     num_classes=self.memory_size), axis=1)
        return indices, w_lu

    def _read_head_addressing(self, k, prev_M):
        # Cosine Similarity
        k = k.unsqueeze(dim=2)
        inner_product = torch.matmul(prev_M, k)
        k_norm = torch.sqrt(torch.sum(torch.square(k), axis=1, keepdim=True))
        M_norm = torch.sqrt(torch.sum(torch.square(prev_M), axis=2, keepdim=True))
        norm_product = M_norm * k_norm
        K = torch.squeeze(inner_product / (norm_product + 1e-8))

        K_exp = torch.exp(K)
        w = K_exp / torch.sum(K_exp, axis=1, keepdim=True)  # eq (18)

        return w

    def _write_head_addressing(self, sig_alpha, prev_w_r, prev_w_lu):
        return sig_alpha * prev_w_r + (1. - sig_alpha) * prev_w_lu  # eq (22)

    def forward(self, inputs, states):
        # import pdb
        # pdb.set_trace()
        prev_read_vector_list = states['read_vector_list']
        prev_controller_state = states['controller_state']
        controller_input = torch.concat([inputs] + prev_read_vector_list, axis=1)
        controller_output, controller_state = self.controller(controller_input,
                                                              prev_controller_state)
        parameters = self.controller_output_to_params(controller_output)
        head_parameter_list = torch.chunk(parameters, self.nb_reads, dim=1)

        prev_w_r_list = states['w_r_list']
        prev_M = states['M']
        prev_w_u = states['w_u']

        prev_indices, prev_w_lu = self._least_used(prev_w_u)
        w_r_list = []
        w_w_list = []
        k_list = []

        for i, head_parameter in enumerate(head_parameter_list):
            k = torch.tanh(head_parameter[:, 0:self.memory_dim])
            sig_alpha = torch.sigmoid(head_parameter[:, -1:])

            w_r = self._read_head_addressing(k, prev_M)
            w_w = self._write_head_addressing(sig_alpha.detach().numpy(),
                                              prev_w_r_list[i].detach().numpy(),
                                              prev_w_lu.detach().numpy())

            w_r_list.append(w_r)
            w_w_list.append(w_w)
            k_list.append(k)

        w_u = self.gamma * prev_w_u + sum(w_r_list) + sum(torch.Tensor(np.array(w_w_list)))
        M_ = prev_M * (1. - torch.nn.functional.one_hot(prev_indices[:, -1],
                                                        self.memory_size)).unsqueeze(dim=2)

        M = M_
        for i in range(self.nb_reads):
            w = torch.Tensor(w_w_list[i]).unsqueeze(dim=2)
            k = k_list[i].unsqueeze(dim=1)
            M = M + torch.matmul(w, k)

        read_vector_list = []
        for i in range(self.nb_reads):
            read_vector = torch.sum(w_r_list[i].unsqueeze(dim=2) * M, axis=1)
            read_vector_list.append(read_vector)

        mann_output = torch.concat([controller_output] + read_vector_list, axis=1)

        state = {
            'controller_state': controller_state,
            'read_vector_list': read_vector_list,
            'w_r_list': w_r_list,
            'w_w_list': w_w_list,
            'w_u': w_u,
            'M': M,
        }

        return mann_output, state
