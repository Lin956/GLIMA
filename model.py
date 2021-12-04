import torch.nn as nn
import torch
import torch.nn.functional as F
from  modules import *


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, q=64, heads=8):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = GMGRU(input_size, hidden_size, q)
        self.attn1 = MultiHeadAttention(hidden_size, heads)
        self.attn2 = MultiHeadAttention(hidden_size, heads)

        # concat:eq.(14)
        self.w_tf = nn.Linear(2*hidden_size, hidden_size)

        # MLP:
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(2 * hidden_size, 1)
        )

        # finally estimation
        self.w_h = nn.Linear(input_size*hidden_size, input_size, bias=False)
        self.w_o = nn.Linear(input_size*hidden_size, input_size, bias=False)

    def forward(self, data):
        """
        RNN input:data
        data: [3, 33, 49]
        X: values
        M: Missing markers
        Delta；time-stamp
        H: [T, D, hidden_size]
        x_rnn_pred: [D, T]
        """
        H, x_rnn_pred = self.gru(data)
        x_rnn_pred = x_rnn_pred.transpose(0, 1)  # [T, D]
        # print(x_rnn_pred)

        # Cross-time
        O_t = self.attn1(H.transpose(0, 1))  # [D, T, hidden_size]
        O_t = O_t.transpose(0, 1)  # [T, D, hidden_size]

        # Cross-feature
        O_f = self.attn2(H)

        # concat:eq.(14)
        O_tf = self.w_tf(torch.cat((O_t, O_f), dim=-1))  # [T, D, hidden_size]

        # MLP
        x_attn_pred = self.fc(O_tf).reshape(-1, self.input_size)
        # print(x_attn_pred.shape)

        H = H.reshape(-1, self.input_size*self.hidden_size)
        O_tf = O_tf.reshape(-1, self.input_size*self.hidden_size)

        alpha1 = torch.softmax(self.w_h(H)/(self.w_h(H) + self.w_o(O_tf)), dim=-1)
        alpha2 = torch.softmax(self.w_o(O_tf)/(self.w_h(H) + self.w_o(O_tf)), dim=-1)
        # print(alpha2.shape)
        # print(alpha2)

        x_t = alpha1 * x_rnn_pred + alpha2 * x_attn_pred
        # print(x_t)

        return x_rnn_pred, x_attn_pred, x_t


"""
x = torch.rand(3, 33, 49)
model = Model(input_size=33, hidden_size=64, q=64, heads=8)
y_1, y_2, y = model(x)
print(y.shape)
"""








