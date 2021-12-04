import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from layers import *

"""
x = torch.Tensor(33, 49)
print(x)
"""
class tMGRU(nn.Module):
    def __init__(self, hidden_size):
        # input_size = 49  time
        super(tMGRU, self).__init__()

        # weights
        self.W_z = nn.Parameter(torch.randn(33, hidden_size, hidden_size), requires_grad=True)
        self.U_zx = nn.Parameter(torch.randn(33, hidden_size, 1, ), requires_grad=True)
        self.U_zh = nn.Parameter(torch.randn(33, hidden_size, hidden_size), requires_grad=True)

        self.W_r = nn.Parameter(torch.randn(33, hidden_size, hidden_size), requires_grad=True)
        # print(torch.randn(hidden_size, hidden_size))
        self.U_rx = nn.Parameter(torch.randn(33, hidden_size, 1), requires_grad=True)
        self.U_rh = nn.Parameter(torch.randn(33, hidden_size, hidden_size), requires_grad=True)

        self.W_h = nn.Parameter(torch.randn(33, hidden_size, hidden_size), requires_grad=True)
        self.U_hx = nn.Parameter(torch.randn(33, hidden_size, 1), requires_grad=True)
        self.U_hh = nn.Parameter(torch.randn(33, hidden_size, hidden_size), requires_grad=True)

        # bias
        self.b_z = nn.Parameter(torch.randn(hidden_size, 1), requires_grad=True)
        self.b_r = nn.Parameter(torch.randn(hidden_size, 1), requires_grad=True)
        self.b_h = nn.Parameter(torch.randn(hidden_size, 1), requires_grad=True)

    def forward(self, H, x, h):
        """
        X = torch.squeeze(data[0])  # .size = (33,49)
        Mask = torch.squeeze(data[1])  # .size = (33,49)
        Delta = torch.squeeze(data[2])  # .size = (33,49)

        # eq.(9)
        gamma_st = torch.exp(-torch.max(self.zeros, self.gamme_st(Delta)))  # eq.(1) [D, hidden_size], D是变量数量

        # initial hidden state matrix
        Hidden_state = Variable(torch.zeros(self.D, self.hidden_size))  # [D, hidden_size]
        H = Hidden_state

        gamma_h = torch.exp(-torch.max(self.zeros, self.gamma_st(d)))  # eq.(1) [hidden_size]

        x_st = self.x_st(H)  # [D, 1]
        """
        """
        H: [D, p_s] -> [D,p_s, 1]
        x: [D] -> [D, 1, 1]
        h: [p_s] -> [p_S, 1]
        p_s: hidden_size
        """
        H_d, H_p = H.shape
        # 更新门
        z_t = torch.sigmoid(torch.matmul(self.W_z, H.reshape(H_d, H_p, 1)) +\
                            torch.matmul(self.U_zx, x.reshape(H_d, 1, 1)) +\
                            torch.matmul(self.U_zh, h.reshape(H_p, 1)) + self.b_z)
        # 重置门
        r_t = torch.sigmoid(torch.matmul(self.W_r, H.reshape(H_d, H_p, 1)) +\
                            torch.matmul(self.U_rx, x.reshape(H_d, 1, 1)) +\
                            torch.matmul(self.U_rh, h.reshape(H_p, 1)) + self.b_r)   # [D, p_s, 1]

        # 隐藏状态
        H_t_hat = torch.tanh(r_t * torch.matmul(self.W_h, H.reshape(H_d, H_p, 1)) +\
                            torch.matmul(self.U_hx, x.reshape(H_d, 1, 1)) +\
                            torch.matmul(self.U_hh, h.reshape(H_p, 1)) + self.b_h)

        H_t = z_t * H.reshape(H_d, H_p, 1) + (1 - z_t) * H_t_hat
        H_t = H_t.reshape(H_d, H_p)  # [D, p_s]
        # print(torch.matmul(self.W_z, H.reshape(H_d, H_p, 1)))
        return H_t


"""
H = torch.randn(33, 64)
x = torch.randn(33)
h = torch.randn(64)
model = tMGRU(hidden_size=64)
y = model(H, x, h)
print(y.shape)
"""


class GMGRU(nn.Module):
    def __init__(self, input_size, hidden_size, q):
        super(GMGRU, self).__init__()
        """
        input size: 33
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.zeros = torch.autograd.Variable(torch.zeros(hidden_size))
        self.gamma_h = nn.Linear(input_size, hidden_size)
        self.gamme_st = nn.Linear(input_size, hidden_size)
        self.grucell = nn.GRUCell(input_size, hidden_size)

        # eq.(2)
        self.x_g = nn.Linear(hidden_size, input_size, bias=True)

        # compute u_t, eq.(5)
        self.w_u0 = nn.Linear(hidden_size, q, bias=True)
        self.w_u1 = nn.Linear(2 * hidden_size, q, bias=True)
        self.w_u = nn.Linear(3 * hidden_size, q, bias=True)

        # eq.(7)
        self.w_h = nn.Linear(hidden_size+q, input_size)

        # tM-GRU
        self.tmgru = tMGRU(hidden_size)

        # eq.(8)
        self.x_st = nn.Linear(hidden_size, 1)

        # eq.(10),weight estimation
        self.w_rg = nn.Linear(hidden_size, input_size, bias=False)
        self.w_rs = nn.Parameter(torch.randn(1, hidden_size), requires_grad=True)

        # 全连接层
        """
        self.fc = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                nn.Dropout(0.5),
                                nn.ReLU(),
                                nn.Linear(hidden_size, input_size)
                                )
        """

    def forward(self, data):
        """
        data: [3, 33, 49]
        X: values
        M: Missing markers
        Delta；time-stamp
        """
        X = torch.squeeze(data[0])  # .size = (33,49)
        Mask = torch.squeeze(data[1])  # .size = (33,49)
        Delta = torch.squeeze(data[2])  # .size = (33,49)

        # initialize hidden_state
        hidden_state = Variable(torch.randn(self.hidden_size))  # [64]
        Hidden_state = Variable(torch.randn(self.input_size, self.hidden_size))  # [D, hidden_size]
        # print(Hidden_state)
        # print(Hidden_state.shape)

        h = hidden_state
        H = Hidden_state

        output = []
        x_pred = []
        H_s = []
        for t in range(X.size(1)):
            x = torch.squeeze(X[:, t:t+1])  # [33]
            m = torch.squeeze(Mask[:, t:t+1])
            d = torch.squeeze(Delta[:, t:t+1])

            # print(x.shape)
            # print(d.shape)
            gamma_h = torch.exp(-torch.max(self.zeros, self.gamma_h(d)))  # eq.(1) [hidden_size]
            # print(gamma_h.shape)

            h = gamma_h * h
            x_gt = self.x_g(h)  # eq.(2)
            # x_gt_c = m * x + (1 - m) * x_gt  # eq.(3)
            # print(x_gt.shape)
            # print(h.shape)

            # tmGRU output
            x_st = self.x_st(H).reshape(-1)  # eq.(8) [input_size, 1]
            # print(x_st)
            # print(x_st.shape)
            x_st_c = m * x + (1 - m) * x_st  # [input_size]
            # print(x_st_c.shape)

            h = self.grucell(x_st_c.reshape(-1, self.input_size), h.reshape(-1, self.hidden_size))  # eq.(4)
            # print(h.shape)
            # h = h.reshape(-1)
            output.append(h)

            # compute u_t, eq.(5)
            h_dense = torch.tensor([item.detach().numpy() for item in output]).reshape(-1, 64)
            # print(h_dense.shape)
            if t > 2:
                # kp
                h_dense = h_dense[t-3:t, :].reshape(1, -1)
                # print(h_dense.shape)
                u_t = self.w_u(h_dense)
            elif t == 0:
                h_dense = hidden_state.reshape(1, -1)   # hidden_state 是初始h_0
                u_t = self.w_u0(h_dense)
            else:
                h_dense = torch.cat((hidden_state.reshape(-1, 64), h_dense[:t, :]), dim=0).reshape(1, -1)
                # print(h_dense.shape)
                if h_dense.shape[-1] == 128:
                    u_t = self.w_u1(h_dense)
                else:
                    u_t = self.w_u(h_dense)  # [1, q]
            # print(u_t.shape)

            h_t_hat = torch.cat((h, u_t), dim=1)  # eq.(6)
            # print(h_t_hat.shape)  # [1, (hidden_size+q)]
            x_gt_hat = self.w_h(h_t_hat)  # eq.(7)

            # x_gt_hat_c = m * x + (1 - m) * x_gt_hat
            # x_pred.append(x_gt_hat)

            # tmGRU
            # eq.(9)
            gamma_st = torch.exp(-torch.max(self.zeros, self.gamme_st(d)))  # eq.(1) [hidden_size]
            H = gamma_st * H
            H = self.tmgru(H, x_st_c, h)
            # print(H.shape)
            H_s.append(H)
            # print(H)

            # weight estimation
            # print(torch.matmul(self.w_rs, H.transpose(0, 1)))
            alpha1 = torch.softmax(self.w_rg(h) / (self.w_rg(h) + torch.matmul(self.w_rs, H.transpose(0, 1))), dim=-1)
            alpha2 = torch.softmax(torch.matmul(self.w_rs, H.transpose(0, 1)) / (self.w_rg(h) +
                                            torch.matmul(self.w_rs, H.transpose(0, 1))), dim=-1)
            # print(alpha1.shape)
            # print(x_gt_hat)
            # print(torch.matmul(self.w_rs, H.transpose(0, 1)) / (self.w_rg(h) + torch.matmul(self.w_rs, H.transpose(0, 1))))
            # print(alpha2)
            # print(self.w_rg(h))
            # print(torch.matmul(self.w_rs, H.transpose(0, 1)))
            # print((self.w_rg(h) + torch.matmul(self.w_rs, H.transpose(0, 1))))

            x_t = alpha1 * x_gt_hat + alpha2 * x_st
            # print(x_t)
            # print(x_t.shape)
            x_pred.append(x_t)

        x_pred = torch.tensor([item.detach().numpy() for item in x_pred]).reshape(-1, self.input_size).transpose(0, 1)
        H_s = torch.tensor([item.detach().numpy() for item in H_s])
        # print(x_pred)
        # print(H_s)

        return H_s, x_pred  # [T, input_size, hidden_size], [33, 49]


print('蒋秃秃')

"""
x = torch.randn(3, 33, 49)
model = GMGRU(input_size=33, hidden_size=64, q=128)
c, pred = model(x)
# print(len(c))
print(c.shape)
print(pred.shape)
# print(x)
"""


# input size:[33, 49, embed_size] or [49, 33, embed_size]
# out size:[33, 49, embed_size]
class EncoderLayer(nn.Module):
    def __init__(self, embed_size, d_ff, heads, dropout=0.0):
        super(EncoderLayer, self).__init__()
        # self-attention
        self.attn = MultiHeadAttention(embed_size, heads)

        # 全连接层
        self.fc = nn.Sequential(nn.Linear(embed_size, d_ff),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, embed_size),
                                nn.Dropout(dropout)
                                )

        # add residual and norm
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        # x.shape: [33, 49, embed_size]

        # positional embedding
        # x_pe = self.pe(x)

        # self-attention
        N, T, E = x.shape
        attn = self.attn(x)  # [D, T, embed_size]

        # add & normal
        norm1 = self.norm1(attn + x)
        # print(norm1.shape)

        # feed-forward: FANG
        ff = self.fc(norm1)

        # add residual & norm
        norm2 = self.norm2(ff + norm1)  # [33, 49, embed_size]

        return norm2


"""
x = torch.randn(33, 49, 512)
model = EncoderLayer(embed_size=512, d_ff=2048, heads=8, dropout=0.5)
y = model(x)
print(y.shape)
"""


# input:[33, 49]
# output:[33, 49 ,embed_size]
class Encoder(nn.Module):
    def __init__(self, embed_size, heads=8, num_layers=6, d_ff=2048, dropout=0.0):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(1, embed_size)
        self.num_layers = num_layers
        self.pe = PositionalEncoding(embed_size, dropout)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(embed_size, d_ff, heads, dropout) for _ in range(num_layers)]
        )

    def forward(self, x):
        # 将特征投影到更高维
        x = x.unsqueeze(2)  # [33, 49, 1] or [49, 33, 1]
        x = self.linear(x)  # [33, 49, embed_size] or [49, 33, embed_size]
        x = F.relu(x)

        # 位置嵌入
        x = self.pe(x)  # [33, 49, embed_size] or [49, 33, embed_size]
        # print(x.shape)

        for encoder in self.encoder_layers:
            out = encoder(x)  # [33, 49, embed_size] or [49, 33, embed_size]
            x = out

        return out


"""
x = torch.randn(33, 49)
model = Encoder(embed_size=512, heads=8, num_layers=3, d_ff=2048, dropout=0.5)
y = model(x)
print(y.shape)
"""


# capture missing MTS relationship
class STransformer(nn.Module):
    def __init__(self, embed_size, heads=8, num_layers=6, d_ff=2048, dropout=0.0):
        super(STransformer, self).__init__()
        self.num_layers = num_layers
        self.trans1 = Encoder(embed_size, heads, num_layers, d_ff, dropout)
        self.trans2 = Encoder(embed_size, heads, num_layers, d_ff, dropout)

        # feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(2*embed_size, embed_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size, embed_size)
        )

    def forward(self, x):
        # x.shape:[33, 49]
        # 时间依赖
        t_trans_out = self.trans1(x)  # [33, 49, embed_size]

        # 变量之间的关系
        x_T = x.transpose(0, 1)  # [44, 39]
        s_trans_out = self.trans2(x_T)
        s_trans_out = s_trans_out.transpose(0, 1)  # [33, 49, embed_size]

        # fusion
        out = torch.cat((t_trans_out, s_trans_out), dim=-1)  # [33, 49, 2 * embed_size]
        out = self.fusion(out)

        # imputation
        # x_tran = torch.tanh(self.f2(out))

        return out


"""
x = torch.randn(33, 49)
model = STransformer(embed_size=512, heads=8, num_layers=3, d_ff=2048, dropout=0.5)
y = model(x)
print(y.shape)
"""


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)  # [50000, 1] -> [[0.], [1.], [2.],....[4999]]
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


