import torch
import torch.nn as nn
from modules import *
import numpy as np
from dataloader import data_dataloader
from function import *
import torch.optim as optim
import matplotlib.pyplot as plt
from model import Model

t_dataset = np.load('input/dataset.npy')
t_out = np.load('input/y1_out.npy')

# print(t_dataset.shape)  [4000, 3, 33, 49]
# print(t_out.shape)  [4000, 1]

train_dataloader, dev_dataloader, test_dataloader = data_dataloader(t_dataset, t_out, train_proportion=0.8, \
                                                                    dev_proportion=0.2, test_proportion=0.2)

if __name__ == '__main__':
    input_size = 33
    num_layers = 3
    hidden_size = 64
    q = 64
    heads = 8

    x_mean = torch.Tensor(np.load('input/x_mean_aft_nor.npy'))
    x_mean = x_mean.reshape(-1, 1)
    print(x_mean.shape)
    x_median = torch.Tensor(np.load('input/x_median_aft_nor.npy'))

    model = Model(input_size, hidden_size, q, heads)

    # criterion = torch.nn.BCELoss()

    learning_rate = 0.0001
    learning_rate_decay = 10
    n_epochs = 20
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    for epoch in range(n_epochs):
        for train_data, train_label in train_dataloader:
            train_data = torch.squeeze(train_data[0])  # [3, 33, 49]
            # print(train_data.shape)
            train_label = train_label.reshape(-1)  # [1]
            x_1, x_2, X = model(train_data)  # [49, 33]
            X = X.requires_grad_()
            # print(train_data)

            # print(X)
            # print(X[i].shape)
            # x1_loss = calculateMAE(x_1.transpose(0, 1), train_data[0], train_data[1])
            # x2_loss = calculateMAE(x_2.transpose(0, 1), train_data[0], train_data[1])
            loss = calculateMAE(X.transpose(0, 1), train_data[0], train_data[1])
            # loss_all = x1_loss + x2_loss + loss
            # print(loss)

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        mean_loss = np.mean(losses)

        print('epoch:{} train_loss:{:.4f}'.format(epoch, mean_loss))

        # TEST DATASET
        for test_data, test_label in test_dataloader:
            test_data = torch.squeeze(test_data[0])  # [3, 33, 49]
            test_label = test_label.reshape(-1)  # [1]
            x_1, x_2, X = model(test_data)  # [33, 49]
            X = X.requires_grad_()

            # print(X[i].shape)
            # x1_loss = calculateMAE(x_1.transpose(0, 1), test_data[0], test_data[1])
            # x2_loss = calculateMAE(x_2.transpose(0, 1), test_data[0], test_data[1])
            loss = calculateMAE(X.transpose(0, 1), test_data[0], test_data[1])

            # print(loss)

            losses.append(loss.item())

        mean_loss = np.mean(losses)

        print('epoch:{} test_loss:{:.4f}'.format(epoch, mean_loss))


    plt.plot(losses, label='X loss')
    plt.legend()
    plt.show()
