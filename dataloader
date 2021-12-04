import torch
import torch.utils.data as utils
import numpy as np


def data_dataloader(dataset, outcomes, \
                    train_proportion=0.08, dev_proportion=0.025, test_proportion=0.025):
    train_index = int(np.floor(dataset.shape[0] * train_proportion))
    dev_index = int(np.floor(dataset.shape[0] * (train_proportion - dev_proportion)))
    test_index = int(np.floor(dataset.shape[0] * test_proportion))

    # split dataset to tarin/dev/test set
    train_data, train_label = dataset[:train_index, :, :, :], outcomes[:train_index, :]
    dev_data, dev_label = dataset[dev_index:train_index, :, :, :], outcomes[dev_index:train_index, :]
    test_data, test_label = dataset[train_index:train_index + test_index, :, :, :], outcomes[train_index:train_index + test_index, :]

    # ndarray to tensor
    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    dev_data, dev_label = torch.Tensor(dev_data), torch.Tensor(dev_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)

    # tensor to dataset
    train_dataset = utils.TensorDataset(train_data, train_label)
    dev_dataset = utils.TensorDataset(dev_data, dev_label)
    test_dataset = utils.TensorDataset(test_data, test_label)

    # dataset to dataloader
    train_dataloader = utils.DataLoader(train_dataset)
    dev_dataloader = utils.DataLoader(dev_dataset)
    test_dataloader = utils.DataLoader(test_dataset)

    print("train_data.shape : {}\t train_label.shape : {}".format(train_data.shape, train_label.shape))
    print("dev_data.shape : {}\t dev_label.shape : {}".format(dev_data.shape, dev_label.shape))
    print("test_data.shape : {}\t test_label.shape : {}".format(test_data.shape, test_label.shape))

    return train_dataloader, dev_dataloader, test_dataloader
