import numpy as np
import torch
from torch import cuda
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

def read_bci_data() :
    S4b_train = np.load('S4b_train.npz')
    X11b_train = np.load('X11b_train.npz')
    S4b_test = np.load('S4b_test.npz')
    X11b_test = np.load('X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1
    test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

    return train_data, train_label, test_data, test_label

def show_result(acc, net_type) :
    if net_type == 0 :
        plt.title('Activation function comparison(EEGNet)', fontsize=18)
    elif net_type == 1 :
        plt.title('Activation function comparison(DeepConvNet)', fontsize=18)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy(%)", fontsize=12)
    x = []
    y = []
    for i in range(len(acc[0])) :
        x.append(i + 1)
        y.append(acc[0][i] * 100)
    if net_type == 2 :
        plt.plot(x, y, color='r', label='EEGNet ReLU')
    else :
        plt.plot(x, y, color='r', label='ReLU train')
    x.clear()
    y.clear()
    for i in range(len(acc[1])) :
        x.append(i + 1)
        y.append(acc[1][i] * 100)
    if net_type == 2 :
        plt.plot(x, y, color='r', label='EEGNet LeakyReLU')
    else :
        plt.plot(x, y, color='g', label='ReLU test')
    x.clear()
    y.clear()
    for i in range(len(acc[2])) :
        x.append(i + 1)
        y.append(acc[2][i] * 100)
    if net_type == 2 :
        plt.plot(x, y, color='r', label='EEGNet ELU')
    else :
        plt.plot(x, y, color='b', label='leaky ReLU train')
    x.clear()
    y.clear()
    for i in range(len(acc[3])) :
        x.append(i + 1)
        y.append(acc[3][i] * 100)
    if net_type == 2 :
        plt.plot(x, y, color='r', label='DeepConvNet ReLU')
    else :
        plt.plot(x, y, color='c', label='leaky ReLU test')
    x.clear()
    y.clear()
    for i in range(len(acc[4])) :
        x.append(i + 1)
        y.append(acc[4][i] * 100)
    if net_type == 2 :
        plt.plot(x, y, color='r', label='DeepConvNet LeakyReLU')
    else :
        plt.plot(x, y, color='m', label='ELU train')
    x.clear()
    y.clear()
    for i in range(len(acc[5])) :
        x.append(i + 1)
        y.append(acc[5][i] * 100)
    if net_type == 2 :
        plt.plot(x, y, color='r', label='DeepConvNet ELU')
    else :
        plt.plot(x, y, color='y', label='ELU test')

    plt.legend(loc='lower right')
    plt.show()

class EEGNet(nn.Module) :
    def __init__(self, func) :
        super(EEGNet, self).__init__()
        if func == 0 :
            self.firstconv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(1,51), stride=1, padding=(0,25) , bias=False),
                nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            self.depthwiseConv = torch.nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=(2,1), stride=1, groups=16, bias=False),
                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0),
                nn.Dropout(p=0.25)
            )
            self.separableConv = torch.nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=(1,15), stride=1, padding=(0,7), bias=False),
                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0),
                nn.Dropout(p=0.25)
            )
            self.classify = torch.nn.Sequential(
                nn.Linear(736, 2, bias=True)
            )
        elif func == 1 :
            self.firstconv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(1,51), stride=1, padding=(0,25) , bias=False),
                nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            self.depthwiseConv = torch.nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=(2,1), stride=1, groups=16, bias=False),
                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.LeakyReLU(),
                nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0),
                nn.Dropout(p=0.25)
            )
            self.separableConv = torch.nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=(1,15), stride=1, padding=(0,7), bias=False),
                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.LeakyReLU(),
                nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0),
                nn.Dropout(p=0.25)
            )
            self.classify = torch.nn.Sequential(
                nn.Linear(736, 2, bias=True)
            )
        elif func == 2 :
            self.firstconv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(1,51), stride=1, padding=(0,25) , bias=False),
                nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            self.depthwiseConv = torch.nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=(2,1), stride=1, groups=16, bias=False),
                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0),
                nn.Dropout(p=0.25)
            )
            self.separableConv = torch.nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=(1,15), stride=1, padding=(0,7), bias=False),
                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0),
                nn.Dropout(p=0.25)
            )
            self.classify = torch.nn.Sequential(
                nn.Linear(736, 2, bias=True)
            )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.flatten(1)
        x = self.classify(x)
        return x

class DeepConvNet(nn.Module):
    def __init__(self, func):
        super(DeepConvNet, self).__init__()
        if func == 0 :
            self.firstConv = nn.Sequential(
                nn.Conv2d(1, 25, kernel_size=(1,5), stride=1, padding=(0,25) , bias=False),
                nn.Conv2d(25, 25, kernel_size=(2,1), stride=1, padding=(0,25) , bias=False),
                nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1,2)),
                nn.Dropout(p=0.5)
            )
            self.secondConv = nn.Sequential(
                nn.Conv2d(25, 50, kernel_size=(1,5)),
                nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1,2)),
                nn.Dropout(p=0.5)
            )
            self.thirdConv = nn.Sequential(
                nn.Conv2d(50, 100, kernel_size=(1,5)),
                nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1,2)),
                nn.Dropout(p=0.5)
            )
            self.fourthConv = nn.Sequential(
                nn.Conv2d(100, 200, kernel_size=(1,5)),
                nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1,2)),
                nn.Dropout(p=0.5)
            )
            self.classify = nn.Sequential(
                nn.Linear(9800, 2, bias=True)
            )
        elif func == 1 :
            self.firstConv = nn.Sequential(
                nn.Conv2d(1, 25, kernel_size=(1,5), stride=1, padding=(0,25) , bias=False),
                nn.Conv2d(25, 25, kernel_size=(2,1), stride=1, padding=(0,25) , bias=False),
                nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=(1,2)),
                nn.Dropout(p=0.5)
            )
            self.secondConv = nn.Sequential(
                nn.Conv2d(25, 50, kernel_size=(1,5)),
                nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=(1,2)),
                nn.Dropout(p=0.5)
            )
            self.thirdConv = nn.Sequential(
                nn.Conv2d(50, 100, kernel_size=(1,5)),
                nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=(1,2)),
                nn.Dropout(p=0.5)
            )
            self.fourthConv = nn.Sequential(
                nn.Conv2d(100, 200, kernel_size=(1,5)),
                nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=(1,2)),
                nn.Dropout(p=0.5)
            )
            self.classify = nn.Sequential(
                nn.Linear(9800, 2, bias=True)
            )
        elif func == 2 :
            self.firstConv = nn.Sequential(
                nn.Conv2d(1, 25, kernel_size=(1,5), stride=1, padding=(0,25) , bias=False),
                nn.Conv2d(25, 25, kernel_size=(2,1), stride=1, padding=(0,25) , bias=False),
                nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1,2)),
                nn.Dropout(p=0.5)
            )
            self.secondConv = nn.Sequential(
                nn.Conv2d(25, 50, kernel_size=(1,5)),
                nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1,2)),
                nn.Dropout(p=0.5)
            )
            self.thirdConv = nn.Sequential(
                nn.Conv2d(50, 100, kernel_size=(1,5)),
                nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1,2)),
                nn.Dropout(p=0.5)
            )
            self.fourthConv = nn.Sequential(
                nn.Conv2d(100, 200, kernel_size=(1,5)),
                nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1,2)),
                nn.Dropout(p=0.5)
            )
            self.classify = nn.Sequential(
                nn.Linear(9800, 2, bias=True)
            )

    def forward(self, x):
        x = self.firstConv(x)
        x = self.secondConv(x)
        x = self.thirdConv(x)
        x = self.fourthConv(x)
        x = x.flatten(1)
        x = self.classify(x)
        return x

def train(model, n_epochs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.device = device
    train_acc_list = []
    test_acc_list = []

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0015, weight_decay=1e-5)

    for epoch in range(n_epochs):
        model.train()
        accuracy_list = []

        for inputs, ground_truth in train_loader:
            inputs = inputs.to(device)
            ground_truth = ground_truth.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, ground_truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accuracy = (outputs.argmax(dim=-1) == ground_truth.to(device)).float().mean()
            accuracy = accuracy.cpu()
            accuracy_list.append(accuracy)

        train_acc = sum(accuracy_list) / len(accuracy_list)
        print(f"Train epoch {epoch + 1:03d}/{n_epochs:03d}, accuracy = {train_acc:.5f}")
        train_acc_list.append(train_acc)


        model.eval()
        valid_accs = []
        for inputs, ground_truth in test_loader:
            inputs = inputs.to(device)
            ground_truth = ground_truth.to(device)
            with torch.no_grad():
                outputs = model(inputs.to(device))
            loss = criterion(outputs, ground_truth.to(device))
            accuracy = (outputs.argmax(dim=-1) == ground_truth.to(device)).float().mean()
            accuracy = accuracy.cpu()
            valid_accs.append(accuracy)

        valid_acc = sum(valid_accs) / len(valid_accs)
        print(f"Test epoch {epoch + 1:03d}/{n_epochs:03d}, accuracy = {valid_acc:.5f}")
        test_acc_list.append(valid_acc)
    return train_acc_list, test_acc_list

def test(model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    test_accs = []
    for inputs, ground_truth in test_loader:
        with torch.no_grad():
            logits = model(inputs.to(device))
            acc = (logits.argmax(dim=-1) == ground_truth.to(device)).float().mean()
            test_accs.append(acc)

    test_acc = sum(test_accs) / len(test_accs)

    print(f"{test_acc.cpu().numpy() * 100:.5f}")

if __name__ == "__main__" :
    # hyper parameters
    batch_size = 1024
    epochs = 350
    criterion = nn.CrossEntropyLoss()

    # read data
    train_data, train_label, test_data, test_label = read_bci_data()
    train_data = np.float32(train_data)
    train_label = np.int64(train_label)
    test_data = np.float32(test_data)
    test_label = np.int64(test_label)
    train_set = []
    test_set = []
    for i in range(len(train_data)):
        train_set.append([train_data[i],train_label[i]])
    for i in range(len(test_data)):
        test_set.append([test_data[i],test_label[i]])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    # acc = []
    # EEG_ReLU = EEGNet(0)
    # acc_t, acc_v = train(EEG_ReLU, epochs)
    # acc.append(acc_t)
    # acc.append(acc_v)
    # EEG_LeakyReLU = EEGNet(1)
    # acc_t, acc_v = train(EEG_LeakyReLU, epochs)
    # acc.append(acc_t)
    # acc.append(acc_v)
    # EEG_ELU = EEGNet(2)
    # acc_t, acc_v = train(EEG_ELU, epochs)
    # acc.append(acc_t)
    # acc.append(acc_v)
    # show_result(acc, 0)

    # acc.clear()
    # deep_ReLU = DeepConvNet(0)
    # acc_t, acc_v = train(deep_ReLU, epochs)
    # acc.append(acc_t)
    # acc.append(acc_v)
    # deep_LeakyReLU = DeepConvNet(1)
    # acc_t, acc_v = train(deep_LeakyReLU, epochs)
    # acc.append(acc_t)
    # acc.append(acc_v)
    # deep_ELU = DeepConvNet(2)
    # acc_t, acc_v = train(deep_ELU, epochs)
    # acc.append(acc_t)
    # acc.append(acc_v)
    # show_result(acc, 1)

    # torch.save(EEG_ReLU.state_dict(), 'EEG_ReLU.pth')
    # torch.save(EEG_LeakyReLU.state_dict(), 'EEG_LeakyReLU.pth')
    # torch.save(EEG_ELU.state_dict(), 'EEG_ELU.pth')
    # torch.save(deep_ReLU.state_dict(), 'deep_ReLU.pth')
    # torch.save(deep_LeakyReLU.state_dict(), 'deep_LeakyReLU.pth')
    # torch.save(deep_ELU.state_dict(), 'deep_ELU.pth')

    EEG_ReLU = EEGNet(0)
    EEG_LeakyReLU = EEGNet(1)
    EEG_ELU = EEGNet(2)
    deep_ReLU = DeepConvNet(0)
    deep_LeakyReLU = DeepConvNet(1)
    deep_ELU = DeepConvNet(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    EEG_ReLU.load_state_dict(torch.load('EEG_ReLU.pth'))
    EEG_ReLU.to(device)
    EEG_LeakyReLU.load_state_dict(torch.load('EEG_LeakyReLU.pth'))
    EEG_LeakyReLU.to(device)
    EEG_ELU.load_state_dict(torch.load('EEG_ELU.pth'))
    EEG_ELU.to(device)
    deep_ReLU.load_state_dict(torch.load('deep_ReLU.pth'))
    deep_ReLU.to(device)
    deep_LeakyReLU.load_state_dict(torch.load('deep_LeakyReLU.pth'))
    deep_LeakyReLU.to(device)
    deep_ELU.load_state_dict(torch.load('deep_ELU.pth'))
    deep_ELU.to(device)
    
    print("EEGNet ReLU accuracy:(%)")
    test(EEG_ReLU)
    print("EEGNet LeakyReLU accuracy:(%)")
    test(EEG_LeakyReLU)
    print("EEGNet ELU accuracy:(%)")
    test(EEG_ELU)
    print("deepConvNet ReLU accuracy:(%)")
    test(deep_ReLU)
    print("deepConvNet LeakyReLU accuracy:(%)")
    test(deep_LeakyReLU)
    print("deepConvNet ELU accuracy:(%)")
    test(deep_ELU)