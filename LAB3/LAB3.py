import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
from numpy.lib.npyio import load
import pandas as pd
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
from PIL import Image
from torchvision import transforms
import torchvision
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.models.resnet import resnet18, resnet50

def save_model(model, name) :
    FILE = name + ".pth"
    torch.save(model.state_dict(), FILE)

def plot_confusion_mat(model, mode) :
    np.set_printoptions(precision=2)
    y_pred, y_true = classifier(model)
    cm = confusion_matrix(y_true, y_pred,
                                labels=[0, 1, 2, 3, 4],
                                normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4])
    disp.plot(cmap=plt.cm.Blues)
    if mode == 0 :
        plt.title("resnet18 pretrained Normalized confusion matrix", fontsize=16)
    elif mode == 1 :
        plt.title("resnet18 nonpretrained Normalized confusion matrix", fontsize=16)
    elif mode == 2 :
        plt.title("resnet50 pretrained Normalized confusion matrix", fontsize=16)
    else :
        plt.title("resnet50 nonpretrained Normalized confusion matrix", fontsize=16)
    plt.ylabel("True label")
    plt.xlabel("Predict label")
    if mode == 0 :
        plt.savefig("resnet18_pre")
    elif mode == 1 :
        plt.savefig("resnet18_nonpre")
    elif mode == 2 :
        plt.savefig("resnet50_pre")
    else :
        plt.savefig("resnet50_nonpre")
    plt.clf()
    # plt.show()

def plot_accuracy(acc_t, acc_v, acc_t_wo, acc_v_wo, mode) :
    if mode == 0 :
        plt.title('Result Comparison(ResNet18)', fontsize=18)
    else :
        plt.title('Result Comparison(ResNet50)', fontsize=18)
    plt.xlabel("Epochs", fontsize=12, labelpad=10)
    plt.ylabel("Accuracy(%)", fontsize=12, labelpad=10)
    x = list(range(1, len(acc_t)+1))
    # train with pretrain
    plt.plot(x, acc_t, color='r', label='Train(with pretraining)')
    # test with pretrain
    plt.plot(x, acc_v, color='b', label='Test(with pretraining)')
    # train without pretrain
    plt.plot(x, acc_t_wo, '-o', color='g', label='Train(w/o pretraining)')
    # test without pretrain
    plt.plot(x, acc_v_wo, '-o', color='k', label='Test(w/o pretraining)')
    plt.legend(loc='upper left')
    if mode == 0 :
        plt.savefig("resnet18_result")
    else :
        plt.savefig("resnet50_result")
    # plt.show()

def getData(mode) :
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)

class RetinopathyLoader(data.Dataset) :
    def __init__(self, root, mode, transform=None) :
        self.to_tensor = transforms.ToTensor()
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        self.transform = transform
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self) :
        return len(self.img_name)

    def __getitem__(self, index) :
        img = Image.open(self.root + self.img_name[index] + ".jpeg")
        label = self.label[index]
        if self.transform :
            img = self.transform(img)

        return (img, label)

def train_and_test(model, epochs) :
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("now device is",device)
    model = model.to(device)
    model.device = device
    train_acc_list = []
    test_acc_list = []

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)


    for epoch in range (epochs) :
        # train
        model.train()
        acc_list = []
        for inputs, ground_truth in train_loader :
            inputs = inputs.to(device)
            ground_truth = ground_truth.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, ground_truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accuracy = (outputs.argmax(dim=-1) == ground_truth.to(device)).float().mean()
            # accuracy = accuracy_score(ground_truth, outputs)
            accuracy = accuracy.cpu()
            acc_list.append(accuracy)

        train_acc = sum(acc_list) / len(acc_list)
        print(f"Train epoch {epoch + 1:03d}/{epochs:03d}, accuracy = {train_acc:.5f}")
        train_acc_list.append(train_acc)

        # test
        acc_list.clear()
        model.eval()
        for inputs, ground_truth in test_loader :
            inputs = inputs.to(device)
            ground_truth = ground_truth.to(device)
            with torch.no_grad():
                outputs = model(inputs)
            loss = criterion(outputs, ground_truth)
            accuracy = (outputs.argmax(dim=-1) == ground_truth.to(device)).float().mean()
            # accuracy = accuracy_score(ground_truth, outputs)
            accuracy = accuracy.cpu()
            acc_list.append(accuracy)
        
        test_acc = sum(acc_list) / len(acc_list)
        print(f"Test epoch {epoch + 1:03d}/{epochs:03d}, accuracy = {test_acc:.5f}")
        test_acc_list.append(test_acc)
    return train_acc_list, test_acc_list

def classifier(model) :
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.device = device
    model.eval()
    test_accs = []
    outputs_list = []
    ground_truth_list = []

    for inputs, ground_truth in test_loader :
        with torch.no_grad() :
            outputs = model(inputs.to(device))
            outputs_list += (outputs.argmax(dim=-1)).tolist()
            ground_truth_list += ground_truth.tolist()
            # acc = accuracy_score(ground_truth.to(device), outputs)
            acc = (outputs.argmax(dim=-1) == ground_truth.to(device)).float().mean()
            test_accs.append(acc)
    acc = sum(test_accs) / len(test_accs)
    print(f"{acc.cpu().numpy() * 100:.5f}")
    return outputs_list, ground_truth_list

if __name__ == "__main__" :
    # hyper parameters
    batch_size = 32
    epochs18 = 15
    epochs50 = 8

    # load data
    trans = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(), 
                                transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])])
    trans2 = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(), 
                                transforms.RandomVerticalFlip(p=0.3),
                                transforms.RandomHorizontalFlip(p=0.3),
                                transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])])
    train_set = RetinopathyLoader(root="D:/research/DLP/LAB3/data/", mode="train", transform=trans2)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    test_set = RetinopathyLoader(root="D:/research/DLP/LAB3/data/", mode="test", transform=trans)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # # ResNet18
    # resnet18_pre = torchvision.models.resnet18(pretrained=True)
    # resnet18_pre.fc = nn.Linear(in_features=512, out_features=5, bias=True)
    # resnet18_pre.eval()
    # resnet18_non_pre = torchvision.models.resnet18()
    # resnet18_non_pre.fc = nn.Linear(in_features=512, out_features=5, bias=True)
    # resnet18_non_pre.eval()

    # acc_t, acc_v = train_and_test(resnet18_pre, epochs18)
    # plot_confusion_mat(resnet18_pre, 0)
    # acc_t_nonpre, acc_v_nonpre = train_and_test(resnet18_non_pre, epochs18)
    # plot_confusion_mat(resnet18_non_pre, 1)
    # plot_accuracy(acc_t, acc_v, acc_t_nonpre, acc_v_nonpre, 0)

    # # ResNet50
    # resnet50_pre = torchvision.models.resnet50(pretrained=True)
    # resnet50_pre.fc = nn.Linear(in_features=2048, out_features=5, bias=True)
    # resnet50_pre.eval()
    # resnet50_non_pre = torchvision.models.resnet50()
    # resnet50_non_pre.fc = nn.Linear(in_features=2048, out_features=5, bias=True)
    # resnet50_non_pre.eval()
    
    # acc_t, acc_v = train_and_test(resnet50_pre, epochs50)
    # plot_confusion_mat(resnet50_pre, 2)
    # acc_t_nonpre, acc_v_nonpre = train_and_test(resnet50_non_pre, epochs50)
    # plot_confusion_mat(resnet50_non_pre, 3)
    # plot_accuracy(acc_t, acc_v, acc_t_nonpre, acc_v_nonpre, 1)

    # # save model
    # save_model(resnet18_pre, 'resnet18_pre')
    # save_model(resnet18_non_pre, 'resnet18_non_pre')
    # save_model(resnet50_pre, 'resnet50_pre')
    # save_model(resnet50_non_pre, 'resnet50_non_pre')

    # load model
    resnet18_pre = torchvision.models.resnet18(pretrained=True)
    resnet18_pre.fc = nn.Linear(in_features=512, out_features=5, bias=True)
    resnet18_non_pre = torchvision.models.resnet18()
    resnet18_non_pre.fc = nn.Linear(in_features=512, out_features=5, bias=True)
    resnet50_pre = torchvision.models.resnet50(pretrained=True)
    resnet50_pre.fc = nn.Linear(in_features=2048, out_features=5, bias=True)
    resnet50_non_pre = torchvision.models.resnet50()
    resnet50_non_pre.fc = nn.Linear(in_features=2048, out_features=5, bias=True)

    resnet18_pre.load_state_dict(torch.load('resnet18_pre.pth'))
    resnet18_non_pre.load_state_dict(torch.load('resnet18_non_pre.pth'))
    resnet50_pre.load_state_dict(torch.load('resnet50_pre.pth'))
    resnet50_non_pre.load_state_dict(torch.load('resnet50_non_pre.pth'))

    # show accuracy
    _,_ = classifier(resnet18_pre)
    _,_ = classifier(resnet18_non_pre)
    _,_ = classifier(resnet50_pre)
    _,_ = classifier(resnet50_non_pre)