import os
import torch
from torch.optim import optimizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import save_image, make_grid
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import json
import torchvision.models as models
from test import Generator

class EvaluationModel():
    def __init__(self):
        #modify the path to your own path
        checkpoint = torch.load('D:/research/DLP/LAB5/classifier_weight.pth')
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Sequential(
            nn.Linear(512,24),
            nn.Sigmoid()
        )
        self.resnet18.load_state_dict(checkpoint['model'])
        self.resnet18 = self.resnet18.cuda()
        self.resnet18.eval()
        self.classnum = 24
    def compute_acc(self, out, onehot_labels):
        batch_size = out.size(0)
        acc = 0
        total = 0
        for i in range(batch_size):
            k = int(onehot_labels[i].sum().item())
            total += k
            outv, outi = out[i].topk(k)
            lv, li = onehot_labels[i].topk(k)
            for j in outi:
                if j in li:
                    acc += 1
        return acc / total
    def eval(self, images, labels):
        with torch.no_grad():
            #your image shape should be (batch, 3, 64, 64)
            out = self.resnet18(images)
            acc = self.compute_acc(out.cpu(), labels.cpu())
            return acc

def get_test_conditions():
    with open('D:/research/DLP/LAB5/objects.json', 'r') as file:
        classes = json.load(file)
    with open('D:/research/DLP/LAB5/new_test_2021_summer.json', 'r') as file:
        test_conds_list = json.load(file)

    labels = torch.zeros(len(test_conds_list), len(classes))
    for i in range(len(test_conds_list)):
        for condition in test_conds_list[i]:
            labels[i, int(classes[condition])] = 1.
    return labels

def train(dataloader, g_model, d_model, z_dim, epochs, lr, batch_size):
    criterion = nn.BCELoss()
    optimizer_g = torch.optim.Adam(g_model.parameters(), lr, betas=(0.5,0.99))
    optimizer_d = torch.optim.Adam(d_model.parameters(), lr, betas=(0.5,0.99))
    eval_model = EvaluationModel()
    best_score = 0

    for epoch in range(1, 1+epochs):
        loss_sum_g = 0
        loss_sum_d = 0
        for i, (images, conds) in enumerate(dataloader):
            g_model.train()
            d_model.train()
            images = images.to(device)
            conds = conds.to(device)
            batch_size=len(images)
            real = torch.ones(batch_size).to(device)
            fake = torch.zeros(batch_size).to(device)

            # train discriminator
            optimizer_d.zero_grad()
            # real loss
            predicts_r = d_model(images, conds)
            loss_real = criterion(predicts_r, real)
            # fake loss
            z = torch.randn(batch_size, z_dim).to(device)
            synthetic_imgs = g_model(z, conds)
            predicts_f = d_model(synthetic_imgs.detach(), conds)
            loss_fake = criterion(predicts_f, fake)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            # train generator
            for _ in range(4):
                optimizer_g.zero_grad()

                z = torch.randn(batch_size, z_dim).to(device)
                synthetic_imgs = g_model(z, conds)
                predicts = d_model(synthetic_imgs, conds)
                loss_g = criterion(predicts, real)

            loss_g.backward()
            optimizer_g.step()

            print(f'epoch: {epoch}, {i}/{len(dataloader)}  loss_g: {loss_g.item():.3f}  loss_d: {loss_d.item():.3f}')
            loss_sum_g += loss_g.item()
            loss_sum_d += loss_d.item()

        # evaluate
        g_model.eval()
        d_model.eval()
        test_conds = get_test_conditions().to(device)
        fixed_z = torch.randn(len(test_conds), z_dim).to(device)
        with torch.no_grad():
            synthetic_imgs = g_model(fixed_z,test_conds)
        score = eval_model.eval(synthetic_imgs, test_conds)
        if score > best_score:
            best_score = score
            torch.save(g_model.state_dict(), os.path.join('D:/research/DLP/LAB5/models', f'epoch{epoch}_score{score:.2f}.pt'))
        print(f'avg loss_g: {loss_sum_g / len(dataloader):.3f}  avg loss_d: {loss_sum_d / len(dataloader):.3f}')
        print(f'testing score: {score:.2f}')
        print('---------------------------------------------')
        # savefig
        save_image(synthetic_imgs, os.path.join('D:/research/DLP/LAB5/results', f'epoch{epoch}.png'), nrow=8, normalize=True)

class Generater(nn.Module):
    def __init__(self, z_dim, c_dim):
        super(Generater, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.condExpand = nn.Sequential(
            nn.Linear(24, c_dim),
            nn.ReLU()
        )
        kernel_size = (4, 4)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(z_dim+c_dim, 512, kernel_size, stride=(2,2), padding=(0,0)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size, stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size, stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size, stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size, stride=(2,2), padding=(1,1)),
            nn.Tanh()
        )

    def forward(self, z, cond):
        """
        (N, z_dim+c_dim, 1, 1) >
        (N, 512, 4, 4) >
        (N, 256, 8, 8) >
        (N, 128, 16, 16) >
        (N, 64, 32, 32) >
        norm value to[-1, +1]
        """
        z = z.view(-1, self.z_dim, 1, 1)
        cond = self.condExpand(cond).view(-1, self.c_dim, 1, 1)
        out = torch.cat((z, cond), dim=1)
        out = self.conv(out)
        return out

    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(self._modules[m], nn.ConvTranspose2d) or isinstance(self._modules[m], nn.Conv2d):
                self._modules[m].weight.data.normal_(mean, std)
                self._modules[m].bias.data.zero_()

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.H, self.W, self.C = img_shape
        self.condExpand = nn.Sequential(
            nn.Linear(24, self.H*self.W*1),
            nn.LeakyReLU()
        )
        kernel_size=(4, 4)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size, stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size, stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size, stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size, stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1, kernel_size, stride=(1,1)),
            nn.Sigmoid()
        )

    def forward(self, X, cond):
        """
        (N, 4, 64, 64) >
        (N, 64, 32, 32) >
        (N, 128, 16, 16) >
        (N, 256, 8, 8) >
        (N, 512, 4, 4) >
        norm value to[0, 1]
        """
        cond = self.condExpand(cond).view(-1, 1, self.H, self.W)
        out = torch.cat((X, cond), dim=1)
        out = self.conv(out)
        out = out.view(-1)
        return out

    def weight_init(self,mean,std):
        for m in self._modules:
            if isinstance(self._modules[m], nn.ConvTranspose2d) or isinstance(self._modules[m], nn.Conv2d):
                self._modules[m].weight.data.normal_(mean, std)
                self._modules[m].bias.data.zero_()

class GetData(Dataset):
    def __init__(self,img_path):
        self.img_path = img_path
        self.max_objects = 0
        with open('D:/research/DLP/LAB5/objects.json', 'r') as file:
            self.classes = json.load(file)
        self.img_names = []
        self.img_conds = []
        with open('D:/research/DLP/LAB5/train.json', 'r') as file:
            dict = json.load(file)
            for img_name, img_cond in dict.items():
                self.img_names.append(img_name)
                self.max_objects = max(self.max_objects, len(img_cond))
                self.img_conds.append([self.classes[condition] for condition in img_cond])
        self.trans = transforms.Compose([transforms.Resize((64,64)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_names[index])).convert('RGB')
        img = self.trans(img)
        condition = self.int2onehot(self.img_conds[index])
        return img, condition

    def int2onehot(self, int_list):
        onehot = torch.zeros(len(self.classes))
        for i in int_list:
            onehot[i] = 1.
        return onehot

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    z_dim = 100
    c_dim = 200
    batch_size = 64
    epochs = 200
    img_shape = (64, 64, 3)
    lr = 0.0002
    num_workers = 2

    # load data
    dataset = GetData('D:/research/DLP/LAB5/images')
    loader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # # create generate & discriminator
    # generator = Generater(z_dim, c_dim).to(device)
    # discrimiator = Discriminator(img_shape).to(device)
    # generator.weight_init(mean=0, std=0.02)
    # discrimiator.weight_init(mean=0, std=0.02)

    # train
    # train(loader_train ,generator, discrimiator, z_dim, epochs, lr, batch_size)

    # test
    conds = get_test_conditions().to(device)
    g_model = Generator(z_dim, c_dim).to(device)
    g_model.load_state_dict(torch.load('epoch189_score0.72.pt'))

    avg_score = 0
    for _ in range(10):
        z = torch.randn(len(conds), z_dim).to(device)
        synthetic_imgs = g_model(z, conds)
        eval_model = EvaluationModel()
        score = eval_model.eval(synthetic_imgs, conds)
        print(f'score: {score:.2f}')
        avg_score += score

    save_image(synthetic_imgs, 'eval.png', nrow=8, normalize=True)
    print()
    print(f'avg score: {avg_score/10:.2f}')