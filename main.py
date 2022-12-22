import torch
from torch.utils import data
from dataloader import *
from VGGSelfModel import *
from DenseSelfModel import *
from DenseModel_work2 import *
from Baseline import *
from loss import *
from utils import *
import time
from tqdm import tqdm
import random
import os
import gc
from ablation import *
from baseline_work2 import *


model = DenseModel_work2()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print('Using', torch.cuda.device_count(), 'GPUs')
    model = nn.DataParallel(model)
model.to(device)
def loss_function(sp, label, fixation):
    loss = kldiv(sp, label) - cc(sp, label)
    return loss

dataset_dir = "/home/dataset/SALICON_wzq/"
train_img_dir = dataset_dir + "images/train/"
train_gt_dir = dataset_dir + "maps/train/"
train_fix_dir = dataset_dir + "fixations/train/"

val_img_dir = dataset_dir + "images/val/"
val_gt_dir = dataset_dir + "maps/val/"
val_fix_dir = dataset_dir + "fixations/val/"

train_img_ids = [nm.split(".")[0] for nm in os.listdir(train_img_dir)]
val_img_ids = [nm.split(".")[0] for nm in os.listdir(val_img_dir)]

#SALICON
train_dataset = SaliconDataset(train_img_dir, train_gt_dir, train_fix_dir, train_img_ids)
val_dataset = SaliconDataset(val_img_dir, val_gt_dir, val_fix_dir, val_img_ids)

dataloader_tra = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True)
dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

def train(model, optimizer, dataloader, device, epoch):
    model.train()
    loss_tra = 0.0
    for image, label_fine, fixation_fine in tqdm(dataloader):
        image = image.to(device)
        label_fine = label_fine.to(device)
        fixation_fine = fixation_fine.to(device)
        optimizer.zero_grad()
        pred = model(image)
        loss = loss_function(pred, label_fine, fixation_fine)
        loss.backward()
        optimizer.step()
        loss_tra += loss.item()
    tqdm.write('Epoch: {:d} | loss_tra_avg: {:.5f}'.format(epoch, loss_tra / len(dataloader)))

def validate(model, dataloader, device, epoch):
    model.eval()
    loss_val = 0.0
    NSS1 = 0.0
    KLD1 = 0.0
    SIM1 = 0.0
    CC1 = 0.0
    IG = 0.0
    AUC1= 0.0

    for image, label_fine, fixation_fine in tqdm(dataloader):
        image = image.to(device)
        label_fine = label_fine.to(device)
        fixation_fine = fixation_fine.to(device)
        pred = model(image)
        blur_map = pred.cpu().squeeze(0).clone().numpy()
        pred1 = torch.FloatTensor(blur(blur_map)).unsqueeze(0).to(device)
        loss = loss_function(pred1, label_fine, fixation_fine)
        loss_NSS1 = nss(pred1, fixation_fine)
        loss_KLD1 = kldiv(pred1, label_fine)
        loss_SIM1 = similarity(pred1, label_fine)
        loss_CC1 = cc(pred1, label_fine)
        loss_AUC1 = auc_judd(pred1, fixation_fine)
        loss_val += loss.item()
        NSS1 += loss_NSS1.item()
        KLD1 += loss_KLD1.item()
        SIM1 += loss_SIM1.item()
        CC1 += loss_CC1.item()
        AUC1 += loss_AUC1.item()
    tqdm.write('Epoch: {:d} | loss_val_avg:{:.5f} NSS1:{:.4f} KLD1:{:.4f} SIM1:{:.4f}'
               ' CC1:{:.4f} IG:{:.4f} AUC1:{:.4f}'.format(epoch, loss_val / len(dataloader),
                                                                       NSS1 / len(dataloader),
                                                                       KLD1 / len(dataloader), SIM1 / len(dataloader),
                                                                       CC1 / len(dataloader), IG / len(dataloader),
                                                                       AUC1 / len(dataloader)))
    return loss_val / len(dataloader)

params = model.parameters()
num_params = 0
for p in model.parameters():
    num_params += p.numel()
print('Model Structure')
print(model)
print("The number of parameters: {}".format(num_params))
LR = 0.001
epoch_size = 3
for epoch in range(10):
    torch.cuda.empty_cache()
    if epoch % epoch_size == 0:
        LR = LR * 0.1
        print('current learning rate: ', LR)
        if epoch != 0:
            model_dict = model.state_dict()
            pretrained_dict = torch.load(pth_dir)
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    train(model, optimizer, dataloader_tra, device, epoch)
    with torch.no_grad():
        loss_val = validate(model, dataloader_val, device, epoch)
        if epoch == 0:
            best_val = loss_val
        if best_val >= loss_val:
            best_val = loss_val
            print('SAVE weights in Epoch {:d}'.format(epoch))
            torch.save(model.state_dict(),
                       '/home/weights/3D_{}.pth'.format(epoch))
            pth_dir = '/home/weights/3D_{}.pth'.format(epoch)


