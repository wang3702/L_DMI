# training on Clothing1M

import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn as nn
from tqdm import tqdm
from model import *
from data import *
import torch.nn.functional as F
import argparse
import os
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str,help="GPU id")
parser.add_argument('-F',type=str,help="Dataset path")
parser.add_argument('-R',type=str,default=None,help="Recovering label path")
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--num_workers', type=int, default=8, help='number of data loading workers')
args = parser.parse_args()
params = vars(args)
#torch.cuda.set_device(args.device)
os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']
batch_size = 256
num_classes = 14

CE = nn.CrossEntropyLoss().cuda()

data_root = params['F']
from Process_Clothing1M import Process_Clothing1M
data = Process_Clothing1M(data_root)
#train_dataset = Clothing(root=data_root, img_transform=train_transform, train=True, valid=False, test=False)
#train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = 32)
#valid_dataset = Clothing(root=data_root, img_transform=test_transform, train=False, valid=True, test=False)
#valid_loader = torch.utils.data.DataLoader(dataset = valid_dataset, batch_size = batch_size, shuffle = False, num_workers = 32)
#test_dataset = Clothing(root=data_root, img_transform=test_transform, train=False, valid=False, test=True)
#test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False, num_workers = 32)


def prepare_Clothing_Dataloader(data,params,num_classes,require_transform=True):
    if True:
        import torchvision.transforms as transforms
        from ops.Transform_ops import RandomFlip, RandomPadandCrop, Resize, CenterCrop
        transform_train = transforms.Compose([
            Resize((256, 256)),
            CenterCrop(224),
            RandomPadandCrop(224, default_pad=32, channel_first=False),
            RandomFlip(channel_first=False),
        ])
        from DataSet import Simple_Clothing_Dataset1
        traindataset=Simple_Clothing_Dataset1(data.train_path,transform_pre=transform_train,require_index=True)
        train_dataloader = torch.utils.data.DataLoader(traindataset, batch_size=params['batch_size'],
                                                   shuffle=True, num_workers=int(params['num_workers']), drop_last=False,
                                                   pin_memory=False)
        valid_dataset = Simple_Clothing_Dataset1(data.val_path,transform_pre=transform_train)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=params['batch_size'],
                                                       shuffle=True, num_workers=int(params['num_workers']),
                                                       drop_last=True,
                                                       pin_memory=True)

        test_dataset = Simple_Clothing_Dataset1(data.test_path,transform_pre=transform_train)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=params['batch_size'],
                                                      shuffle=True, num_workers=int(params['num_workers']),
                                                      drop_last=True,
                                                      pin_memory=True)

    return train_dataloader,valid_dataloader,test_dataloader
train_loader, valid_loader, test_loader =prepare_Clothing_Dataloader(data,params,num_classes)
#add recovering label
if params['R'] is not None:  # when it's none, used for comparison
    Rever_Label_Path = os.path.abspath(params['R'])
    Recover_Label = np.load(Rever_Label_Path)
    Recover_Label = np.argmax(Recover_Label, axis=1)
    train_loader.dataset.Rebuild_Label(Recover_Label)
def train(train_loader, model, optimizer, criterion=CE):
    model.train()

    for i, (idx, input, target) in enumerate(tqdm(train_loader)):
        if idx.size(0) != batch_size:
            break
        input = torch.autograd.Variable(input.cuda())
        target = torch.autograd.Variable(target.cuda())

        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (idx, input, target) in enumerate(tqdm(test_loader)):
            input = torch.Tensor(input).cuda()
            target = torch.autograd.Variable(target).cuda()
            total += target.size(0)
            output = model(input)
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total

    return accuracy


def DMI_loss(output, target):
    outputs = F.softmax(output, dim=1)
    targets = target.reshape(target.size(0), 1)
    y_onehot = torch.FloatTensor(target.size(0), num_classes)
    y_onehot.zero_()
    targets = targets.cpu()
    y_onehot.scatter_(1, targets, 1)
    y_onehot = y_onehot.transpose(0, 1).cuda()
    #mat = y_onehot @ outputs
    mat=torch.mm(y_onehot, outputs)
    return -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)


def main_ce():
    model_ce = resnet50().cuda()
    model_ce = nn.DataParallel(model_ce, device_ids=None)
    best_ce_acc = 0

    for epoch in range(10):
        print("epoch=", epoch)
        learning_rate = 0.01
        if epoch >= 5:
            learning_rate = 0.001

        optimizer_ce = torch.optim.SGD(model_ce.parameters(), momentum=0.9, weight_decay=1e-3, lr=learning_rate)

        print("traning model_ce...")
        train(train_loader=train_loader, model=model_ce, optimizer=optimizer_ce)
        print("validating model_ce...")
        valid_acc = test(model=model_ce, test_loader=valid_loader)
        print('valid_acc=', valid_acc)
        if valid_acc > best_ce_acc:
            best_ce_acc = valid_acc
            torch.save(model_ce, './model_ce')

    model_ce = torch.load('./model_ce')
    test_acc = test(model=model_ce, test_loader=test_loader)
    print('model_ce_final_test_acc=', test_acc)

def main_dmi():
    model_dmi = torch.load('./model_ce')
    #model_dmi = model_dmi.cuda()
    #model_dmi = nn.DataParallel(model_dmi, device_ids=None)
    best_acc = 0

    for epoch in range(10):
        print("epoch=", epoch)
        learning_rate = 1e-6
        if epoch >= 5:
            learning_rate = 5e-7

        optimizer_dmi = torch.optim.SGD(model_dmi.parameters(), momentum=0.9, weight_decay=1e-3, lr=learning_rate)

        print("traning model_dmi...")
        train(train_loader=train_loader, model=model_dmi, optimizer=optimizer_dmi, criterion=DMI_loss)
        print("validating model_dmi...")
        valid_acc = test(model=model_dmi, test_loader=valid_loader)
        print('valid_acc=', valid_acc)
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model_dmi, './model_dmi')

    model_ce = torch.load('./model_dmi')
    test_acc = test(model=model_ce, test_loader=test_loader)
    print('model_dmi_final_test_acc=', test_acc)


if __name__ == '__main__':
    main_ce()
    main_dmi()
