import torchvision.transforms as transforms
import os
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from sicap_ffc.dataset import SicapDataset
from sicap_ffc.model import FFCResNet
from sicap_ffc.utils import *

DATASET_LOC = '../SiCAPv2/dataset/path'
TRAIN_FILE = 'partition/Test/Train.xlsx'
TEST_FILE = 'partition/Test/Test.xlsx'
IMAGES_PATH = 'images'
PRELOAD = True
EPOCH = 90
BATCH_SIZE = 32

def main():
    composed = transforms.Compose([transforms.Resize((224,224))])

    train_dataset = SicapDataset(os.path.join(DATASET_LOC, TRAIN_FILE), os.path.join(DATASET_LOC, IMAGES_PATH), transform = composed, preload=PRELOAD)
    test_dataset = SicapDataset(os.path.join(DATASET_LOC, TEST_FILE), os.path.join(DATASET_LOC, IMAGES_PATH), transform = composed, preload=PRELOAD)
    
    model = FFCResNet(ratio=0.5, lfu=True, use_se=False)
    model = model.cuda()

    start_epoch = 0
    end_epoch = EPOCH
    lr = 0.1
    momentum = 0.9
    weight_decay = 1e-4
    batch_size = BATCH_SIZE

    #add weights for each sample

    y_train = train_dataset.y
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    unique_labels = np.unique(y_train)
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[np.where(unique_labels == t)] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight.flatten())
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, sampler=sampler)

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True)

    tf_writer = SummaryWriter(log_dir="log")

    for epoch in range(start_epoch, end_epoch):
        train(train_loader, model, criterion, optimizer, epoch, tf_writer)

        acc1 = validate(val_loader, model, criterion, tf_writer, epoch=epoch)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'ffc_resnet50',
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)

if __name__ == '__main__':
    main()

