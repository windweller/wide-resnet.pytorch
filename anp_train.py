"""
TODO: test out grabbing PyTorch variables
and writing custom loss functions
"""

# Note: this only runs on 0.12.1 PyTorch

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import datetime
import logging

from networks import *
from torch.autograd import Variable

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--seed', default=123, type=int, help='fix the random seed, so it is comparable')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
parser.add_argument('--anp', action='store_true', help='turn on activation norm penalty')
parser.add_argument('--beta', default=0.0002, type=float, help='turn on activation norm penalty')
parser.add_argument('--anp_pos', default="last", help='run on the position of ANP: first|1|2|3|relu|last', type=str)
parser.add_argument('--noweight_decay', '-nwd', action='store_true', help='whether to turn off weight decay')
parser.add_argument('--checkpoint', default='checkpoint', type=str, help='input the checkpoint folder name')
parser.add_argument('--run_dir', default='sandbox', type=str, help='where the model should be saved')
args = parser.parse_args()

torch.manual_seed(args.seed)

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)

best_acc = 0
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

# Open a file for Logger to save
if not os.path.exists(args.run_dir):
    os.makedirs(args.run_dir)
file_handler = logging.FileHandler("{0}/log.txt".format(args.run_dir))
logger.addHandler(file_handler)

# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])  # meanstd transformation

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

if (args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 10
elif (args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 100

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


# Return network & file name
def getNetwork(args):
    # if (args.net_type == 'lenet'):
    #     net = LeNet(num_classes)
    #     file_name = 'lenet'
    # elif (args.net_type == 'vggnet'):
    #     net = VGG(args.depth, num_classes)
    #     file_name = 'vgg-' + str(args.depth)
    if (args.net_type == 'resnet'):
        net = ResNet(args.depth, num_classes)
        file_name = 'resnet-' + str(args.depth)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes)
        file_name = 'wide-resnet-' + str(args.depth) + 'x' + str(args.widen_factor)
    else:
        print('Error : Network should be either [ResNet / Wide_ResNet')
        sys.exit(0)

    return net, file_name


# Test only option
if (args.testOnly):
    logger.info('\n[Test Phase] : Model setup')
    assert os.path.isdir(args.checkpoint), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./' + args.run_dir + '/' +args.checkpoint+'/' + args.dataset + os.sep + file_name + '.t7')
    net = checkpoint['net']

    if use_cuda:
        net.cuda()
        # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model
    acc = 100. * correct / total
    logger.info("| Test Result\tAcc@1: %.2f%%" % (acc))

    sys.exit(0)

# Model
print('\n[Phase 2] : Model setup')
if args.resume:
    # Load checkpoint
    logger.info('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./' + args.checkpoint +'/' + args.dataset + os.sep + file_name + '.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    logger.info('| Building net type [' + args.net_type + ']...')
    net, file_name = getNetwork(args)
    net.apply(conv_init)

if use_cuda:
    net.cuda()
    # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

def kl_norm(output):
    # x: (batch_size x output_dim)
    return output.pow(2).sum() / 2.

# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    # TODO for ANP: activation norm penalty
    # we use per-parameter option for optim package: http://pytorch.org/docs/master/optim.html
    # you can grab parameters from each layer...
    if args.noweight_decay:
        wd = 0
    else:
        wd = 5e-4
    optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=0.9, weight_decay=wd)

    logger.info('\n=> Training Epoch #%d, LR=%.4f' % (epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)  # Forward Propagation
        loss = criterion(outputs, targets)  # Loss

        if args.anp:
            # add ANP loss
            # "first|1|2|3|last"
            if args.anp_pos == "first":
                ixh = kl_norm(net.first_conv_out) * args.beta
            elif args.anp_pos == "1":
                ixh = kl_norm(net.block1_out) * args.beta
            elif args.anp_pos == "2":
                ixh = kl_norm(net.block2_out) * args.beta
            elif args.anp_pos == "3":
                ixh = kl_norm(net.block3_out) * args.beta
            elif args.anp_pos == "relu":
                ixh = kl_norm(net.relu_out) * args.beta
            elif args.anp_pos == "last":  # default
                ixh = kl_norm(net.pre_softmax_out) * args.beta
            else:
                print('Error : choose anp_pos from first|1|2|3|relu|last')
                sys.exit(0)

        loss += ixh
        loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%% ixh: %.3f%%'
                         % (epoch, num_epochs, batch_idx + 1,
                            (len(trainset) // batch_size) + 1, loss.data[0], 100. * correct / total, ixh))
        sys.stdout.flush()

    # end of training iteration
    logger.info('|Train Epoch [%3d/%3d] Final Iter \t\tLoss: %.4f Acc@1: %.3f%% ixh: %.3f%%'
                         % (epoch, num_epochs, loss.data[0], 100. * correct / total, ixh))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model
    acc = 100. * correct / total
    logger.info("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" % (epoch, loss.data[0], acc))

    if acc > best_acc:
        logger.info('| Saving Best model...\t\t\tTop1 = %.2f%%' % (acc))
        state = {
            'net': net, # net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(args.checkpoint):
            os.mkdir(args.checkpoint)
        save_point = './' + args.checkpoint + "/" + args.dataset + os.sep
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point + file_name + '.t7')
        best_acc = acc


print('\n[Phase 3] : Training model')
logger.info('| Training Epochs = ' + str(num_epochs))
logger.info('| Initial Learning Rate = ' + str(args.lr))
logger.info('| Optimizer = ' + str(optim_type))

elapsed_time = 0
for epoch in range(start_epoch, start_epoch + num_epochs):
    start_time = time.time()

    train(epoch)
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    logger.info('| Elapsed time : %d:%02d:%02d' % (cf.get_hms(elapsed_time)))

logger.info('\n[Phase 4] : Testing model')
logger.info('* Test results : Acc@1 = %.2f%%' % (best_acc))
