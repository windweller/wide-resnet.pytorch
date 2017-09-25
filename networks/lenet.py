import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np

# def conv_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         init.xavier_uniform(m.weight, gain=np.sqrt(2))
#         init.constant(m.bias, 0)

class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        self.pre_softmax_out = F.relu(self.fc2(out))  # so this is true ANP...norm penalty after activation hmmm
        out = self.fc3(self.pre_softmax_out)

        return(out)

if __name__ == '__main__':
    net = LeNet(10)
    x = torch.randn(2, 3, 32, 32).type(torch.FloatTensor)

    x = Variable(x)
    y_pred = net.forward(x)

    # fast way to generate labels:
    # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/4

    batch_size = 2
    nb_digits = 10

    y = torch.LongTensor(batch_size, 1).random_() % nb_digits
    y_onehot = torch.FloatTensor(batch_size, nb_digits)

    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)

    y_onehot = Variable(y_onehot)

    print "norm of pre_softmax_out: "
    print net.pre_softmax_out.norm()

    loss = (y_pred - y_onehot).pow(2).sum()

    print "loss: "
    print loss.data

    combined_loss = loss + net.pre_softmax_out.norm()

    # However, can't do this step anymore, grad is only there if needed
    # https://discuss.pytorch.org/t/problem-on-variable-grad-data/957/5

    # SGD anything from optim package does this
    # optimizer.zero_grad()
    # params = net.parameters()
    # for p in params:
    #     p.grad.data.zero_()

    combined_loss.backward()

    params = net.parameters()
    for p in params:
        print p.grad.data.shape