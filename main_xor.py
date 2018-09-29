'''add arg parse with input sz options'''
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import os


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.fc1.weight.data)
        self.fc1.bias.data.zero_()
        nn.init.xavier_uniform_(self.fc2.weight.data)
        self.fc2.bias.data.zero_()

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


def randXorGen(m, n):
    x = np.round(np.random.rand(m, 2)).astype(float)
    y = (x[:, 0] + x[:, 1]) % 2
    x = Variable(torch.DoubleTensor(torch.from_numpy(x)), requires_grad=False)
    y = Variable(torch.DoubleTensor(torch.from_numpy(y)), requires_grad=False)
    return x, y


def train():
    n_epoch = 10000
    batch_size = 100
    model = Net(2, 2, 1).double()
    criterion = nn.BCELoss(reduction='sum').double()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(n_epoch):
        optimizer.zero_grad()
        x, y = randXorGen(batch_size, 1)
        y_est = model(x)
        loss = criterion(y_est[:, 0], y)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print('====> Epoch: {} loss: {:.4f}'.format(epoch, loss))

    if not os.path.isdir('model'):
        os.mkdir('model')
    torch.save(model.state_dict(), 'model/xor.pt')
    return model


def test(model):
    test_sz = 100
    x_test, y_test = randXorGen(test_sz, 1)
    y_est = model(x_test)
    y_test = y_test.data.numpy().reshape(test_sz,)
    y_est = y_est.data.numpy().reshape(test_sz,)
    y_est = (y_est > 0.5)*1.
    err = np.sum(y_est-y_test == 0) / np.double(test_sz)
    print(err)


if __name__ == "__main__":
    model = train()
    # model = Net(2, 4, 1).double()
    #model.load_state_dict(torch.load('model/xor.pt'))
    #model.eval()
    test(model)