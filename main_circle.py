'''add arg parse with input sz options'''
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, output_size)
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.fc1.weight.data)
        self.fc1.bias.data.zero_()
        nn.init.xavier_uniform_(self.fc2.weight.data)
        self.fc2.bias.data.zero_()

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        #x = torch.sigmoid(self.fc3(x))
        # = torch.sigmoid(self.fc4(x))
        #x = torch.sigmoid(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
        return x


def randCircleGen(m):
    y = np.round(np.random.rand(m))
    idx = y.astype(int)
    r = np.array([10., 5.])
    phi = 2 * np.pi * np.random.rand(m).astype(float)
    x = np.zeros((m, 2))
    x[:, 0] = r[idx]*np.cos(phi)
    x[:, 1] = r[idx]*np.sin(phi)
    x = Variable(torch.DoubleTensor(torch.from_numpy(x)), requires_grad=False)
    y = Variable(torch.DoubleTensor(torch.from_numpy(y)), requires_grad=False)
    return x, y


def train():
    n_epoch = 10000
    batch_size = 100
    model = Net(2, 4, 1).double()
    criterion = nn.BCELoss(reduction='sum').double()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(n_epoch):
        optimizer.zero_grad()
        x, y = randCircleGen(batch_size)
        y_est = model(x)
        loss = criterion(y_est[:, 0], y)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print('====> Epoch: {} loss: {:.4f}'.format(epoch, loss))

    if not os.path.isdir('model'):
        os.mkdir('model')
    torch.save(model.state_dict(), 'model/circle.pt')
    return model


def test(model):
    test_sz = 100
    x_test, y_test = randCircleGen(test_sz)
    y_est = model(x_test)
    x_test = x_test.data.numpy()
    y_test = y_test.data.numpy().reshape(test_sz,)

    y_est = y_est.data.numpy().reshape(test_sz,)
    y_est = (y_est > 0.5)*1.
    acc = np.sum(y_est-y_test == 0) / np.double(test_sz)
    plt.plot(x_test[y_test.astype(int) == 0, 0], x_test[y_test.astype(int) == 0, 1], 'b.')
    plt.plot(x_test[y_test.astype(int) == 1, 0], x_test[y_test.astype(int) == 1, 1], 'r.')
    plt.plot(x_test[y_est.astype(int) == 0, 0], x_test[y_est.astype(int) == 0, 1], 'co')
    plt.plot(x_test[y_est.astype(int) == 1, 0], x_test[y_est.astype(int) == 1, 1], 'gx')
    plt.grid(True)
    plt.show()

    print("accuracy={:.2f}%".format(acc*100))


if __name__ == "__main__":
    model = train()
    # model = Net(2, 4, 1).double()
    #model.load_state_dict(torch.load('model/xor.pt'))
    #model.eval()
    test(model)