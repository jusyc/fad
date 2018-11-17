import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter


class FAD(object):
    def __init__(self, params):
        self.params = params
        self.hyperparams = self.params['hyperparams']

    def train(self):
        m, n = self.params['Xtrain'].shape
        m_test, n_test = self.params['Xtest'].shape
        n_h = self.hyperparams['n_h']
        Xtrain = Variable(torch.tensor(self.params['Xtrain'].values).float())
        ytrain = Variable(torch.tensor(self.params['ytrain'].values.reshape(m, 1)).float())
        Xtest = Variable(torch.tensor(self.params['Xtest'].values).float())
        ytest = Variable(torch.tensor(self.params['ytest'].values.reshape(m_test, 1)).float())

        model = torch.nn.Sequential(
            torch.nn.Linear(n, n_h),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.hyperparams['dropout_rate']),
            torch.nn.Linear(n_h, 1),
            torch.nn.Sigmoid(),
        )

        loss_fn = torch.nn.BCELoss(size_average=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.hyperparams['learning_rate'])

        for t in range(self.hyperparams['num_iters']):
            ypred_train = model(Xtrain)
            loss = loss_fn(ypred_train, ytrain)

            ypred_test = model(Xtest)
            loss_test = loss_fn(ypred_test, ytest)

            if t % 100 == 0:
                print('Iteration: {}'.format(t))
                print('Train loss: {}'.format(loss))
                print('Test loss: {}'.format(loss_test))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        ypred_test = model(Xtest)
        loss_test = loss_fn(ypred_test, ytest)
        print('Test loss {}'.format(loss_test))
