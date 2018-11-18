import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter


class Model(object):
    def __init__(self, params):
        self.params = params
        self.method = self.params['method']
        self.adversarial = self.method != 'basic'
        self.logpath = self.params['logpath']
        self.hyperparams = self.params['hyperparams']
        self.model = self.build_model()
        self.data = self.process_data()

    def build_model(self):
        model = dict()

        m, n = self.params['Xtrain'].shape
        m_test, n_test = self.params['Xtest'].shape
        n_h = self.hyperparams['n_h']

        model['model'] = torch.nn.Sequential(
            torch.nn.Linear(n, n_h),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.hyperparams['dropout_rate']),
            torch.nn.Linear(n_h, 1),
            torch.nn.Sigmoid(),
        )
        model['loss_fn'] = torch.nn.BCELoss(size_average=True)
        model['optimizer'] = torch.optim.Adam(model['model'].parameters(), lr=self.hyperparams['learning_rate'])

        if self.adversarial:
            n_h_adv = self.hyperparams['n_h_adv']

            if self.method == 'parity':
                n_adv = 1
            elif self.method == 'odds':
                n_adv = 2
            else:
                raise Exception('Unknown method: {}'.format(self.method))

            model['adv_model'] = torch.nn.Sequential(
                torch.nn.Linear(n_adv, n_h_adv),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.hyperparams['dropout_rate']),
                torch.nn.Linear(n_h_adv, 1),
                torch.nn.Sigmoid(),
            )
            model['adv_loss_fn'] = torch.nn.BCELoss(size_average=True)
            model['adv_optimizer'] = torch.optim.Adam(model['adv_model'].parameters(), lr=self.hyperparams['learning_rate'])

        return model

    def process_data(self):
        data = dict()
        m, n = self.params['Xtrain'].shape
        m_test, n_test = self.params['Xtest'].shape
        n_h = self.hyperparams['n_h']

        data['Xtrain'] = Variable(torch.tensor(self.params['Xtrain'].values).float())
        data['ytrain'] = Variable(torch.tensor(self.params['ytrain'].values.reshape(m, 1)).float())
        data['Xtest'] = Variable(torch.tensor(self.params['Xtest'].values).float())
        data['ytest'] = Variable(torch.tensor(self.params['ytest'].values.reshape(m_test, 1)).float())
        if self.adversarial:
            data['ztrain'] = Variable(torch.tensor(self.params['ztrain'].values.reshape(m, 1)).float())
            data['ztest'] = Variable(torch.tensor(self.params['ztest'].values.reshape(m_test, 1)).float())

        return data

    def train(self):
        # Load in model and data
        model = self.model['model']
        loss_fn = self.model['loss_fn']
        optimizer = self.model['optimizer']
        Xtrain = self.data['Xtrain']
        Xtest = self.data['Xtest']
        ytrain = self.data['ytrain']
        ytest = self.data['ytest']
        if self.adversarial:
            adv_model = self.model['adv_model']
            adv_loss_fn = self.model['adv_loss_fn']
            adv_optimizer = self.model['adv_optimizer']
            ztrain = self.data['ztrain']
            ztest = self.data['ztest']

        model.train()

        # Set up logging
        logfile = self.logpath + '-training'
        modelfile = self.logpath + '-model.pth'
        if self.adversarial:
            advfile = self.logpath + '-adv.pth'
        writer = SummaryWriter(logfile)

        for t in range(self.hyperparams['num_iters']):
            # Forward step
            ypred_train = model(Xtrain)
            loss_train = loss_fn(ypred_train, ytrain)

            ypred_test = model(Xtest)
            loss_test = loss_fn(ypred_test, ytest)

            if self.adversarial:
                if self.method == 'parity':
                    adv_input_train = ypred_train
                    adv_input_test = ypred_test
                elif self.method == 'odds':
                    adv_input_train = torch.cat((ypred_train, ytrain), 1)
                    adv_input_test = torch.cat((ypred_test, ytest), 1)

                zpred_train = adv_model(adv_input_train)
                adv_loss_train = adv_loss_fn(zpred_train, ztrain)

                zpred_test = adv_model(adv_input_test)
                adv_loss_test = adv_loss_fn(zpred_test, ztest)

                combined_loss_train = loss_train - self.hyperparams['alpha'] * adv_loss_train
                combined_loss_test = loss_test - self.hyperparams['alpha'] * adv_loss_test

            # Train log
            if t % 100 == 0:
                print('Iteration: {}'.format(t))
                if self.adversarial:
                    print('Predictor train loss: {:.4f}'.format(loss_train))
                    print('Adversary train loss: {:.4f}'.format(adv_loss_train))
                    print('Combined train loss:  {:.4f}'.format(combined_loss_train))

                    write_log(writer, 'pred_loss_train', loss_train, t)
                    write_log(writer, 'pred_loss_test', loss_test, t)
                    write_log(writer, 'adv_loss_train', adv_loss_train, t)
                    write_log(writer, 'adv_loss_test', adv_loss_test, t)
                    write_log(writer, 'combined_loss_train', combined_loss_train, t)
                    write_log(writer, 'combined_loss_test', combined_loss_test, t)
                else:
                    print('Train loss: {:.4f}'.format(loss_train))

                    write_log(writer, 'loss_train', loss_train, t)
                    write_log(writer, 'loss_test', loss_test, t)
            # Save model
            if t > 0 and t % 10000 == 0:
                torch.save(model, modelfile)
                if self.adversarial:
                    torch.save(adv_model, advfile)

            # Backward step
            if self.adversarial:
                combined_loss_train.backward()
            else:
                loss_train.backward()

            optimizer.step()
            optimizer.zero_grad()

        # save final model
        torch.save(model, modelfile)
        if self.adversarial:
            torch.save(adv_model, advfile)
        writer.close()

    def eval(self):
        model = self.model['model']
        loss_fn = self.model['loss_fn']
        optimizer = self.model['optimizer']
        Xtrain = self.data['Xtrain']
        Xtest = self.data['Xtest']
        ytrain = self.data['ytrain']
        ytest = self.data['ytest']
        if self.adversarial:
            adv_model = self.model['adv_model']
            adv_loss_fn = self.model['adv_loss_fn']
            adv_optimizer = self.model['adv_optimizer']
            ztrain = self.data['ztrain']
            ztest = self.data['ztest']

        model.eval()

        evalfile = self.logpath + '-eval.csv'

        ypred_test = model(Xtest)
        loss_test = loss_fn(ypred_test, ytest)
        print('Test loss {}'.format(loss_test))

        #TODO: Create dictionary of evaluation results/metrics and save to evalfile.


def write_log(writer, key, loss, iter):
    writer.add_scalar(key, loss.item(), iter)
