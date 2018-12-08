import pandas as pd
import numpy as np
import torch
from metrics import get_metrics
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import itertools
import os
import pprint

# static constants
HYPERPARAMS = ['learning_rate', 'num_iters', 'n_h', 'n_h_adv', 'dropout_rate', 'alpha']
intermediate_metrics = False

class Model(object):
    def __init__(self, params):
        self.params = params
        self.method = self.params['method']
        self.adversarial = self.method != 'basic'
        self.num_classes = self.params['num_classes']
        self.logpath = self.params['logpath']
        self.hyperparams = self.params['hyperparams']
        self.model = self.build_model()
        self.data = self.process_data()

    def valid_hyperparam(self, i):
        return (i < 3 or i == 4 or self.adversarial)

    def get_indexes(self):
        num_models = []
        for i in range(len(HYPERPARAMS)):
            if self.valid_hyperparam(i):
                num_models.append(range(len(self.hyperparams[HYPERPARAMS[i]])))
            else:
                num_models.append([None]) # placeholder value if no such hyperparameter
        return itertools.product(*num_models)

    def get_hyperparams(self, indexes):
        hyperparams = []
        for i in range(len(indexes)):
            if self.valid_hyperparam(i):
                hyperparams.append(self.hyperparams[HYPERPARAMS[i]][indexes[i]])
            else:
                hyperparams.append(None)
        return hyperparams

    def hyperparams_to_string(self, indexes):
        res = ''
        for i in range(len(HYPERPARAMS)):
            if i > 0:
                res += '-'
            if self.valid_hyperparam(i):
                res += HYPERPARAMS[i] + '_' + str(self.hyperparams[HYPERPARAMS[i]][indexes[i]])
        return res

    def build_model(self):
        models = {}
        for indexes in self.get_indexes():
                models[indexes] = self.build_single_model(indexes)
        return models

    def build_single_model(self, indexes):
        model = dict()

        m, n = self.params['Xtrain'].shape
        m_valid, n_valid = self.params['Xvalid'].shape
        m_test, n_test = self.params['Xtest'].shape
        n_h = self.hyperparams['n_h'][indexes[2]]

        model['model'] = torch.nn.Sequential(
            torch.nn.Linear(n, n_h),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.hyperparams['dropout_rate'][indexes[4]]),
            torch.nn.Linear(n_h, 1),
            torch.nn.Sigmoid(),
        )
        model['loss_fn'] = torch.nn.BCELoss(size_average=True)
        model['optimizer'] = torch.optim.Adam(model['model'].parameters(), lr=self.hyperparams['learning_rate'][indexes[0]])

        if self.adversarial:
            n_h_adv = self.hyperparams['n_h_adv'][indexes[3]]
            if self.num_classes > 2:
                n_h_out = self.num_classes
            else:
                n_h_out = 1

            if self.method == 'parity':
                n_adv = 1
            elif self.method == 'odds' or 'opportunity':
                n_adv = 2
            else:
                raise Exception('Unknown method: {}'.format(self.method))

            model['adv_model'] = torch.nn.Sequential(
                torch.nn.Linear(n_adv, n_h_adv),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.hyperparams['dropout_rate'][indexes[4]]),
	            torch.nn.Linear(n_h_adv, n_h_out),
                torch.nn.Sigmoid(),
            )
            if (self.num_classes > 2):
                model['adv_loss_fn'] = torch.nn.CrossEntropyLoss(size_average=True)
            else:
                model['adv_loss_fn'] = torch.nn.BCELoss(size_average=True)
            model['adv_optimizer'] = torch.optim.Adam(model['adv_model'].parameters(), lr=self.hyperparams['learning_rate'][indexes[0]])

        return model

    def process_data(self):
        data = dict()
        m, n = self.params['Xtrain'].shape
        m_valid, n_valid = self.params['Xvalid'].shape
        m_test, n_test = self.params['Xtest'].shape
        n_h = self.hyperparams['n_h']

        if self.method == 'opportunity':
            data['adv_train_mask'] = self.params['ytrain'] == 1
            self.params['ztrain'] = self.params['ztrain'][data['adv_train_mask']]
            data['adv_train_mask'] = torch.ByteTensor(data['adv_train_mask'].astype(int).values.reshape(m, 1))

            data['adv_valid_mask'] = self.params['yvalid'] == 1
            self.params['zvalid'] = self.params['zvalid'][data['adv_valid_mask']]
            data['adv_valid_mask'] = torch.ByteTensor(data['adv_valid_mask'].astype(int).values.reshape(m_valid, 1))

            data['adv_test_mask'] = self.params['ytest'] == 1
            self.params['ztest'] = self.params['ztest'][data['adv_test_mask']]
            data['adv_test_mask'] = torch.ByteTensor(data['adv_test_mask'].astype(int).values.reshape(m_test, 1))

        data['Xtrain'] = Variable(torch.tensor(self.params['Xtrain'].values).float())
        data['ytrain'] = Variable(torch.tensor(self.params['ytrain'].values.reshape(m, 1)).float())
        data['Xvalid'] = Variable(torch.tensor(self.params['Xvalid'].values).float())
        data['yvalid'] = Variable(torch.tensor(self.params['yvalid'].values.reshape(m_valid,1)).float())
        data['Xtest'] = Variable(torch.tensor(self.params['Xtest'].values).float())
        data['ytest'] = Variable(torch.tensor(self.params['ytest'].values.reshape(m_test, 1)).float())
        if self.num_classes > 2:
            data['ztrain'] = Variable(torch.tensor(self.params['ztrain'].values.reshape(self.params['ztrain'].shape[0],)).long())
            data['zvalid'] = Variable(torch.tensor(self.params['zvalid'].values.reshape(self.params['zvalid'].shape[0],)).long())
            data['ztest'] = Variable(torch.tensor(self.params['ztest'].values.reshape(self.params['ztest'].shape[0],)).long())
        else:
            data['ztrain'] = Variable(torch.tensor(self.params['ztrain'].values.reshape(self.params['ztrain'].shape[0],)).float())
            data['zvalid'] = Variable(torch.tensor(self.params['zvalid'].values.reshape(self.params['zvalid'].shape[0],)).float())
            data['ztest'] = Variable(torch.tensor(self.params['ztest'].values.reshape(self.params['ztest'].shape[0],)).float())

        return data

    def train(self):
        for indexes in self.get_indexes():
            self.train_single_model(indexes)

    def load_trained_models(self):
        for indexes in self.get_indexes():
            hyperparam_values = self.hyperparams_to_string(indexes)
            modelfile = self.logpath + '-model/' + hyperparam_values + '-model.pth'
            self.model[indexes]['model'] = torch.load(modelfile)

    def create_dir(self, dirname):
        if (not os.path.exists(dirname)):
            os.makedirs(dirname)

    def train_single_model(self, indexes):
        # Load in model and data
        model = self.model[indexes]['model']
        loss_fn = self.model[indexes]['loss_fn']
        optimizer = self.model[indexes]['optimizer']
        Xtrain = self.data['Xtrain']
        Xvalid = self.data['Xvalid']
        Xtest = self.data['Xtest']
        ytrain = self.data['ytrain']
        yvalid = self.data['yvalid']
        ytest = self.data['ytest']
        ztrain = self.data['ztrain']
        zvalid = self.data['zvalid']
        ztest = self.data['ztest']
        if self.adversarial:
            adv_model = self.model[indexes]['adv_model']
            adv_loss_fn = self.model[indexes]['adv_loss_fn']
            adv_optimizer = self.model[indexes]['adv_optimizer']

        model.train()

        # Set up logging
        self.create_dir(self.logpath + '-training/')
        self.create_dir(self.logpath + '-metrics/')
        self.create_dir(self.logpath + '-model/')
        if self.adversarial:
            self.create_dir(self.logpath + '-adv/')
        hyperparam_values = self.hyperparams_to_string(indexes)
        logfile = self.logpath + '-training/' + hyperparam_values
        metrics_file = self.logpath + '-metrics/' + hyperparam_values + '-metrics.csv'
        metrics = []
        modelfile = self.logpath + '-model/' + hyperparam_values + '-model.pth'
        if self.adversarial:
            advfile = self.logpath + '-adv/' + hyperparam_values + '-adv.pth'
        writer = SummaryWriter(logfile)

        for t in range(self.hyperparams['num_iters'][indexes[1]]):
            # Forward step
            ypred_train = model(Xtrain)
            loss_train = loss_fn(ypred_train, ytrain)

            ypred_valid = model(Xvalid)
            loss_valid = loss_fn(ypred_valid, yvalid)

            ypred_test = model(Xtest)
            loss_test = loss_fn(ypred_test, ytest)

            if self.adversarial:
                if self.method == 'parity':
                    adv_input_train = ypred_train
                    adv_input_valid = ypred_valid
                    adv_input_test = ypred_test
                elif self.method == 'odds':
                    adv_input_train = torch.cat((ypred_train, ytrain), 1)
                    adv_input_valid = torch.cat((ypred_valid, yvalid), 1)
                    adv_input_test = torch.cat((ypred_test, ytest), 1)
                elif self.method == 'opportunity':
                    adv_input_train = torch.stack((torch.masked_select(ypred_train, self.data['adv_train_mask']),
                                                 torch.masked_select(ytrain, self.data['adv_train_mask'])), 1)
                    adv_input_valid = torch.stack((torch.masked_select(ypred_valid, self.data['adv_valid_mask']),
                                                 torch.masked_select(yvalid, self.data['adv_valid_mask'])), 1)
                    adv_input_test = torch.stack((torch.masked_select(ypred_test, self.data['adv_test_mask']),
                                                torch.masked_select(ytest, self.data['adv_test_mask'])), 1)

                zpred_train = adv_model(adv_input_train)
                adv_loss_train = adv_loss_fn(zpred_train, ztrain)

                zpred_valid = adv_model(adv_input_valid)
                adv_loss_valid = adv_loss_fn(zpred_valid, zvalid)

                zpred_test = adv_model(adv_input_test)
                adv_loss_test = adv_loss_fn(zpred_test, ztest)

                combined_loss_train = loss_train - self.hyperparams['alpha'][indexes[5]] * adv_loss_train
                combined_loss_valid = loss_valid - self.hyperparams['alpha'][indexes[5]] * adv_loss_valid
                combined_loss_test = loss_test - self.hyperparams['alpha'][indexes[5]] * adv_loss_test

            # Train log
            if t % 100 == 0:
                print('Iteration: {}'.format(t))
                if self.adversarial:
                    print('Predictor train loss: {:.4f}'.format(loss_train))
                    print('Predictor valid loss: {:.4f}'.format(loss_train))
                    print('Adversary train loss: {:.4f}'.format(adv_loss_train))
                    print('Adversary valid loss: {:.4f}'.format(adv_loss_valid))
                    print('Combined train loss:  {:.4f}'.format(combined_loss_train))
                    print('Combined valid loss:  {:.4f}'.format(combined_loss_valid))

                    write_log(writer, 'pred_loss_train', loss_train, t)
                    write_log(writer, 'pred_loss_valid', loss_valid, t)
                    write_log(writer, 'pred_loss_test', loss_test, t)
                    write_log(writer, 'adv_loss_train', adv_loss_train, t)
                    write_log(writer, 'adv_loss_valid', adv_loss_valid, t)
                    write_log(writer, 'adv_loss_test', adv_loss_test, t)
                    write_log(writer, 'combined_loss_train', combined_loss_train, t)
                    write_log(writer, 'combined_loss_valid', combined_loss_valid, t)
                    write_log(writer, 'combined_loss_test', combined_loss_test, t)
                else:
                    print('Train loss: {:.4f}'.format(loss_train))
                    print('Valid loss: {:.4f}'.format(loss_valid))

                    write_log(writer, 'loss_train', loss_train, t)
                    write_log(writer, 'loss_valid', loss_valid, t)
                    write_log(writer, 'loss_test', loss_test, t)

                # print('Train metrics:')
                # metrics_train = metrics.get_metrics(ypred_train.data.numpy(), ytrain.data.numpy(), ztrain.data.numpy(), self.num_classes)
                if (intermediate_metrics):
                    print('Validation metrics:')
                    metrics_valid = get_metrics(ypred_valid.data.numpy(), yvalid.data.numpy(), zvalid.data.numpy(), self.get_hyperparams(indexes), self.num_classes, 'valid_set')
                    pprint.pprint(metrics_valid)
                    metrics.append(metrics_valid) # -- NO LONGER COMPUTING INTERMEDIATE METRICS

            # Save model
            if t > 0 and t % 10000 == 0:
                torch.save(model, modelfile)
                if self.adversarial:
                    torch.save(adv_model, advfile)

            # Backward step
            if self.adversarial:
                # adv update
                adv_optimizer.zero_grad()
                adv_loss_train.backward(retain_graph=True)
                adv_optimizer.step()
                # pred update
                optimizer.zero_grad()
                combined_loss_train.backward()
            else:
                optimizer.zero_grad()
                loss_train.backward()

            optimizer.step()

        # save final model
        torch.save(model, modelfile)
        if self.adversarial:
            torch.save(adv_model, advfile)
        writer.close()

        if (intermediate_metrics):
            metrics = pd.DataFrame(metrics)
            metrics.to_csv(metrics_file)

    def eval(self):
        evalfile = self.logpath + '-eval.csv'
        test_metrics = []
        for indexes in self.get_indexes():
            test_metrics.append(self.eval_single_model(indexes))

        pd.concat(test_metrics).to_csv(evalfile)

    def eval_single_model(self, indexes):
        model = self.model[indexes]['model']
        # loss_fn = self.model[indexes]['loss_fn']
        # optimizer = self.model[indexes]['optimizer']
        Xtrain = self.data['Xtrain']
        Xvalid = self.data['Xvalid']
        Xtest = self.data['Xtest']
        ytrain = self.data['ytrain']
        yvalid = self.data['yvalid']
        ytest = self.data['ytest']
        ztrain = self.data['ztrain']
        zvalid = self.data['zvalid']
        ztest = self.data['ztest']

        model.eval()
        ypred_valid = model(Xvalid)
        if self.adversarial:
            adv_model = self.model[indexes]['adv_model']
            adv_model.eval()

            if self.method == 'parity':
                adv_input_valid = ypred_valid
            elif self.method == 'odds':
                adv_input_valid = torch.cat((ypred_valid, yvalid), 1)
            elif self.method == 'opportunity':
                adv_input_valid = torch.cat((torch.masked_select(ypred_valid, self.data['adv_valid_mask']),
                                             torch.masked_select(yvalid, self.data['adv_valid_mask'])), 1)
            zpred_valid = adv_model(adv_input_valid)
            metrics_valid = pd.DataFrame(get_metrics(ypred_valid.data.numpy(), yvalid.data.numpy(), zvalid.data.numpy(), self.get_hyperparams(indexes), k=self.num_classes, evaluation_file='valid_set', zpred=zpred_valid.data.numpy()), index=[0])
        else:
            metrics_valid = pd.DataFrame(get_metrics(ypred_valid.data.numpy(), yvalid.data.numpy(), zvalid.data.numpy(), self.get_hyperparams(indexes), k=self.num_classes, evaluation_file='valid_set'), index=[0])
        print
        print('Final test metrics for model with ' + self.hyperparams_to_string(indexes) + ' on validation:')
        pprint.pprint(metrics_valid)

        ypred_test = model(Xtest)
        if self.adversarial:
            if self.method == 'parity':
                adv_input_test = ypred_test
            elif self.method == 'odds':
                adv_input_test = torch.cat((ypred_test, ytest), 1)
            elif self.method == 'opportunity':
                adv_input_test = torch.cat((torch.masked_select(ypred_test, self.data['adv_test_mask']),
                                            torch.masked_select(ytest, self.data['adv_test_mask'])), 1)
            zpred_test = adv_model(adv_input_test)
            metrics_test = pd.DataFrame(get_metrics(ypred_test.data.numpy(), ytest.data.numpy(), ztest.data.numpy(), self.get_hyperparams(indexes), k=self.num_classes, evaluation_file='test_set', zpred=zpred_test.data.numpy()), index=[0])
        else:
            metrics_test = pd.DataFrame(get_metrics(ypred_test.data.numpy(), ytest.data.numpy(), ztest.data.numpy(), self.get_hyperparams(indexes), k=self.num_classes, evaluation_file='test_set'), index=[0])
        print
        print('Final test metrics for model with ' + self.hyperparams_to_string(indexes) + ' on test:')
        pprint.pprint(metrics_test)
        return pd.concat([metrics_valid, metrics_test])



def write_log(writer, key, loss, iter):
    writer.add_scalar(key, loss.item(), iter)


def write_log_array(writer, key, array, iter):
    writer.add_text(key, np.array_str(array), iter)
