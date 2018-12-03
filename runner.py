import argparse
import json
import os
import pandas as pd
from model import Model


class Runner(object):
    '''Runs experiments from JSON config.'''
    def __init__(self):
        args = self.get_parser().parse_args()
        config_file = args.config
        self.evalonly = args.evalonly
        self.unpack_config(config_file)
        self.load_data()
        self.train_params = self.build_params()

    def run(self):
        model = Model(self.train_params)
        if self.evalonly:
            model.load_trained_models()
        else:
            model.train()
        model.eval()

    def get_parser(self):
        parser = argparse.ArgumentParser(description='Run FAD experiments')
        parser.add_argument('config', help='JSON config filename')
        parser.add_argument('--evalonly', action='store_true', help='Do not train models, just load and evaluate.')
        return parser

    def unpack_config(self, config_file):
        mypath = os.path.abspath(os.path.dirname(__file__))
        config = json.load(open(os.path.join(mypath, config_file), 'r'))
        print(config)
        self.logpath = os.path.join(mypath, config_file[:-5].replace('experiments', 'logs'))
        self.Xtrain_file = os.path.join(mypath, config['Xtrain'])
        self.ytrain_file = os.path.join(mypath, config['ytrain'])
        self.Xvalid_file = os.path.join(mypath, config['Xvalid'])
        self.yvalid_file = os.path.join(mypath, config['yvalid'])
        self.Xtest_file = os.path.join(mypath, config['Xtest'])
        self.ytest_file = os.path.join(mypath, config['ytest'])
        self.ztrain_file = os.path.join(mypath, config['ztrain'])
        self.zvalid_file = os.path.join(mypath, config['zvalid'])
        self.ztest_file = os.path.join(mypath, config['ztest'])
        self.method = config['method']
        self.hyperparams = config['hyperparams']
        self.num_classes = config['num_classes']

    def load_data(self):
        self.Xtrain = pd.read_pickle(self.Xtrain_file)
        self.ytrain = pd.read_pickle(self.ytrain_file)
        self.Xvalid = pd.read_pickle(self.Xvalid_file)
        self.yvalid = pd.read_pickle(self.yvalid_file)
        self.Xtest = pd.read_pickle(self.Xtest_file)
        self.ytest = pd.read_pickle(self.ytest_file)
        self.ztrain = pd.read_pickle(self.ztrain_file)
        self.zvalid = pd.read_pickle(self.zvalid_file)
        self.ztest = pd.read_pickle(self.ztest_file)

    def build_params(self):
        params = dict()
        params['Xtrain'] = self.Xtrain
        params['ytrain'] = self.ytrain
        params['Xvalid'] = self.Xvalid
        params['yvalid'] = self.yvalid
        params['Xtest'] = self.Xtest
        params['ytest'] = self.ytest
        params['method'] = self.method
        params['hyperparams'] = self.hyperparams
        params['num_classes'] = self.num_classes
        params['logpath'] = self.logpath
        params['ztrain'] = self.ztrain
        params['zvalid'] = self.zvalid
        params['ztest'] = self.ztest
        return params


if __name__ == '__main__':
    runner = Runner()
    runner.run()
