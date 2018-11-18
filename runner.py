import argparse
import json
import os
import pandas as pd
import model as ml


class Runner(object):
    '''Runs experiments from JSON config.'''
    def __init__(self):
        args = self.get_parser().parse_args()
        config_file = args.config
        self.adversarial = False
        self.unpack_config(config_file)
        self.load_data()
        self.train_params = self.build_params()

    def run(self):
        model = ml.Model(self.train_params)
        model.train()
        model.eval()

    def get_parser(self):
        parser = argparse.ArgumentParser(description='Run FAD experiments')
        parser.add_argument('config', help='JSON config filename')
        return parser

    def unpack_config(self, config_file):
        mypath = os.path.abspath(os.path.dirname(__file__))
        config = json.load(open(os.path.join(mypath, config_file), 'r'))
        print(config)
        self.logpath = os.path.join(mypath, config_file[:-5].replace('experiments', 'logs'))
        self.Xtrain_file = os.path.join(mypath, config['Xtrain'])
        self.ytrain_file = os.path.join(mypath, config['ytrain'])
        self.Xtest_file = os.path.join(mypath, config['Xtest'])
        self.ytest_file = os.path.join(mypath, config['ytest'])
        self.method = config['method']
        if self.method != "basic":
            self.adversarial = True
            self.ztrain_file = os.path.join(mypath, config['ztrain'])
            self.ztest_file = os.path.join(mypath, config['ztest'])
        self.hyperparams = config['hyperparams']

    def load_data(self):
        self.Xtrain = pd.read_pickle(self.Xtrain_file)
        self.ytrain = pd.read_pickle(self.ytrain_file)
        self.Xtest = pd.read_pickle(self.Xtest_file)
        self.ytest = pd.read_pickle(self.ytest_file)
        if self.adversarial:
            self.ztrain = pd.read_pickle(self.ztrain_file)
            self.ztest = pd.read_pickle(self.ztest_file)

    def build_params(self):
        params = dict()
        params['Xtrain'] = self.Xtrain
        params['ytrain'] = self.ytrain
        params['Xtest'] = self.Xtest
        params['ytest'] = self.ytest
        params['method'] = self.method
        params['hyperparams'] = self.hyperparams
        params['logpath'] = self.logpath
        if self.adversarial:
            params['ztrain'] = self.ztrain
            params['ztest'] = self.ztest
        return params


if __name__ == '__main__':
    runner = Runner()
    runner.run()
