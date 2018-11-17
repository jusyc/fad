import argparse
import json
import os
import pandas as pd
import numpy as np
import pickle


class Runner(object):
    '''Runs experiments from JSON config.'''
    def __init__(self):
        args = self.get_parser().parse_args()
        config_file = args.config
        self.unpack_config(config_file)
        self.load_data()
        self.train_params = self.build_params()

    def get_parser(self):
        parser = argparse.ArgumentParser(description='Run FAD experiments')
        parser.add_argument('config', help='JSON config filename')
        return parser

    def unpack_config(self, config_file):
        mypath = os.path.abspath(os.path.dirname(__file__))
        config = json.load(open(os.path.join(mypath, config_file), 'r'))
        self.Xtrain_file = os.path.join(mypath, config['Xtrain'])
        self.ytrain_file = os.path.join(mypath, config['ytrain'])
        self.Xtest_file = os.path.join(mypath, config['Xtest'])
        self.ytest_file = os.path.join(mypath, config['ytest'])
        self.method = os.path.join(mypath, config['method'])

    def load_data(self):
        self.Xtrain = pd.read_pickle(self.Xtrain_file)
        self.ytrain = pd.read_pickle(self.ytrain_file)
        self.Xtest = pd.read_pickle(self.Xtest_file)
        self.ytest = pd.read_pickle(self.ytest_file)

    def build_params(self):
        params = dict()
        params['Xtrain'] = self.Xtrain
        params['ytrain'] = self.ytrain
        params['Xtest'] = self.Xtest
        params['ytest'] = self.ytest
        params['method'] = self.method


if __name__ == '__main__':
    runner = Runner()