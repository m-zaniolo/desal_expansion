#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 17:36:20 2021

@author: martazaniolo
"""
import sys
sys.path.append('src')
sys.path.append('ptreeopt')
import simulation
from simulation import SB
from sim_individual import SBsim
from src import *
import numpy as np
import numpy.matlib as mat
import argparse
from platypus import *
from platypus.experimenter import experiment, calculate, display
#from concurrent.futures import ProcessPoolExecutor
from sb_problem import SB_Problem
from plot_optimization import *
import matplotlib.pyplot as plt
import pickle
from platypus import ProcessPoolEvaluator
import configparser
import pandas as pd
import logging

class OptimizationParameters(object):
    def __init__(self):
        ###DEMO OPTIMIZATION, use higher values of max_gen and npop if results are not converged
        self.max_gen  = 100 
        self.npop     = 360
        self.nfe      = self.max_gen*self.npop
        self.cores    = 2
        self.nseeds   = 2
        self.N        = 3 #hidden nodes
        self.M        = 2 #inputs
        self.K        = 1 #outputs
        self.nparam   = self.N*(2*self.M + self.K) + self.K
        self.nobjs    = 2
        self.lb       = np.concatenate( (mat.repmat([-1,0], 1, self.M ), np.zeros([1,self.K])), axis = 1 )
        self.LB       = np.concatenate( (np.zeros([1,self.K]), mat.repmat( self.lb, 1, self.N ) ), axis = 1 ) #optimization params lower bound
        self.ub       = np.concatenate( (mat.repmat([1,1], 1, self.M ), np.ones([1,self.K])), axis = 1 ) 
        self.UB       = np.concatenate( (np.ones([1,self.K]), mat.repmat( self.ub, 1, self.N ) ), axis = 1 ) #optimization params upper bound
        self.log_freq = 2000

class Solution():
    pass

def run(opt_par, config):

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    if 0: #switch to 1 for parallelizing computation across number of available cores
        with ProcessPoolEvaluator(opt_par.cores) as pool:
            algorithm = [(NSGAII, { "population_size":opt_par.npop, "offspring_size":opt_par.npop,  "evaluator":pool, "log_frequency":opt_par.log_freq, "verbose":3})]
            problem   = [SB_Problem(opt_par, config)]
            results = experiment(algorithm, problem, seeds = opt_par.nseeds, nfe=opt_par.nfe)
    else:
        algorithm = [(NSGAII, { "population_size":opt_par.npop, "offspring_size":opt_par.npop})]
        problem   = [SB_Problem(opt_par, config)]
        results = experiment(algorithm, problem, seeds = opt_par.nseeds, nfe=opt_par.nfe)


    return results 


if __name__ == '__main__':

    #example run for a baseline configuration
    conf_string = 'conf2_2024.txt'

    ### commented below, run with configuration ID as input
    #parser = argparse.ArgumentParser()
    #parser.add_argument("config_file", help="config_file")

    #args = parser.parse_args()
    #conf_string = 'conf2_' + args.config_file + '.txt'

    
    opt_par = OptimizationParameters()
    config = configparser.ConfigParser()
    config.read('configuration_files/' + conf_string )

    print( conf_string )


    # run optimization
    result = run(opt_par, config)

    objs  = []
    param = []

    for seed in range(opt_par.nseeds):
        solutions = result['NSGAII']['SB_Problem'][seed]

        objs.extend([ [s.objectives[0], s.objectives[1]] for s in solutions]) 
        param.extend([s.variables for s in solutions])

    mask = is_pareto_efficient(np.array(objs))

    objs_eff   = []
    param_eff  = []
    for m, o, p in zip(mask, objs, param):
        if m:
            objs_eff.append(o)
            param_eff.append(p)


    #find no deficit solution
    idx_nodeficit = np.argmin(np.array(objs_eff)[:,1] )

    #creat solution structure
    solution = Solution()
    solution.best_score = objs_eff[idx_nodeficit]
    solution.best_solution = param_eff[idx_nodeficit]
    solution.all_solutions = param_eff
    solution.objs = objs_eff
    solution.log = []
    

    #simulate solution
    sim_model = SBsim(opt_par, config)
    sim = SB(opt_par, config)
    for i in range(3):
        log = sim_model.simulate(param_eff[idx_nodeficit], i)
        solution.sim_model = sim_model
        solution.log.append(log)

    # dictionary with list object in values
    details = {
        'gibraltar_storage' : [config['SURFACE_WATER_SOURCES']['gibraltar_storage']],
        'cachuma_carryover' : [float(config['SURFACE_WATER_SOURCES']['cachuma_carryover'])],
        'cachuma_allocation': [float(config['SURFACE_WATER_SOURCES']['cachuma_nominal_allocation'])],
        'swp_allocation'    : [float(config['SURFACE_WATER_SOURCES']['swp_nominal_allocation'])],

        'intensity'         : [config['DROUGHT_TYPES']['type']],

        'water_sold'        : [float(config['DESAL']['water_sold'])],
        'efficiency'        : [float(config['DESAL']['efficiency'])],

        'demand_projection' : [config['DEMAND']['demand_projection']],
    }
    
    # creating a Dataframe object
    df = pd.DataFrame(details)
    solution.config = df

    string = 'results/results_' + args.config_file + '.dat'
    with open(string, 'wb') as f:
        pickle.dump(solution, f)

    plot_pareto(objs_eff)
    plt.savefig('figure.png')

