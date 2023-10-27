#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: martazaniolo
simulation.py simulates the Santa Barbara water system given a desal expansion
decision policy and a set of hydrological scenarios and returns objective values

"""

import numpy as np
from cachuma_lake import Cachuma
from gibraltar_lake import Gibraltar
from swp_lake import SWP
from policy import *
import numpy.matlib as mat
import numba
from numba import njit
import random


class log_results:
    pass
    class traj:
        pass
    class cost:
        pass



def monthly_demand(annual_dem):
    #this function partitions annual demand across the 12 months, according to historical average fraction of demand use per month
    month_dem = []
    monthly_fract = [0.0621, 0.0548, 0.0696, 0.0798, 0.0966, 0.0947, 0.1078, 0.1096, 0.1003, 0.0909, 0.0709, 0.0630]
    for a in annual_dem:
        for m in monthly_fract:
            month_dem.append(m*a)
    return month_dem

@njit
def nsim_equiv_res(s, u, n, s_max):
    # extremely fast reservoir mass balance computation
    r         = max(0, min(u, s))
    s_        = s + n - r
    s_        = max(0, min(s_, s_max ))

    return s_, r



class SB(object):
   ############# define relevant class parameters
    def __init__(self, opt_par, config):
        self.T           = 12 # period
        self.gibraltar   = Gibraltar(config)
        self.cachuma     = Cachuma(config)
        self.swp         = SWP(config)
        self.H           = self.gibraltar.H # length of time horizon
        self.Ny          = int(self.H/self.T) #number of years
        demand_scenario  = config['DEMAND']['demand_projection']
        annual_dem       = np.loadtxt('data/demand_' + demand_scenario + '.txt')
        self.sedim_scen  = self.gibraltar.sedimentation_scen
        self.demand      = monthly_demand(annual_dem)

        self.nom_cost_sw = 100
        self.nom_cost_rs = 420
        drought_type = config['DROUGHT_TYPES']['type']
        self.mds   = np.loadtxt('data/mission_type'+str(drought_type) + '.txt')
        self.nsim        = 3
        self.N           = opt_par.N #hidden nodes
        self.M           = opt_par.M #inputs
        self.K           = opt_par.K #outputs


        self.max_swp_market = 275

        # desal
        self.water_sold  = 1430*float(config['DESAL']['water_sold'])
        self.efficiency  = float(config['DESAL']['efficiency'])
        self.time_exp    = 24



    def simulate(self, P):

        #extract and interpret RBF paramters from param list P
        param, lin_param = set_param(P, self.N, self.M, self.K)

        self.H  = self.gibraltar.H
        H       = self.H
        self.Ny = H/self.T


        ncs     = self.cachuma.inflow
        ngis    = self.gibraltar.inflow
        nswps   = self.swp.inflow
        smax_gi = self.gibraltar.smax
        smax_ca = self.cachuma.smax
        smax_sw = self.swp.smax
        max_annual_market = 600 #assumption derived from WVSB, they can buy up to 600 AF/y from spot market
        montecito_agreement = 300 #montecito transfers back 300AF year in cachuma for SB once a year
        sustainable_yield = 1000/12 #assumption derived from WVSB


        J = []
        Jdef = []

        for s in range(self.nsim): # repeat simulation for the number of hydrological scenarios

            #initialize variables
            nc    = ncs[s,:]
            ngi   = ngis[s,:]
            nswp  = nswps[s,:]
            md    = self.mds[s,:]


            sc   = np.zeros(H+1)
            sgi  = np.zeros(H+1)
            sswp = np.zeros(H+1)
            deficit = np.zeros(H+1)
            rgi  = np.zeros(H+1)
            sc[0]      = self.cachuma.s0
            sgi[0]     = self.gibraltar.s0
            sswp[0]    = self.swp.s0
            u_ = []

            initial_desal = 260 - self.water_sold/12
            initial_np = 60
            installed_capacity = (initial_desal+initial_np)*np.ones(H)

            desal_capac  = initial_desal*np.ones(H)
            virtual_capac = initial_desal*np.ones(H)
            wwtp_capac   = initial_np*np.ones(H) # waste water treat plant for centralized P and NP reuse

            def_penalty       = 0
            penalty           = 0
            demand = self.demand
            sgw = 0
            smax_gi = self.gibraltar.smax



            for t in range(H): #simulate for the duration of the time horizon
   ############# compute value of indicators at time T

                storage_t    = self.compute_stor(sc, t)
                storage_t   += self.compute_stor(sswp, t)
                storage_t   += self.compute_stor(sgi, t)

                annual_demand     = self.compute_stor(demand, t)

                inputs = [storage_t/35000, annual_demand/1700]

   ############## extract action from policy
                u        = get_output(inputs, param, lin_param, self.N, self.M, self.K)
                des_size = u[0]*2000
                u_.append(inputs)

   ############## read policy decisions and implement it in model
                if des_size > virtual_capac[t]: # it can be bigger but not smaller
                    T = min(H, t+int( self.time_exp))
                    virtual_capac[t:H] = des_size
                    desal_capac[T:H] = np.floor(des_size)

                installed_capacity[t] = sum([desal_capac[t], wwtp_capac[t]])

   ############## simulation of surface water reservoirs
                dem = demand[t]
                # demand from surface water = total demand - tech installed - mission tunnel inflow - gw sustainable yield
                d = max( 0, dem - installed_capacity[t]*self.efficiency - md[t] - sustainable_yield)

                # release decision is proportional to the storage in each reservoir
                smax_gi    = smax_gi - self.sedim_scen
                SS = 0.0001+(sc[t] + sgi[t] + sswp[t])
                uc  = sc[t]/SS
                ugi = sgi[t]/SS
                uswp = sswp[t]/SS

                if uswp*d > self.swp.max_release:
                    while uswp*d > self.swp.max_release:
                        uswp -= 0.05
                        uc += 0.04
                        ugi += 0.01

                # surface water allocation in cachuma swp and comes in the form of an annual allocation
                # distributed in the month of October for Cachuma and May for SWP
                if (t%12)==9: # October
                    sc[t]  = sc[t]*self.cachuma.carryover + (1- self.cachuma.carryover)*0.3 #a third of curtailed allocation is redistributed to SB
                    nc_    = nc[int((t-9)/self.T)] + montecito_agreement
                else:
                    nc_ = 0

                if (t%12)==4: #May
                    nswp_ = nswp[int((t-4)/self.T)]
                else:
                    nswp_ = 0

                # mass balance of water reservoirs
                s_, r_c  = nsim_equiv_res(sc[t], uc*d, nc_, smax_ca) #self.cachuma.integration(sc[t], uc, nc_, d)
                sc[t+1] = s_

                s_, r_gi  = nsim_equiv_res(sgi[t], ugi*d, ngi[t], smax_gi) #self.gibraltar.integration(sgi[t], ugi, ngi[t], d)
                sgi[t+1] = s_
                rgi[t+1] = r_gi

                s_, r_swp  = nsim_equiv_res(sswp[t], uswp*d, nswp_, smax_sw) #self.swp.integration(sswp[t], uswp, nswp_, d)
                sswp[t+1] = s_

                sgw += 250/12 # groundwater storage annual recharge

                # calculation of deficit for penalty
                deficit[t+1] = max( 0, demand[t] - r_swp - r_c - r_gi - md[t] - installed_capacity[t] - sustainable_yield)
                if deficit[t+1] < 1e-4:
                    deficit[t+1] = 0

                # restricted purchase of market water to mitigate the deficit
                max_deliverable_market = max( 0, self.max_swp_market - r_swp )
                max_market = min(max_deliverable_market, max_annual_market/12) #assumption derived from WV, they buy up to 600AF/y or 50AF/month
                market = min( max_market, deficit[t+1] )
                deficit[t+1] = deficit[t+1] - market

                # dip into groundwater reserves if still in a deficit
                extract_gw = min( sgw, deficit[t+1] )
                deficit[t+1] = deficit[t+1] - extract_gw
                sgw -= extract_gw

############# calculate deficit
            #first 3 years are spin-up time to allow expansion decisions to come into effect
            deficit_annual_ = np.reshape(deficit[37:], (50, 12)).T
            def_ann = sum(deficit_annual_)
            perc_def_ann = [d / 10374 * 100 for d in def_ann] #percentage of demand is calculated wrt to the same reference: annual baseline demand of 10374 AF/y
            penalty = 0
            #calculate deficit category: 0%, max 10%, max 20%
            if sum(def_ann) == 0: #no deficit
                Jdef.append( 0 )

            elif sum([d < 10 for d in perc_def_ann ]) == len(perc_def_ann) : # always less than 10%
                Jdef.append( 10.05 )

            elif sum([d < 20 for d in perc_def_ann ]) == len(perc_def_ann) : # always less than 5%
                Jdef.append( 20.05 )

            else: #large penalty for excessive deficit
                penalty = 10000
                Jdef.append( penalty ) # penalty

            J.append( np.mean(desal_capac[37:]) + penalty ) #remove spin-up time

        return [np.mean(J), np.max(Jdef)]


    def compute_stor(self, sc, t):
        if t== 0:
            st = sc[0]
        elif t < 12:
            st = np.mean(sc[:t])
        else:
            st = np.mean(sc[t-11:t])
        return st
