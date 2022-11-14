#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function

import os
import ast
import numpy as np
from absl import app
from absl import flags

import tensorflow as tf
from env import Environment
from game import CFRRL_Game
from model import Model
from network import Network
from config import get_config
from solver import Solver
import csv
import time
FLAGS = flags.FLAGS
flags.DEFINE_string('ckpt', '', 'apply a specific checkpoint')
flags.DEFINE_boolean('eval_delay', False, 'evaluate delay or not')

def sim(config, model,network,solver, game,tm_idx):
    #for tm_idx in game.tm_indexes:
    state = game.get_state(tm_idx)
    if config.method == 'actor_critic':
        policy = model.actor_predict(np.expand_dims(state, 0)).numpy()[0]
    elif config.method == 'pure_policy':
        policy = model.policy_predict(np.expand_dims(state, 0)).numpy()[0]
    actions = policy.argsort()[-game.max_moves:]

    egr = game.evaluate(tm_idx,network,solver,"RL",actions) 
    return egr
def main(_):
    #Using cpu for testing
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.get_logger().setLevel('INFO')

    config = get_config(FLAGS) or FLAGS
 
    
    
    solver = Solver()
    for num_paths in range(int(config.min_num_of_paths),int(config.num_of_paths)+1):
        for network_topology in config.topology_file:
            for edge_capacity_bound in config.edge_capacity_bounds:
                network = Network(config,edge_capacity_bound,True)
                for fidelity_threshold_up_range in config.fidelity_threshold_ranges:
                    network.fidelity_threshold_range = fidelity_threshold_up_range
                    network.set_each_wk_k_fidelity_threshold()
                    for edge_fidelity_range in config.edge_fidelity_ranges:
                        network.end_level_purification_flag = True
                        network.set_edge_fidelity(edge_fidelity_range)
                        # we get all the paths for all workloads
                        network.num_of_paths = num_paths
                        network.get_path_info()
                        each_wk_scheme_egr ={}
                        each_wk_idx_optimal = {}
                        """we first find the candidate paths and use it for action dimention"""
                        # we se the state dimention and action dimention


                        game = CFRRL_Game(config, network)
                        model = Model(config, game.state_dims, game.action_dim, game.max_moves)
                        last_chckpoint = model.restore_ckpt(FLAGS.ckpt)
                        while(last_chckpoint <=config.max_step):
                            time.sleep(1)
                            model = Model(config, game.state_dims, game.action_dim, game.max_moves)
                            current_chckpoint = model.restore_ckpt(FLAGS.ckpt)
                            if config.method == 'actor_critic':
                                learning_rate = model.lr_schedule(model.actor_optimizer.iterations.numpy()).numpy()
                            elif config.method == 'pure_policy':
                                learning_rate = model.lr_schedule(model.optimizer.iterations.numpy()).numpy()
                            print('\nstep %d, learning rate: %f\n'% (current_chckpoint, learning_rate))
                            if last_chckpoint<current_chckpoint:
                                last_chckpoint = current_chckpoint
                                for wk_idx in range(len(game.wk_indexes)):
                                    try:
                                        if wk_idx in each_wk_scheme_egr:
                                            EGR_egr = each_wk_scheme_egr[wk_idx]["EGR"]
                                            hop_egr = each_wk_scheme_egr[wk_idx]["hop"]
                                            EGRSQUARE_egr = each_wk_scheme_egr[wk_idx]["EGRsquare"]
                                            optimal_egr = each_wk_scheme_egr[wk_idx]["Optimal"]

                                        else:
                                            EGR_egr = game.evaluate(wk_idx,network,solver,"EGR",[])
                                            hop_egr = game.evaluate(wk_idx,network,solver,"hop",[])
                                            EGRSQUARE_egr = game.evaluate(wk_idx,network,solver,"EGRsquare",[])
                                            optimal_egr = game.evaluate(wk_idx,network,solver,"Optimal",[])
                                            try:
                                                each_wk_scheme_egr[wk_idx]["EGR"]= EGR_egr
                                                each_wk_scheme_egr[wk_idx]["hop"]= hop_egr
                                                each_wk_scheme_egr[wk_idx]["EGRsquare"]= EGRSQUARE_egr
                                                each_wk_scheme_egr[wk_idx]["Optimal"]= optimal_egr
                                            except:
                                                each_wk_scheme_egr[wk_idx] = {}
                                                each_wk_scheme_egr[wk_idx]["EGR"]= EGR_egr
                                                each_wk_scheme_egr[wk_idx]["hop"]= hop_egr
                                                each_wk_scheme_egr[wk_idx]["EGRsquare"]= EGRSQUARE_egr
                                                each_wk_scheme_egr[wk_idx]["Optimal"]= optimal_egr
                                    except:
                                        EGR_egr = game.evaluate(wk_idx,network,solver,"EGR",[])
                                        hop_egr = game.evaluate(wk_idx,network,solver,"hop",[])
                                        EGRSQUARE_egr = game.evaluate(wk_idx,network,solver,"EGRsquare",[])
                                        optimal_egr = game.evaluate(wk_idx,network,solver,"Optimal",[])
                                        try:
                                            each_wk_scheme_egr[wk_idx]["EGR"]= EGR_egr
                                            each_wk_scheme_egr[wk_idx]["hop"]= hop_egr
                                            each_wk_scheme_egr[wk_idx]["EGRsquare"]= EGRSQUARE_egr
                                            each_wk_scheme_egr[wk_idx]["Optimal"]= optimal_egr
                                        except:
                                            each_wk_scheme_egr[wk_idx] = {}
                                            each_wk_scheme_egr[wk_idx]["EGR"]= EGR_egr
                                            each_wk_scheme_egr[wk_idx]["hop"]= hop_egr
                                            each_wk_scheme_egr[wk_idx]["EGRsquare"]= EGRSQUARE_egr
                                            each_wk_scheme_egr[wk_idx]["Optimal"]= optimal_egr
                                    toplogy_wk_epoch_scheme_result = config.testing_results
                                    rl_egr= sim(config,model,network,solver, game,wk_idx)
                                    
                                    if rl_egr >optimal_egr:
                                        
                                        import pdb
                                        pdb.set_trace()
                                    
                                    with open(toplogy_wk_epoch_scheme_result, 'a') as newFile:                                
                                        newFileWriter = csv.writer(newFile)
                                        newFileWriter.writerow([config.topology_file,wk_idx,network.num_of_paths,
                                        "optimal",optimal_egr,
                                        "RL",rl_egr,
                                        "EGR",EGR_egr,
                                        "hop",hop_egr,
                                        "EGRsquare",EGRSQUARE_egr,current_chckpoint,
                                                            config.initial_learning_rate,
                                                               config.learning_rate_decay_rate,
                                                               config.moving_average_decay,
                                                               config.entropy_weight,config.optimizer])
                                    print("epoch #",current_chckpoint,"# paths",network.num_of_paths,"wk_idx",wk_idx,
                                                "optimal",optimal_egr,
                                                "RL",rl_egr,
                                                "EGR",EGR_egr,
                                                "hop",hop_egr,
                                                "EGRsquare",EGRSQUARE_egr)
                                    try:
                                        each_wk_idx_optimal[wk_idx].append(optimal_egr)
                                    except:
                                        each_wk_idx_optimal[wk_idx]=[optimal_egr]
                                        
                                    for wk_idx,optimal_values in each_wk_idx_optimal.items():
                                        if int(sum(optimal_values)/len(optimal_values))!=int(optimal_values[0]):
                                            print("ERROR!!!", sum(optimal_values)/len(optimal_values),optimal_values[0])
                                            pdb.set_trace()
                            else:
                                print("last %s current %s"%(last_chckpoint,current_chckpoint))


# In[ ]:


if __name__ == '__main__':
    app.run(main)


# In[ ]:




