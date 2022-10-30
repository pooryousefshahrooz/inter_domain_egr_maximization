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

    game.evaluate(tm_idx,network,solver,"RL",actions) 

def main(_):
    #Using cpu for testing
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.get_logger().setLevel('INFO')

    config = get_config(FLAGS) or FLAGS
 
    
    
    solver = Solver()
    for num_paths in range(1,int(config.num_of_paths)+1):
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
                        """we first find the candidate paths and use it for action dimention"""
                        # we se the state dimention and action dimention


                        game = CFRRL_Game(config, network)
                        model = Model(config, game.state_dims, game.action_dim, game.max_moves)

                        step = model.restore_ckpt(FLAGS.ckpt)
                        if config.method == 'actor_critic':
                            learning_rate = model.lr_schedule(model.actor_optimizer.iterations.numpy()).numpy()
                        elif config.method == 'pure_policy':
                            learning_rate = model.lr_schedule(model.optimizer.iterations.numpy()).numpy()
                        print('\nstep %d, learning rate: %f\n'% (step, learning_rate))

                        sim(config,model,network,solver, game,0)


# In[ ]:


if __name__ == '__main__':
    app.run(main)


# In[ ]:




