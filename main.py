#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function

import numpy as np
import random
from tqdm import tqdm
import multiprocessing as mp
from absl import app
from absl import flags
import ast
from network import Network
from config import get_config
from solver import Solver
import time
import os
FLAGS = flags.FLAGS


# In[ ]:


def main(_):
    config = get_config(FLAGS) or FLAGS
    for num_paths in range(int(config.min_num_of_paths),int(config.num_of_paths)+1):
        for edge_capacity_bound in config.edge_capacity_bounds:
            network = Network(config,edge_capacity_bound,False)
            for fidelity_threshold_up_range in config.fidelity_threshold_ranges:
                network.fidelity_threshold_range = fidelity_threshold_up_range
                network.set_each_wk_k_fidelity_threshold()
                for edge_fidelity_range in config.edge_fidelity_ranges:
                    for purificaion_scheme in ["end_level"]:
                        if purificaion_scheme =="end_level":
                            network.end_level_purification_flag = True
                        else:
                            network.end_level_purification_flag = False

                        network.end_level_purification_flag = True
                        network.set_edge_fidelity(edge_fidelity_range)
                        # we get all the paths for all workloads
                        network.num_of_paths = num_paths
                        network.get_path_info()
                        for q_value in config.q_values:
                            network.q_value = q_value
                            for scheme in config.schemes:
                                if scheme in ["EGR","EGRSquare","Hop"]:
                                    network.evaluate_shortest_path_routing(scheme)
                                elif scheme =="Genetic":
                                    network.evaluate_genetic_algorithm_for_path_selection()
                                elif scheme =="RL":
                                    network.evaluate_rl_for_path_selection()
                                else:
                                    print("not valid scheme (%s): set schemes from EGR, EGRSquare,Hop, Genetic, or RL keywords"%(scheme))


# In[ ]:


if __name__ == '__main__':
    app.run(main)

