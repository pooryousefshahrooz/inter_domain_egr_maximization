#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import os
import sys
from docplex.mp.progress import *
from docplex.mp.progress import SolutionRecorder
import docplex.mp.model as cpx
import networkx as nx
import time
from config import get_config
from absl import flags
FLAGS = flags.FLAGS


# In[ ]:


class Solver:
    def __init__(self):
        pass
    def CPLEX_maximizing_EGR(self,wk_idx,network):
#         for k in network.each_wk_organizations[wk_idx]:
#             for u in network.each_wk_each_k_user_pair_ids[wk_idx][k]:
#                 print("we are k %s u %s "%(k,u))
#                 for p in network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][u]:
#                     print("wk %s k %s w %s user %s w %s path %s"%(wk_idx,k,network.each_wk_k_weight[wk_idx][k],u,network.each_wk_k_u_weight[wk_idx][k][u],p))

#         print("network.max_edge_capacity",network.max_edge_capacity,type(network.max_edge_capacity))
        opt_model = cpx.Model(name="inter_organization_EGR")
        x_vars  = {(k,p): opt_model.continuous_var(lb=0, ub= network.max_edge_capacity,
                                  name="w_{0}_{1}".format(k,p))  for k in network.each_wk_organizations[wk_idx]
                   for u in network.each_wk_each_k_user_pair_ids[wk_idx][k] for p in network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][u]}

        
    #     for k in network.K:
    #         for u in network.each_k_user_pairs[k]:
    #             for p in network.each_k_u_paths[k][u]:
    #                 print("organization %s user %s #paths %s cost %s p %s path %s"%(k,u,network.num_of_paths,network.each_link_cost_metric,p,network.set_of_paths[p]))

    #     time.sleep(9)
        #Edge constraint
        for edge in network.set_E:
            if network.end_level_purification_flag:
                opt_model.add_constraint(
                    opt_model.sum(x_vars[k,p]*network.each_wk_k_u_weight[wk_idx][k][u] 
                    * network.get_required_purification_EPR_pairs(p,network.get_each_wk_k_threshold(wk_idx,k))
                    for k in network.each_wk_organizations[wk_idx] for u in network.each_wk_each_k_user_pair_ids[wk_idx][k]
                    for p in network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][u]
                    if network.check_path_include_edge(edge,p))
                     <= network.each_edge_capacity[edge], ctname="edge_capacity_{0}".format(edge))
            else:
                opt_model.add_constraint(
                    opt_model.sum(x_vars[k,p]*network.each_wk_k_u_weight[wk_idx][k][u] *
                    network.get_required_edge_level_purification_EPR_pairs(edge,p,network.each_wk_k_fidelity_threshold[k],wk_idx)
                    for k in network.each_wk_organizations[wk_idx] for u in network.each_wk_each_k_user_pair_ids[k]
                    for p in network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][u]
                    if network.check_path_include_edge(edge,p))

                     <= network.each_edge_capacity[edge], ctname="edge_capacity_{0}".format(edge))

        objective = opt_model.sum(x_vars[k,p]*network.each_wk_k_weight[wk_idx][k] * network.each_wk_k_u_weight[wk_idx][k][u]*network.q_value**(network.get_path_length(p)-1)
                              for k in network.each_wk_organizations[wk_idx]
                              for u in network.each_wk_each_k_user_pair_ids[wk_idx][k] 
                              for p in network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][u]
                              )


        # for maximization
        opt_model.maximize(objective)

    #     opt_model.solve()
        #opt_model.print_information()
        #try:
        opt_model.solve()


        #print('docplex.mp.solution',opt_model.solution)
        objective_value = -1
        try:
            if opt_model.solution:
                objective_value =opt_model.solution.get_objective_value()
        except ValueError:
            print(ValueError)

        opt_model.clear()

        return objective_value

