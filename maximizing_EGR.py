#!/usr/bin/env python
# coding: utf-8

# In[1]:



import csv
from network import Network
import os
import sys
from docplex.mp.progress import *
from docplex.mp.progress import SolutionRecorder
import networkx as nx
import time
from config import get_config
from absl import flags
FLAGS = flags.FLAGS


# In[11]:


def CPLEX_maximizing_EGR(network):
    import docplex.mp.model as cpx
    opt_model = cpx.Model(name="inter_organization_EGR")
    x_vars  = {(k,p): opt_model.continuous_var(lb=0, ub= network.max_edge_capacity,
                              name="w_{0}_{1}".format(k,p))  for k in network.K
               for u in network.each_k_user_pairs[k] for p in network.each_k_u_paths[k][u]}
    
#     for k in network.K:
#         for u in network.each_k_user_pairs[k]:
#             for p in network.each_k_u_paths[k][u]:
#                 print("organization %s w %s user %s w %s path %s"%(k,network.each_k_weight[k],u,network.each_k_u_weight[k][u],p))
                
#     for k in network.K:
#         for u in network.each_k_user_pairs[k]:
#             for p in network.each_k_u_paths[k][u]:
#                 print("organization %s user %s #paths %s cost %s p %s path %s"%(k,u,network.num_of_paths,network.each_link_cost_metric,p,network.set_of_paths[p]))
                
#     time.sleep(9)
    #Edge constraint
    for edge in network.set_E:
        if network.end_level_purification_flag:
            opt_model.add_constraint(
                opt_model.sum(x_vars[k,p]*
                network.get_required_purification_EPR_pairs(p,network.get_each_k_threshold(k))
                for k in network.K for u in network.each_k_user_pairs[k]
                for p in network.each_k_u_paths[k][u]
                if network.check_path_include_edge(edge,p))
                 <= network.each_edge_capacity[edge], ctname="edge_capacity_{0}".format(edge))
        else:
            opt_model.add_constraint(
                opt_model.sum(x_vars[k,p]*
                network.get_required_edge_level_purification_EPR_pairs(edge,p,network.each_k_fidelity_threshold[k])
                for k in network.K for u in network.each_k_user_pairs[k]
                for p in network.each_k_u_paths[k][u]
                if network.check_path_include_edge(edge,p))

                 <= network.each_edge_capacity[edge], ctname="edge_capacity_{0}".format(edge))
   
    objective = opt_model.sum(x_vars[k,p]*network.each_k_weight[k] * network.each_k_u_weight[k][u] 
                          for k in network.K
                          for u in network.each_k_user_pairs[k] 
                          for p in network.each_k_u_paths[k][u]
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


# In[12]:


def maximizing_EGR():
    config = get_config(FLAGS) or FLAGS
    for network_topology,file_path in each_network_topology_file.items():
        for i in range(experiment_repeat):
            network = Network(config,file_path)
            network.set_organizations()
            network.set_organizations_user_pairs()
            network.set_each_organization_weight()
            network.set_each_user_pair_weight()
            for fidelity_threshold_up_range in fidelity_threshold_ranges:
                network.fidelity_threshold_range = fidelity_threshold_up_range
                network.set_each_k_fidelity_threshold()
                for num_paths in [1,2,3,4,5,6]:
                    network.num_of_paths = num_paths
                    for edge_fidelity_range in edge_fidelity_ranges:
                        network.set_edge_fidelity(edge_fidelity_range)
                        for link_cost_metric in link_cost_metrics:
                            network.each_link_cost_metric =link_cost_metric 
                            network.set_link_weight(link_cost_metric)
                            try:
                                network.set_paths_in_the_network()
                                for purificaion_scheme in purification_schemes:
                                    objective_value=-1
                                    try:
                                        if purificaion_scheme =="end_level":
                                            network.end_level_purification_flag = True
                                        else:
                                            network.end_level_purification_flag = False
                                        objective_value = CPLEX_maximizing_EGR(network)
                                    except ValueError:
                                        print(ValueError)
                                    print("for purificaion %s link cost %s topology %s iteration %s from %s  fidelity range %s  path number %s objective_value %s"%
                                    (purificaion_scheme,link_cost_metric,network_topology,i,experiment_repeat,fidelity_threshold_up_range,num_paths, objective_value))  
                                    with open(results_file_path, 'a') as newFile:                                
                                        newFileWriter = csv.writer(newFile)
                                        newFileWriter.writerow([network_topology,link_cost_metric,num_paths,
                                                                objective_value,i,
                                                                fidelity_threshold_up_range,
                                                                edge_fidelity_range,purificaion_scheme]) 
                            except ValueError:
                                print(ValueError)
                                pass


# In[13]:


experiment_repeat =50
edge_fidelity_ranges =[0.9,0.92,0.94,0.96]
fidelity_threshold_ranges = [0.7,0.8,0.9,0.94,0.96]
link_cost_metrics = ["EGR","hop","EGRsquare"]
purification_schemes = ["edge_level","end_level"]
results_file_path = "results/egr.csv"

each_network_topology_file = {}

each_network_topology_file = {"ATT":'data/ATT',"SURFnet":'data/SURFnet'}
maximizing_EGR()


# In[ ]:




