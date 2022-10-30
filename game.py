#!/usr/bin/env python
# coding: utf-8

# In[8]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import ast
import csv
from tqdm import tqdm
import numpy as np
import time
import pdb
import itertools
from itertools import combinations
# from pulp import LpMinimize, LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, value, GLPK

OBJ_EPSILON = 1e-12


# In[ ]:


#!/usr/bin/env python
# coding: utf-8

class Game(object):
    def __init__(self, config, random_seed=1000):
        self.random_state = np.random.RandomState(seed=random_seed)
        self.model_type = config.model_type



# In[ ]:


class CFRRL_Game():
    def __init__(self, config,network):
        random_seed=1000
        self.random_state = np.random.RandomState(seed=random_seed)
        self.model_type = config.model_type
        
        self.project_name = config.project_name
        #env.num_pairs = self.num_pairs
        self.action_dim = network.path_counter
        self.each_wk_each_k_user_pair_ids =network.each_wk_each_k_user_pair_ids
        self.max_moves = network.num_of_paths*config.number_of_user_pairs
        self.num_of_organizations=network.num_of_organizations
        self.number_of_user_pairs=network.number_of_user_pairs
        
        self.each_wk_each_k_user_pair_ids= network.each_wk_each_k_user_pair_ids
        self.each_wk_k_weight=network.each_wk_k_weight
        self.each_wk_k_u_weight = network.each_wk_k_u_weight
        self.each_wk_k_fidelity_threshold=network.each_wk_k_fidelity_threshold
        #print("self.max_moves %s self.action_dim %s self.max_moves %s self.action_dim %s"
       #       %(self.max_moves , self.action_dim, self.max_moves, self.action_dim))
        #assert self.max_moves <= self.action_dim, (self.max_moves, self.action_dim)
        self.wk_cnt=network.work_load_counter
        self.wk_indexes = np.arange(0, self.wk_cnt)
        
        state = np.zeros((1, config.num_of_organizations*config.number_of_user_pairs,1), dtype=np.float32)   # state  []
        self.state_dims =  state.shape
        if config.method == 'pure_policy':
            self.baseline = {}
        self.each_wk_optimal_egr = {}
        self.each_wk_action_reward = {}
        self.each_wk_optimal_egr_paths = {}
        #self.generate_inputs(normalization=True)
        #self.state_dims = self.normalized_traffic_matrices.shape[1:]
        print('Input dims :', self.state_dims)
        print('Max moves :', self.max_moves)
     
    def select_random_user_pairs(self):
        each_t_user_pairs = {}
        each_user_each_t_weight = {}
        candidate_user_pairs = []
        number_of_user_pairs = 3
        nodes = []
        for i in range(25):
            nodes.append(i)
        for src in nodes:
            for dst in nodes:
                if src!=dst:
                    if (src,dst) not in candidate_user_pairs and (dst,src) not in candidate_user_pairs:
                        candidate_user_pairs.append((src,dst))
        selected_user_pairs = []

        while(len(selected_user_pairs)<number_of_user_pairs):
            user_pair = candidate_user_pairs[random.randint(0,len(candidate_user_pairs)-1)]
            if user_pair not in selected_user_pairs:
                selected_user_pairs.append(user_pair)

        return selected_user_pairs
    def generate_work_load(self):
        pairs_in_order = []
        num_nodes = 26
        f = open("data/ATT_original", 'r')
        traffic_matrices = []
        work_load_counter = 0
        for line in f:
            pairs_in_order = []
            volumes = line.strip().split(' ')
            volumes = line.strip().split(' ')
            total_volume_cnt = len(volumes)
            for v in range(total_volume_cnt):
                i = int(v/num_nodes)
                j = v%num_nodes
                pairs_in_order.append((i,j))

        for work_load in range(1000):
            print(" workload %s done from %s "%(work_load,1000))
            selected_user_pairs = select_random_user_pairs()
            line_string = ""
            for user_pair in pairs_in_order:
                if user_pair in selected_user_pairs:
                    weight = 1
                else:
                    weight = 0
                if line_string:
                    line_string = line_string+" "+str(weight)
                else:
                    line_string = str(weight)
            with open('data/ATTWK', 'a') as file: 
                file.write(line_string+"\n")
    def generate_inputs(self, normalization=True):
        pairs_in_order = []
        num_nodes = 26
        f = open("data/ATT_original", 'r')
        traffic_matrices = []
        work_load_counter = 0
        all_active_user_pairs_acros_wks = []
        for line in f:
            pairs_in_order = []
            volumes = line.strip().split(' ')
            volumes = line.strip().split(' ')
            total_volume_cnt = len(volumes)
            for v in range(total_volume_cnt):
                i = int(v/num_nodes)
                j = v%num_nodes
                pairs_in_order.append((i,j))
    

    def get_state(self, wk_idx):
        state = np.zeros((1, self.num_of_organizations*self.number_of_user_pairs,1), dtype=np.float32)
        indx= 0
        for k,user_pairs in self.each_wk_each_k_user_pair_ids[wk_idx].items():
            for user in user_pairs:
                state[0][indx] = user
                indx+=1
#         for k,user_pair_ids in self.each_wk_each_k_user_pair_ids[wk_idx].items():
#             state[0][indx] = self.each_wk_k_weight[wk_idx][k]
#             indx+=1
#         for k,user_pairs in self.each_wk_each_k_user_pair_ids[wk_idx].items():
#             for user in user_pairs:
#                 state[0][indx] = self.each_wk_k_u_weight[wk_idx][k][user]
#                 indx+=1
#         for k in self.each_wk_each_k_user_pair_ids[wk_idx]:
#             state[0][indx] = self.each_wk_k_fidelity_threshold[wk_idx][k]
#             indx+=1
        
        #print("state is ",state)
        return state
    def compute_egr(self,actions,wk_idx,network,solver):
        network.each_wk_each_k_each_user_pair_id_paths = {}
        for k,user_pair_ids in network.each_wk_each_k_user_pair_ids[wk_idx].items():
            for user_pair in user_pair_ids:
                having_at_least_one_path_flag = False
                path_ids=[]
                for link_cost_metric in network.link_cost_metrics:
                    for p_id in network.each_scheme_each_user_pair_paths[link_cost_metric][user_pair]:
                        path_ids.append(p_id)
                for path_id in path_ids:
                    if path_id in actions:
                        having_at_least_one_path_flag = True
                        try:
                            network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair].append(path_id)
                        except:
                            try:
                                network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair] = [path_id]
                            except:
                                try:
                                    network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k]={}
                                    network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair] = [path_id]
                                except:
                                    network.each_wk_each_k_each_user_pair_id_paths[wk_idx]={}
                                    network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k]={}
                                    network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair] = [path_id]
                if not having_at_least_one_path_flag:
                    try:
                        network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair] = []
                    except:
                        try:
                            network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k]={}
                            network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair] = []
                        except:
                            network.each_wk_each_k_each_user_pair_id_paths[wk_idx]={}
                            network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k]={}
                            network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair] = []
        
        if not network.setting_basic_fidelity_flag:
            network.set_each_path_basic_fidelity()
            network.setting_basic_fidelity_flag = True
            """we set the required EPR pairs to achieve each fidelity threshold"""
            network.set_required_EPR_pairs_for_each_path_each_fidelity_threshold(wk_idx)       
        
        egr = solver.CPLEX_maximizing_EGR(wk_idx,network)
        return egr
    def reward(self, wk_idx,network, actions,solver):
#         print("compuiting reward.....")
        chosen_paths = []
        for item in actions:
            chosen_paths.append(item)
        chosen_paths.sort()
        try:
            if wk_idx in self.each_wk_action_reward:
                if tuple(chosen_paths) in self.each_wk_action_reward[wk_idx]:
                    rl_egr = self.each_wk_action_reward[wk_idx][tuple(chosen_paths)]
                else:
                    rl_egr = self.compute_egr(actions,wk_idx,network,solver)
                    self.each_wk_action_reward[wk_idx][tuple(chosen_paths)] = rl_egr

            else:
                rl_egr = self.compute_egr(actions,wk_idx,network,solver)
                self.each_wk_action_reward[wk_idx] = {}
                self.each_wk_action_reward[wk_idx][tuple(chosen_paths)] = rl_egr
        except:
            rl_egr = self.compute_egr(actions,wk_idx,network,solver)
            self.each_wk_action_reward[wk_idx] = {}
            self.each_wk_action_reward[wk_idx][tuple(chosen_paths)] = rl_egr

#         print("rl gave us egr ",rl_egr)
#         for u,paths in network.each_wk_each_k_each_user_pair_id_paths[wk_idx][0].items():
#             print("for wk %s k %s u %s paths %s"%(wk_idx,0,u,paths))
        
        try:
            if wk_idx in self.each_wk_optimal_egr:
                optimal_egr = self.each_wk_optimal_egr[wk_idx]
                optimal_egr_paths = self.each_wk_optimal_egr_paths[wk_idx]
            else:
                optimal_egr,optimal_paths = self.compute_optimal_egr(wk_idx,network,solver)
                self.each_wk_optimal_egr[wk_idx]= optimal_egr
                self.each_wk_optimal_egr_paths[wk_idx] =optimal_paths
        except:
            optimal_egr,optimal_paths = self.compute_optimal_egr(wk_idx,network,solver)
            self.each_wk_optimal_egr[wk_idx]= optimal_egr
            self.each_wk_optimal_egr_paths[wk_idx] =optimal_paths
#         if optimal_egr==0 or rl_egr>optimal_egr:
#             for k in network.each_wk_organizations[wk_idx]:
#                 for u in network.each_wk_each_k_user_pair_ids[wk_idx][k]:
#                     print("k %s u %s"%(k,u))
#                     for p in network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][u]:
#                         print("wk %s k %s w %s user %s w %s path %s edges %s"%(wk_idx,k,network.each_wk_k_weight[wk_idx][k],u,network.each_wk_k_u_weight[wk_idx][k][u],p,network.set_of_paths[p]))
#             pdb.set_trace()
        reward = rl_egr
        #else:
            #reward=0
        
        #print("workload %s is rl_egr %s optimal egr %s reward %s "%(wk_idx,rl_egr,optimal_egr,reward))
#         if reward >1:
#             print("actions are  %s egr is %s"%(actions,rl_egr))
#             print("actions of optimal are %s egr is %s "%(optimal_egr_paths,optimal_egr))
#             #time.sleep(10)
            #pdb.set_trace()
        
        return reward
    
    def find_combos(self,arr,k):
        combos = list(combinations(arr, k))
        return combos
    def get_all_possible_actions(self,each_user_paths,k):
        all_user_paths = []
        for user,paths in each_user_paths.items():
            paths = self.find_combos(paths,k)
            if paths:
                all_user_paths.append(paths)

        all_possible_actions = list(itertools.product(*all_user_paths))
        return all_possible_actions
    def compute_optimal_egr(self,wk_idx,network,solver):
        max_egr = 0
        network.each_wk_each_k_each_user_pair_id_paths = {}
        each_user_pair_paths = {}
        for k,user_pair_ids in network.each_wk_each_k_user_pair_ids[wk_idx].items():
            for user_pair in user_pair_ids:
                each_user_pair_paths[user_pair] = network.each_pair_paths[user_pair]
                
        all_possible_actions = self.get_all_possible_actions(each_user_pair_paths,network.num_of_paths)
        for action in all_possible_actions:
            actions = []
            for item in action:
                for i in item:
                    actions.append(i)
            egr = self.compute_egr(actions,wk_idx,network,solver)
            if egr >max_egr:
                max_egr = egr
                optimal_paths = actions
        if max_egr>0:
            return max_egr,optimal_paths
        else:
            return 0,[]
    def get_all_trainig_epochs(self):
        indx = 1
        epochs = [1,39]
        while(max(epochs)<719):
            epochs.append(max(epochs)+40)
        while(max(epochs)<20000):
            epochs.append(max(epochs)+200)
        return (list(epochs))
    def advantage(self, tm_idx, reward):
        if tm_idx not in self.baseline:
            return reward

        total_v, cnt = self.baseline[tm_idx]
        
        #print(reward, (total_v/cnt))

        return reward - (total_v/cnt)

    def update_baseline(self, tm_idx, reward):
        if tm_idx in self.baseline:
            total_v, cnt = self.baseline[tm_idx]

            total_v += reward
            cnt += 1

            self.baseline[tm_idx] = (total_v, cnt)
        else:
            self.baseline[tm_idx] = (reward, 1)

   
        
    def evaluate(self,wk_idx,network,solver,scheme,actions):
        egr = 0
        if scheme =="RL":
            egr = self.compute_egr(actions,wk_idx,network,solver)
        elif scheme  in ["hop","EGR","EGRsquare"]:
            actions = []
            for k in network.each_wk_organizations[wk_idx]:
                for user_pair_id in network.each_wk_each_k_user_pair_ids[wk_idx][k]:
                    paths = network.each_scheme_each_user_pair_paths[scheme][user_pair_id]
                    for path in paths:
                        actions.append(path)
            egr = self.compute_egr(actions,wk_idx,network,solver)
        if scheme =="Optimal":
            egr,_ = self.compute_optimal_egr(wk_idx,network,solver)
#             if egr==0:
#                 for k in network.each_wk_organizations[wk_idx]:
#                     for u in network.each_wk_each_k_user_pair_ids[wk_idx][k]:
#                         for p in network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][u]:
#                             print("wk %s k %s w %s user %s w %s path %s edges %s"%(wk_idx,k,network.each_wk_k_weight[wk_idx][k],u,network.each_wk_k_u_weight[wk_idx][k][u],p,network.set_of_paths[p]))
#                 pdb.set_trace()
        return egr    
        
                
                


# In[4]:


# import numpy as np

# state = np.zeros(( 6), dtype=np.float32)   # state  [1, link, B]
# input_dims =  state.shape
# print(state)
# print(state.shape)


# In[ ]:





# In[ ]:





# In[6]:


# # wk_idx 9 one set of paths [22, 70, 119] for user pair 24 
# # wk_idx 9 one set of paths [23, 71, 120] for user pair 25 
# # wk_idx 9 one set of paths [] for user pair 26 
# import itertools
# from itertools import combinations
# def find_combos(arr,k):
#         combos = list(combinations(arr, k))
#         return combos
# def get_all_possible_actions(each_user_paths,k):
#     all_user_paths = []
#     for user,paths in each_user_paths.items():
#         paths = find_combos(paths,k)
#         if paths:
#             all_user_paths.append(paths)
#     print("all_user_paths",all_user_paths)
#     all_possible_actions = list(itertools.product(*all_user_paths))
#     return all_possible_actions
# k=1
# each_user_paths = {24:[22, 70, 119],25:[23, 71, 120],26:[]}
# all_possible_actions = get_all_possible_actions(each_user_paths,k)
# # print("all_possible_actions",all_possible_actions)
# for actions in all_possible_actions:
#     print("one possible action ",actions)


# In[ ]:




