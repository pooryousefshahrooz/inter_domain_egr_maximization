#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import networkx as nx
from itertools import islice
import matplotlib.pyplot as plt
import random
import time
import math as mt
import csv
import os
import random
from solver import Solver
import pdb


# In[1]:


class Network:
    def __init__(self,config,edge_capacity_bound,training_flag):
        self.data_dir = './data/'
        self.topology_file = self.data_dir+config.topology_file
        
        self.training = training_flag
        self.set_E = []
        self.each_id_pair ={}
        self.pair_id = 0

        self.min_edge_fidelity = float(config.min_edge_fidelity)
        self.max_edge_fidelity = float(config.max_edge_fidelity)
        self.num_of_paths = int(config.num_of_paths)
        
        
        
        self.each_pair_id  ={}
       
        self.set_of_paths = {}
        self.each_u_paths = {}
        self.each_n_f_purification_result = {}
        self.each_edge_target_fidelity = {}
        self.each_u_all_real_paths = {}
        self.each_u_all_real_disjoint_paths = {}
        self.each_u_paths = {}
        self.nodes = []
        self.oracle_for_target_fidelity = {}
        self.global_each_basic_fidelity_target_fidelity_required_EPRs = {}
        self.all_basic_fidelity_target_thresholds = []
        self.path_counter_id = 0
        self.pair_id = 0
        self.q_value = 1
        self.each_u_weight={}
        self.each_path_legth = {}
        self.K= []
        self.each_k_u_all_paths = {}
        self.each_k_u_all_disjoint_paths={}
        self.each_wk_each_k_each_user_pair_id_paths = {}
        self.number_of_user_pairs = int(config.number_of_user_pairs)
        self.num_of_organizations = int(config.num_of_organizations)
        self.each_wk_k_fidelity_threshold = {}
        self.each_k_path_path_id = {}
        self.each_wk_each_k_user_pairs = {}
        self.each_wk_each_k_user_pair_ids = {}
        self.each_user_pair_all_paths = {}
        self.each_k_weight = {}
        self.each_k_u_weight = {}
        self.each_pair_paths = {}
        self.each_scheme_each_user_pair_paths = {}
        self.max_edge_capacity = 0
        self.setting_basic_fidelity_flag = False
        self.each_link_cost_metric = "Hop"
        self.link_cost_metrics = []
        for scheme in config.schemes:
            if scheme in ["EGR","Hop","EGRSquare"]:
                self.link_cost_metrics.append(scheme)
                
        self.cut_off_for_path_searching = int(config.cut_off_for_path_searching)
        self.load_topology(edge_capacity_bound)
        
        self.set_wk_organizations_user_pairs_from_file()
        self.set_wk_organizations()
        
        self.set_each_wk_organization_weight()
        
        self.set_each_wk_user_pair_weight()
        
        
    
    def evaluate_shortest_path_routing(self,link_cost_metric):
        """this function evaluates the entanglement generation rate using 
        shortest paths computed based on the given link cost metric"""
        self.each_wk_each_k_each_user_pair_id_paths = {}
        solver = Solver()
        self.each_link_cost_metric =link_cost_metric 
        self.set_link_weight(link_cost_metric)
        for wk_idx in self.work_loads:
            self.set_paths_in_the_network(wk_idx)
            if not self.setting_basic_fidelity_flag:
                self.set_each_path_basic_fidelity()
                self.setting_basic_fidelity_flag = True
                """we set the required EPR pairs to achieve each fidelity threshold"""
                self.set_required_EPR_pairs_for_each_path_each_fidelity_threshold(wk_idx)       
            # calling the IBM CPLEX solver to solve the optimization problem
            egr = solver.CPLEX_maximizing_EGR(wk_idx,self)
            print("for work_load %s we have egr as %s"%(wk_idx,egr))
    
    
    def get_each_user_all_paths(self,wk_idx):
        """this function will set all the paths of each user pair for the given work load"""
        self.each_user_pair_all_paths = {}
        for k, user_pair_ids in self.each_wk_each_k_user_pair_ids[wk_idx].items():
            for user_pair_id in user_pair_ids:
                for path_id in self.each_pair_paths[user_pair_id]:
                    try:
                        self.each_user_pair_all_paths[user_pair_id].append(path_id)
                    except:
                        self.each_user_pair_all_paths[user_pair_id] = [path_id]
            
        
    def generate_chromosome(self,wk_idx):
        """this function generates a population from finite number of chromosomes"""
        # we use the data structure filled in function get_each_user_all_paths() for this
        self.chromosomes = [[1,2,3]]
        pass
        
    def crossover_operation(self):
        """this function applies crossover operator to the given population"""
        pass
    def mutation_operation(self):
        """this function applies mutation operator to the given population"""
        pass
    def set_paths_from_chromosome(self,chromosome):
        """this function uses the information in the chromosome 
        to set the paths to the data structure that will be used by solver"""
        pass
        
    def evaluate_genetic_algorithm_for_path_selection(self):
        """this function implements the main work flow of the genetic algorithm"""
        solver = Solver()
        runs_of_genetic_algorithm = 0
        genetic_algorithm_running_flag = True
        max_runs_of_genetic_algorithm = 100 # maximum number of populations during genetic algorithm search
        for wk_idx in self.work_loads:# Each work load includes a different set of user pairs in the network 
            """we set the set of all paths (all n shortest paths using different link cost metrics)"""
            self.get_each_user_all_paths(wk_idx)
            self.generate_chromosome(wk_idx)
            max_fitness_value = 0
            best_chromosome = ""
            while(genetic_algorithm_running_flag):
                self.crossover_operation()
                self.mutation_operation()
                for chromosome in self.chromosomes:
                    self.set_paths_from_chromosome(chromosome)
                    fitness_value  = solver.CPLEX_maximizing_EGR(wk_idx,self)
                    # we store the best fitness value and the chromosome associated to it
                   # in our final loop of genetic algorithm
                    if runs_of_genetic_algorithm >= max_runs_of_genetic_algorithm:
                        genetic_algorithm_running_flag = False
                        if fitness_value>max_fitness_value:
                            max_fitness_value = fitness_value
                            best_chromosome = chromosome
            runs_of_genetic_algorithm+=1
        
            print("for work load %s we have entanglement generation rate of %s using these paths %s"%
                  (wk_idx,max_fitness_value,best_chromosome))
    
    
    def evaluate_rl_for_path_selection(self):
        pass
                                    
    def load_topology(self,each_edge_capacity_upper_bound):
        self.set_E=[]
        self.each_edge_capacity={}
        self.nodes = []
        self.each_edge_fidelity = {}
        self.link_idx_to_sd = {}
        self.link_sd_to_idx = {}
        self.g = nx.Graph()
        print('[*] Loading topology...', self.topology_file)
        f = open(self.topology_file, 'r')
        header = f.readline()
        for line in f:
            line = line.strip()
            link = line.split('\t')
            #print(line,link)
            i, s, d,  c = link
            if int(s) not in self.nodes:
                self.nodes.append(int(s))
            if int(d) not in self.nodes:
                self.nodes.append(int(d))
            self.set_E.append((int(s),int(d)))
#             random_fidelity = random.uniform(self.min_edge_fidelity,self.max_edge_fidelity)
#             self.each_edge_fidelity[(int(s),int(d))] = round(random_fidelity,3)
#             self.each_edge_fidelity[(int(d),int(s))] = round(random_fidelity,3)
#             edge_capacity = round(float(c),3)
            
            
            self.max_edge_capacity = each_edge_capacity_upper_bound
            edge_capacity  = random.uniform(1,each_edge_capacity_upper_bound)
            edge_capacity = float(c)
           
            self.each_edge_capacity[(int(s),int(d))] = edge_capacity
            self.each_edge_capacity[(int(d),int(s))] = edge_capacity
            self.g.add_edge(int(s),int(d),capacity=edge_capacity,weight=1)
            self.g.add_edge(int(d),int(s),capacity=edge_capacity,weight=1)
        f.close()
        
    def get_path_info(self):
        self.all_user_pairs_across_wks = []
        self.each_pair_paths = {}
        self.each_scheme_each_user_pair_paths = {}
        set_of_all_paths = []
        self.path_counter_id = 0
        self.path_counter = 0
        for wk,ks in self.each_wk_organizations.items():
            for k in ks:
                #print("we have %s users for k %s"%(self.each_wk_k_user_pairs[wk][k],k))
                for u in self.each_wk_each_k_user_pairs[wk][k]:
                    if u not in self.all_user_pairs_across_wks:
                        self.all_user_pairs_across_wks.append(u)
        
        for link_cost_metric in self.link_cost_metrics:
            for user_pair in self.all_user_pairs_across_wks:
                user_pair_id = self.each_pair_id[user_pair]
                self.each_pair_paths[user_pair_id]=[]
        
        for link_cost_metric in self.link_cost_metrics:
            self.each_link_cost_metric =link_cost_metric 
            self.set_link_weight(link_cost_metric)
            for user_pair in self.all_user_pairs_across_wks:
                user_pair_id = self.each_pair_id[user_pair]
                paths = self.get_paths_between_user_pairs(user_pair)
                having_atleast_one_path_flag = False
                path_flag = False
                for path in paths:
                    node_indx = 0
                    path_edges = []
                    for node_indx in range(len(path)-1):
                        path_edges.append((path[node_indx],path[node_indx+1]))
                        node_indx+=1
                    if self.get_basic_fidelity(path_edges)>0.65:
                        path_flag= True
                        try:
                            self.each_scheme_each_user_pair_paths[link_cost_metric][user_pair_id].append(self.path_counter_id)
                        except:
                            try:
                                self.each_scheme_each_user_pair_paths[link_cost_metric][user_pair_id]=[self.path_counter_id]
                            except:
                                self.each_scheme_each_user_pair_paths[link_cost_metric]={}
                                self.each_scheme_each_user_pair_paths[link_cost_metric][user_pair_id]=[self.path_counter_id]
                        
                        
                        having_atleast_one_path_flag = True
                        set_of_all_paths.append(path_edges)
                        self.set_each_path_length(self.path_counter_id,path)
                        self.set_of_paths[self.path_counter_id] = path_edges
                        
                        try:
                            self.each_pair_paths[user_pair_id].append(self.path_counter_id)
                        except:
                            self.each_pair_paths[user_pair_id] = [self.path_counter_id]

                        self.path_counter_id+=1  
                        self.path_counter+=1
                
                
                if not path_flag:
                    try:
                        self.each_scheme_each_user_pair_paths[link_cost_metric][user_pair_id]=[]
                    except:
                        self.each_scheme_each_user_pair_paths[link_cost_metric]={}
                        self.each_scheme_each_user_pair_paths[link_cost_metric][user_pair_id] = []
                        
             
        
    def get_paths_between_user_pairs(self,user_pair):
        
        return self.k_shortest_paths(user_pair[0], user_pair[1], self.cut_off_for_path_searching,"weight")
            
    def set_edge_fidelity(self,edge_fidelity_range):
        
        self.max_edge_fidelity = edge_fidelity_range
        self.min_edge_fidelity = edge_fidelity_range
        for edge in self.g.edges:
            random_fidelity = random.uniform(self.min_edge_fidelity,self.max_edge_fidelity)
            self.each_edge_fidelity[edge] = round(random_fidelity,3)
            self.each_edge_fidelity[(int(edge[1]),int(edge[0]))] = round(random_fidelity,3)
            
    def set_link_weight(self,link_cost_metric):
        for edge in self.g.edges:
            edge_capacity = self.each_edge_capacity[edge]
            if link_cost_metric =="Hop":
                weight1=1
                weight2 = 1
                self.g[edge[0]][edge[1]]['weight']=weight1
                self.g[edge[1]][edge[0]]['weight']= weight2
            elif link_cost_metric =="EGR":
                weight1=1/edge_capacity
                weight2 = 1/edge_capacity
                self.g[edge[0]][edge[1]]['weight']=weight1
                self.g[edge[1]][edge[0]]['weight']= weight2
            elif link_cost_metric =="EGRSquare":
                weight1=1/(edge_capacity**2)
                weight2 = 1/(edge_capacity**2)
                self.g[edge[0]][edge[1]]['weight']=weight1
                self.g[edge[1]][edge[0]]['weight']= weight2
            elif link_cost_metric =="Bruteforce":
                weight1=1
                weight2 = 1
                self.g[edge[0]][edge[1]]['weight']=1
                self.g[edge[1]][edge[0]]['weight']= 1
        
    def set_wk_organizations(self):
        self.each_wk_organizations={}
        for wk in self.work_loads:
            for i in range(self.num_of_organizations):
                try:
                    self.each_wk_organizations[wk].append(i)
                except:
                    self.each_wk_organizations[wk]=[i]
                
    def set_organizations(self):
        self.K=[]
        for i in range(self.num_of_organizations):
            self.K.append(i)
    def set_each_wk_organization_weight(self):
        self.each_wk_k_weight = {}
        for wk,ks in self.each_wk_organizations.items():
            if len(ks)==1:
                weight = 1
                try:
                    self.each_wk_k_weight[wk][ks[0]] = weight
                except:
                    self.each_wk_k_weight[wk]= {}
                    self.each_wk_k_weight[wk][ks[0]] = weight
            else:
                for k in ks:
                    weight = random.uniform(0.1,1)
                    self.each_wk_k_weight[wk][k] = weight
    def set_each_organization_weight(self):
        self.each_k_weight = {}
        if len(self.K)==1:
            weight = 1
            self.each_k_weight[k] = weight
        else:
            for k in self.K:
                weight = random.uniform(0.1,1)
                self.each_k_weight[k] = weight
    def set_wk_organizations_user_pairs_from_file(self):
        self.each_wk_each_k_user_pair_ids = {}
        self.each_wk_each_k_user_pairs = {}
        self.pair_id = 0
        self.each_id_pair ={}
        self.each_pair_id={}
        self.work_loads=[]
        """we assume there is only one organization for now!"""
        k = 0
        num_nodes = len(self.nodes)
        if self.training:
            f = open(self.topology_file+"WK2", 'r')
        else:
            f = open(self.topology_file+"WK", 'r')
        self.work_load_counter = 0
        all_active_user_pairs_acros_wks = []
        for line in f:
            self.work_loads.append(self.work_load_counter)
            volumes = line.strip().split(' ')
            volumes = line.strip().split(' ')
            total_volume_cnt = len(volumes)
            for v in range(total_volume_cnt):
                i = int(v/num_nodes)
                j = v%num_nodes
                if float(volumes[v])!=0:
                    try:
                        self.each_wk_each_k_user_pairs[self.work_load_counter][k].append((i,j))
                    except:
                        self.each_wk_each_k_user_pairs[self.work_load_counter]={}
                        self.each_wk_each_k_user_pairs[self.work_load_counter][k]=[(i,j)]
            
            self.work_load_counter+=1      
        for wk,k_users in self.each_wk_each_k_user_pairs.items():
            for k,selected_fixed_user_pairs in k_users.items():
                selected_ids = []
                for pair in selected_fixed_user_pairs:
                    if pair not in self.each_pair_id:
                        self.each_id_pair[self.pair_id] = pair
                        self.each_pair_id[pair] = self.pair_id
                        selected_ids.append(self.pair_id)
                        self.pair_id+=1
                    else:
                        pair_id = self.each_pair_id[pair]
                        selected_ids.append(pair_id)
                try:
                    self.each_wk_each_k_user_pair_ids[wk][k]= selected_ids
                except:
                    try:
                        self.each_wk_each_k_user_pair_ids[wk][k]= selected_ids
                    except:
                        self.each_wk_each_k_user_pair_ids[wk]={}
                        self.each_wk_each_k_user_pair_ids[wk][k]= selected_ids
    def set_organizations_user_pairs(self):
        self.each_k_user_pairs = {}
        self.pair_id = 0
        self.each_id_pair ={}
        self.each_pair_id={}
        for k in self.K:
            candidate_user_pairs = []
            while(len(candidate_user_pairs)<self.number_of_user_pairs):
                for src in self.nodes:
                    for dst in self.nodes:
                        if src!=dst and (src,dst) not in self.set_E and (dst,src) not in self.set_E:
                            if (src,dst) not in candidate_user_pairs and (dst,src) not in candidate_user_pairs:
                                candidate_user_pairs.append((src,dst))
            selected_fixed_user_pairs = []
            while(len(selected_fixed_user_pairs)<self.number_of_user_pairs):
                user_pair = candidate_user_pairs[random.randint(0,len(candidate_user_pairs)-1)]
                if user_pair not in selected_fixed_user_pairs:
                    selected_fixed_user_pairs.append(user_pair)
            selected_ids = []
            #print("selected_fixed_user_pairs",selected_fixed_user_pairs)
            for pair in selected_fixed_user_pairs:
                self.each_id_pair[self.pair_id] = pair
                self.each_pair_id[pair] = self.pair_id
                selected_ids.append(self.pair_id)
                self.pair_id+=1
            try:
                self.each_k_user_pairs[k]= selected_ids
            except:
                self.each_k_user_pairs[k]= selected_ids
    def set_each_wk_user_pair_weight(self):
        self.each_wk_k_u_weight ={}
        for wk,k_us in  self.each_wk_each_k_user_pair_ids.items():
            for k, user_pairs in k_us.items():
                for u in user_pairs:
                    weight = random.uniform(0.1,1)
                    weight = 1
                    try:
                        self.each_wk_k_u_weight[wk][k][u] = weight
                    except:
                        try:
                            self.each_wk_k_u_weight[wk][k] ={}
                            self.each_wk_k_u_weight[wk][k][u] = weight
                        except:
                            self.each_wk_k_u_weight[wk] ={}
                            self.each_wk_k_u_weight[wk][k] = {}
                            self.each_wk_k_u_weight[wk][k][u] = weight
        
    def set_each_k_user_pair_paths(self,wk_idx):
        """we set self.num_of_paths for each user pair of each organization """
        for k,user_pair_ids in self.each_wk_each_k_user_pair_ids[wk_idx].items():
            for user_pair_id in user_pair_ids:
                user_pair = self.each_id_pair[user_pair_id]
                having_at_least_one_path_flag = False
                for path in self.k_shortest_paths(user_pair[0], user_pair[1], self.num_of_paths,"weight"):
                    node_indx = 0
                    path_edges = []
                    for node_indx in range(len(path)-1):
                        path_edges.append((path[node_indx],path[node_indx+1]))
                        node_indx+=1
                    if self.get_this_path_fidelity(path_edges)>=0.6:
                        self.set_each_path_length(self.path_counter_id,path)
                        self.set_of_paths[self.path_counter_id] = path_edges
                        path_id=self.path_counter_id 
                        try:
                            if len(self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id])<network.num_of_paths:
                                having_at_least_one_path_flag = True
                                try:
                                    self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id].append(path_id)
                                except:
                                    try:
                                        self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id] = [path_id]
                                    except:
                                        try:
                                            self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k]={}
                                            self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id] = [path_id]
                                        except:
                                            self.each_wk_each_k_each_user_pair_id_paths[wk_idx]={}
                                            self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k]={}
                                            self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id] = [path_id]
                        except:
                            having_at_least_one_path_flag = True
                            try:
                                self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id] = [path_id]
                            except:
                                try:
                                    self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k]={}
                                    self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id] = [path_id]
                                except:
                                    self.each_wk_each_k_each_user_pair_id_paths[wk_idx]={}
                                    self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k]={}
                                    self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id] = [path_id]
                        self.path_counter_id+=1
                if not having_at_least_one_path_flag:
#                     print("this user %s did not have even one valid path "%(user_pair_id))
                    try:
                        self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id] = []
                    except:
                        try:
                            self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k]={}
                            self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id] = []
                        except:
                            self.each_wk_each_k_each_user_pair_id_paths[wk_idx]={}
                            self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k]={}
                            self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id] = []

#                     shortest_disjoint_paths = nx.edge_disjoint_paths(self.g,s=user_pair[0],t=user_pair[1])             
        
    def k_shortest_paths(self, source, target, k, weight):
        return list(
            islice(nx.shortest_simple_paths(self.g, source, target, weight=weight), k)
        )
    
    
    
    
    def set_paths_in_the_network(self,wk_idx):
        self.reset_pair_paths()
        self.set_each_k_user_pair_paths(wk_idx)
        self.set_each_path_basic_fidelity()
        """we set the required EPR pairs to achieve each fidelity threshold"""
        self.set_required_EPR_pairs_for_each_path_each_fidelity_threshold(wk_idx)
        
    def reset_pair_paths(self):
        self.path_id = 0
        self.path_counter_id = 0
        self.set_of_paths = {}
        self.each_k_u_paths = {}
        self.each_k_u_disjoint_paths = {}
        self.each_k_path_path_id = {}
        self.each_k_u_all_paths = {}
        self.each_k_u_all_disjoint_paths = {}
        self.each_k_path_path_id={}
        
    def set_each_user_pair_demands(self):
        self.each_t_each_request_demand = {}
        num_of_pairs= len(list(self.user_pairs))
        tm = spike_tm(num_of_pairs+1,num_spikes,spike_mean,1)
        
        traffic = tm.at_time(1)
        printed_pairs = []
        user_indx = 0
        for i in range(num_of_pairs):
            for j in range(num_of_pairs):
                if i!=j:
                    if (i,j) not in printed_pairs and (j,i) not in printed_pairs and user_indx<num_of_pairs:
                        printed_pairs.append((i,j))
                        printed_pairs.append((j,i))
#                             print("num_of_pairs %s time %s traffic from %s to %s is %s and user_indx %s"%(num_of_pairs, time,i,j,traffic[i][j],user_indx))
                        request = user_pairs[time][user_indx]
                        user_indx+=1
                        demand = max(1,traffic[i][j])
                        try:
                            self.each_u_request_demand[request] = demand
                        except:
                            self.each_u_demand[request] = demand
        for request in self.user_pairs[time]:
            try:
                self.each_u_demand[0][request] = 0
            except:
                self.each_u_demand[0]={}
                self.each_u_demand[0][request] = 0
    
    def get_each_wk_k_threshold(self,wk_idx,k):
        return self.each_wk_k_fidelity_threshold[wk_idx][k]
    def set_each_user_weight(self):
        for time,user_pairs in self.each_t_user_pairs.items():
            #print("for time ",time)
            for user in user_pairs:
                user_pair = self.each_id_pair[user]
                weight = 1
                try:
                    self.each_k_u_weight[k][u] = weight
                except:
                    self.each_u_weight[k]= {}
                    self.each_u_weight[k][u] = weight
        
    def reset_variables(self):
        self.each_id_pair ={}
        self.pair_id = 0
        self.max_edge_capacity = int(config.max_edge_capacity)
        self.min_edge_capacity = int(config.min_edge_capacity)
        self.min_edge_fidelity = float(config.min_edge_fidelity)
        self.max_edge_fidelity = float(config.max_edge_fidelity)
        self.num_of_paths = int(config.num_of_paths)
        self.path_selection_scheme = config.path_selection_scheme
        self.each_pair_id  ={}
       
        self.set_of_paths = {}
        self.each_u_paths = {}
        self.each_n_f_purification_result = {}
        self.each_edge_target_fidelity = {}
        self.each_u_all_real_paths = {}
        self.each_u_all_real_disjoint_paths = {}
        self.each_u_paths = {}
        self.nodes = []
        self.oracle_for_target_fidelity = {}
        self.each_k_path_path_id = {}
        self.global_each_basic_fidelity_target_fidelity_required_EPRs = {}
        self.all_basic_fidelity_target_thresholds = []
        self.path_counter_id = 0
        self.pair_id = 0
        self.each_u_weight={}
        self.each_path_legth = {}
        self.load_topology()
    
            
    
    
        
   
    
    def set_each_wk_k_fidelity_threshold(self):
        self.each_wk_k_fidelity_threshold = {}
        possible_thresholds_based_on_given_range = []
        
        possible_thresholds_based_on_given_range.append(self.fidelity_threshold_range)
        for wk,ks in self.each_wk_organizations.items():
            for k in ks:
                try:
                    self.each_wk_k_fidelity_threshold[wk][k]= possible_thresholds_based_on_given_range[random.randint(0,len(possible_thresholds_based_on_given_range)-1)]
                except:
                    self.each_wk_k_fidelity_threshold[wk]= {}
                    self.each_wk_k_fidelity_threshold[wk][k] = possible_thresholds_based_on_given_range[random.randint(0,len(possible_thresholds_based_on_given_range)-1)]
    
        
    
    def set_each_path_length(self,path_id,path):
        self.each_path_legth[path_id] = len(path)
    
    def get_next_fidelity_and_succ_prob_BBPSSW(self,F):
        succ_prob = (F+((1-F)/3))**2 + (2*(1-F)/3)**2
        output_fidelity = (F**2 + ((1-F)/3)**2)/succ_prob

        return output_fidelity, succ_prob

    def get_next_fidelity_and_succ_prob_DEJMPS(self,F1,F2,F3,F4):
        succ_prob = (F1+F2)**2 + (F3+F4)**2
        output_fidelity1 = (F1**2 + F2**2)/succ_prob
        output_fidelity2 = (2*F3*F4)/succ_prob
        output_fidelity3 = (F3**2 + F4**2)/succ_prob
        output_fidelity4 = (2*F1*F2)/succ_prob

        return output_fidelity1, output_fidelity2, output_fidelity3, output_fidelity4, succ_prob

    def get_avg_epr_pairs_BBPSSW(self,F_init,F_target):
        F_curr = F_init
        n_avg = 1.0
        while(F_curr < F_target):
            F_curr,succ_prob = get_next_fidelity_and_succ_prob_BBPSSW(F_curr)
            n_avg = n_avg*(2/succ_prob)
        return  n_avg

    def get_avg_epr_pairs_DEJMPS(self,F_init,F_target):
        F_curr = F_init
        F2 = F3 = F4 = (1-F_curr)/3
        n_avg = 1.0
        while(F_curr < F_target):
            F_curr,F2, F3, F4, succ_prob = self.get_next_fidelity_and_succ_prob_DEJMPS(F_curr, F2, F3, F4)
            n_avg = n_avg*(2/succ_prob)
            
        return  n_avg
    
    
    def set_required_EPR_pairs_for_each_path_each_fidelity_threshold(self,wk_idx):
        targets = []
        for k,target_fidelity in self.each_wk_k_fidelity_threshold[wk_idx].items():
            if target_fidelity not in targets:
                targets.append(target_fidelity)
        targets.append(0.6)
        targets.sort()
        counter = 0
        for path,path_basic_fidelity in self.each_path_basic_fidelity.items():
            counter+=1
            try:
                if path_basic_fidelity in self.global_each_basic_fidelity_target_fidelity_required_EPRs:
                    for target in targets:
                        n_avg = self.global_each_basic_fidelity_target_fidelity_required_EPRs[path_basic_fidelity][target]
                        try:
                            self.oracle_for_target_fidelity[path][target] = n_avg
                        except:
                            self.oracle_for_target_fidelity[path] = {}
                            self.oracle_for_target_fidelity[path][target] = n_avg
                else:
                    for target in targets:
                        n_avg = self.get_avg_epr_pairs_DEJMPS(path_basic_fidelity ,target)
                        try:
                            self.global_each_basic_fidelity_target_fidelity_required_EPRs[path_basic_fidelity][target] =n_avg 
                        except:
                            self.global_each_basic_fidelity_target_fidelity_required_EPRs[path_basic_fidelity]={}
                            self.global_each_basic_fidelity_target_fidelity_required_EPRs[path_basic_fidelity][target] =n_avg 
                        try:
                            self.oracle_for_target_fidelity[path][target] = n_avg
                        except:
                            self.oracle_for_target_fidelity[path] = {}
                            self.oracle_for_target_fidelity[path][target] = n_avg
            except:
                for target in targets:
                    n_avg  = self.get_avg_epr_pairs_DEJMPS(path_basic_fidelity ,target)
                    try:
                        self.global_each_basic_fidelity_target_fidelity_required_EPRs[path_basic_fidelity][target] =n_avg 
                    except:
                        self.global_each_basic_fidelity_target_fidelity_required_EPRs[path_basic_fidelity]={}
                        self.global_each_basic_fidelity_target_fidelity_required_EPRs[path_basic_fidelity][target] =n_avg                     
                    try:
                        self.oracle_for_target_fidelity[path][target] = n_avg
                    except:
                        self.oracle_for_target_fidelity[path] = {}
                        self.oracle_for_target_fidelity[path][target] = n_avg
    
    def get_required_edge_level_purification_EPR_pairs(self,edge,p,F,wk_idx):
        longest_p_lenght   = 0
        max_F_threshold = 0
        for k,users in self.each_wk_k_user_pairs[wk_idx].items():
            for u in users:
                if p in self.each_k_u_paths[k][u]:
                    path = self.set_of_paths[p]
                    longest_p_lenght = len(path)
                    if self.each_wk_k_fidelity_threshold[wk_idx][k] > max_F_threshold:
                        max_F_threshold = self.each_wk_k_fidelity_threshold[wk_idx][k]
        if longest_p_lenght==0:
            new_target = self.each_edge_fidelity[edge]
            
        else:
            new_target = (3*(4/3*max_F_threshold-1/3)**(1/longest_p_lenght)+1)/4
            
        edge_basic_fidelity = self.each_edge_fidelity[edge]
        try:
            if new_target in self.each_edge_target_fidelity[edge]:
                return self.each_edge_target_fidelity[edge][new_target]
            else:
                n_avg = self.get_avg_epr_pairs_DEJMPS(edge_basic_fidelity ,new_target)
                try:
                    self.each_edge_target_fidelity[edge][new_target] = n_avg
                except:
                    self.each_edge_target_fidelity[edge] = {}
                    self.each_edge_target_fidelity[edge][new_target] = n_avg
                return n_avg
        except:
            if longest_p_lenght==0:
                new_target = self.each_edge_fidelity[edge]
            else:
                new_target = (3*(4/3*max_F_threshold-1/3)**(1/longest_p_lenght)+1)/4
                
            edge_basic_fidelity = self.each_edge_fidelity[edge]
            n_avg = self.get_avg_epr_pairs_DEJMPS(edge_basic_fidelity ,new_target)
            try:
                self.each_edge_target_fidelity[edge][new_target] = n_avg
            except:
                self.each_edge_target_fidelity[edge] ={}
                self.each_edge_target_fidelity[edge][new_target] = n_avg
            return n_avg
    def get_required_purification_EPR_pairs(self,p,threshold):

        return self.oracle_for_target_fidelity[p][threshold]
        
    def get_real_longest_path(self,user_or_storage_pair,number_of_paths):
        all_paths=[]
        for path in nx.all_simple_paths(self.g,source=user_or_storage_pair[0],target=user_or_storage_pair[1]):
            #all_paths.append(path)

            node_indx = 0
            path_edges = []
            for node_indx in range(len(path)-1):
                path_edges.append((path[node_indx],path[node_indx+1]))
                node_indx+=1
            all_paths.append(path_edges)

        all_paths.sort(key=len,reverse=True)
        if len(all_paths)>=number_of_paths:
            return all_paths[:number_of_paths]
        else:
            return all_paths
                        
    def get_real_path(self,user_or_storage_pair_id):
        if self.path_selection_scheme=="shortest":
            path_selecion_flag = False
            path_counter = 1
            paths = []
            #print("user_or_storage_pair",user_or_storage_pair)
            #print("self.each_user_pair_all_real_paths[user_or_storage_pair]",self.each_user_pair_all_real_paths[user_or_storage_pair])
            for path in self.each_user_pair_all_real_paths[user_or_storage_pair_id]:
                #print("we can add this path",path)
                if path_counter<=self.num_of_paths:
                    node_indx = 0
                    path_edges = []
                    for node_indx in range(len(path)-1):
                        path_edges.append((path[node_indx],path[node_indx+1]))
                        node_indx+=1
                    paths.append(path_edges)

                path_counter+=1
        elif self.path_selection_scheme=="shortest_disjoint":
            path_selecion_flag = False
            path_counter = 1
            paths = []
            #print("self.each_user_pair_all_real_paths[user_or_storage_pair]",self.each_user_pair_all_real_paths[user_or_storage_pair])
            for path in self.each_user_pair_all_real_disjoint_paths[user_or_storage_pair_id]:
                #print("we can add this path",path)
                if path_counter<=self.num_of_paths:
                    node_indx = 0
                    path_edges = []
                    for node_indx in range(len(path)-1):
                        path_edges.append((path[node_indx],path[node_indx+1]))
                        node_indx+=1
                    paths.append(path_edges)

                path_counter+=1
            
        return paths
                    
      
                    
    def get_basic_fidelity(self,path_edges):
        if path_edges:
            basic_fidelity = 1/4+(3/4)*(4*self.each_edge_fidelity[path_edges[0]]-1)/3
            for edge in path_edges[1:]:
                basic_fidelity  = (basic_fidelity)*((4*self.each_edge_fidelity[edge]-1)/3)
            basic_fidelity = basic_fidelity
        else:
            print("Error")
            return 0.6
        return round(basic_fidelity,3)
        
    def set_each_path_basic_fidelity(self):
        self.each_path_basic_fidelity = {}
        for path,path_edges in self.set_of_paths.items():
            if path_edges:
                basic_fidelity = 1/4+(3/4)*(4*self.each_edge_fidelity[path_edges[0]]-1)/3
                for edge in path_edges[1:]:
                    basic_fidelity  = (basic_fidelity)*((4*self.each_edge_fidelity[edge]-1)/3)
                basic_fidelity = basic_fidelity
            else:
                print("Error")
                break
            self.each_path_basic_fidelity[path]= round(basic_fidelity,3)

   
    def get_edges(self):
        return self.set_E
    def get_this_path_fidelity(self,path_edges):
        if path_edges:
            basic_fidelity = 1/4+(3/4)*(4*self.each_edge_fidelity[path_edges[0]]-1)/3
            for edge in path_edges[1:]:
                basic_fidelity  = (basic_fidelity)*((4*self.each_edge_fidelity[edge]-1)/3)
        else:
            basic_fidelity  = 0.999
        return basic_fidelity

    def check_path_include_edge(self,edge,path):
        if edge in self.set_of_paths[path]:
            return True
        elif edge not  in self.set_of_paths[path]:
            return False

    def check_request_use_path(self,k,p):
        if p in self.each_u_paths[k]:
            return True
        else:
            return False
    def get_path_length(self,path):
        return self.each_path_legth[path]-1

    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:








# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




