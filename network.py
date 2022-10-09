#!/usr/bin/env python
# coding: utf-8

# In[2]:


import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import math as mt
import csv
import os
import random


# In[ ]:


class Network:
    def __init__(self,config,topology_file,edge_fidelity_range,link_cost_metric):
        self.data_dir = './data/'
        self.topology_file = topology_file
        
        
        self.set_E = []
        self.each_id_pair ={}
        self.pair_id = 0
        
        self.min_edge_fidelity = float(config.min_edge_fidelity)
        self.max_edge_fidelity = float(edge_fidelity_range)
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
        self.each_u_weight={}
        self.each_path_legth = {}
        self.K= []
        self.each_k_u_all_paths = {}
        self.each_k_u_all_disjoint_paths={}
        self.number_of_user_pairs = int(config.number_of_user_pairs)
        self.num_of_organizations = int(config.num_of_organizations)
        self.each_k_fidelity_threshold = {}
        self.each_k_path_path_id = {}
        self.each_k_weight = {}
        self.each_k_u_weight = {}
        self.max_edge_capacity = 0
        self.load_topology(link_cost_metric)
    
    def load_topology(self,link_cost_metric):
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
            random_fidelity = random.uniform(self.min_edge_fidelity,self.max_edge_fidelity)
            self.each_edge_fidelity[(int(s),int(d))] = round(random_fidelity,3)
            self.each_edge_fidelity[(int(d),int(s))] = round(random_fidelity,3)
            edge_capacity = round(float(c),3)
            self.each_edge_capacity[(int(s),int(d))] = edge_capacity
            if edge_capacity > self.max_edge_capacity:
                self.max_edge_capacity = edge_capacity
            if link_cost_metric =="hop":
                weight1=1
                weight2 = 1
            elif link_cost_metric =="EGR":
                weight1=1/edge_capacity
                weight2 = 1/edge_capacity
            elif link_cost_metric =="EGRsquare":
                weight1=1/(edge_capacity**2)
                weight2 = 1/(edge_capacity**2)
            elif link_cost_metric =="Bruteforce":
                weight1=1
                weight2 = 1
            self.g.add_edge(int(s),int(d),weight=weight1)
            self.g.add_edge(int(d),int(s),weight=weight2)
        f.close()
    def set_organizations(self):
        for i in range(self.num_of_organizations):
            self.K.append(i)
    def set_each_organization_weight(self):
        for k in self.K:
            weight = random.uniform(0.1,1)
            self.each_k_weight[k] = weight
            
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
    def set_each_user_pair_weight(self):
        for k, user_pairs in self.each_k_user_pairs.items():
            for u in user_pairs:
                weight = random.uniform(0,1)
                try:
                    self.each_k_u_weight[k][u] = weight
                except:
                    self.each_k_u_weight[k] ={}
                    self.each_k_u_weight[k][u] = weight
                    
        
    def set_each_k_user_pair_all_paths(self):
        #print("we are getting all paths for pairs ",pairs)
        for k,user_pairs_ids in self.each_k_user_pairs.items():
            for user_pair_id in user_pairs_ids:
                user_pair = self.each_id_pair[user_pair_id]
                shortest_paths = nx.all_shortest_paths(self.g,source=user_pair[0],target=user_pair[1], weight='weight')
                paths = []
                for p in shortest_paths:
                    paths.append(p)
                try:
                    self.each_k_u_all_paths[k][user_pair_id] = paths
                except:
                    self.each_k_u_all_paths[k]={}
                    self.each_k_u_all_paths[k][user_pair_id] = paths
        for k,user_pairs_ids in self.each_k_user_pairs.items():
            for user_pair_id in user_pairs_ids:
                user_pair = self.each_id_pair[user_pair_id]
                if user_pair[0]==user_pair[1]:
                    paths = [[user_pair[0]]]
                else:
                    shortest_disjoint_paths = nx.edge_disjoint_paths(self.g,s=user_pair[0],t=user_pair[1])
                    import pdb
                    paths = []
                    for p in shortest_disjoint_paths:
                        paths.append(p)
                try:
                    self.each_k_u_all_disjoint_paths[user_pair_id] = paths
                except:
                    self.each_k_u_all_disjoint_paths[k]={}
                    self.each_k_u_all_disjoint_paths[k][user_pair_id] = paths
        
 
    def set_paths(self):
        for k,user_pairs_ids in self.each_k_user_pairs.items():
            for user_pair_id in user_pairs_ids:
                this_user_pair_has_one_real_path = False
                paths = self.each_k_u_all_paths[k][user_pair_id]
                path_counter = 0
                for path in paths:
                    if path_counter <self.num_of_paths:
                        node_indx = 0
                        path_edges = []
                        for node_indx in range(len(path)-1):
                            path_edges.append((path[node_indx],path[node_indx+1]))
                            node_indx+=1
                        if self.get_this_path_fidelity(path_edges)>=0.6:
                            this_user_pair_has_one_real_path = True
                            self.path_existance_flag= True
                            self.set_each_path_length(self.path_counter_id,path)
                            self.set_of_paths[self.path_counter_id] = path_edges
                            try:
                                self.each_k_path_path_id[k][tuple(path_edges)] = self.path_counter_id
                            except:
                                self.each_k_path_path_id[k]={}
                                self.each_k_path_path_id[k][tuple(path_edges)] = self.path_counter_id
                            try:
                                self.each_k_u_paths[k][user_pair_id].append(self.path_counter_id)
                            except:
                                try:
                                    self.each_k_u_paths[k][user_pair_id]=[self.path_counter_id]
                                except:
                                    self.each_k_u_paths[k]={}
                                    self.each_k_u_paths[k][user_pair_id]=[self.path_counter_id]
                            self.path_counter_id+=1  
                            path_counter+=1
                if not this_user_pair_has_one_real_path:
                    try:
                        self.each_k_u_paths[k][user_pair_id]=[]
                    except:
                        self.each_k_u_paths[k]={}
                        self.each_k_u_paths[k][user_pair_id]=[]
    
    def set_paths_in_the_network(self):
        self.reset_pair_paths()
        self.set_each_k_user_pair_all_paths()
        self.set_paths()
        self.set_each_path_basic_fidelity()
        """we set the required EPR pairs to achieve each fidelity threshold"""
        self.set_required_EPR_pairs_for_each_path_each_fidelity_threshold()
    def reset_pair_paths(self):
        self.set_of_paths = {}
        self.each_k_u_paths = {}
        self.each_k_u_disjoint_paths = {}
        self.each_k_path_path_id = {}

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
    
    def get_k_threshold(self,k):
        return self.each_k_fidelity_threshold[k]
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
    
            
    
    
        
   
    
    def set_each_k_fidelity_threshold(self):
        self.each_k_fidelity_threshold = {}
        possible_thresholds_based_on_given_range = []
        
        possible_thresholds_based_on_given_range.append(self.fidelity_threshold_range)
        for k in self.each_k_user_pairs:
            try:
                self.each_k_fidelity_threshold[k]= possible_thresholds_based_on_given_range[random.randint(0,len(possible_thresholds_based_on_given_range)-1)]
            except:
                self.each_k_fidelity_threshold[k] = {}
                self.each_k_fidelity_threshold[k] = possible_thresholds_based_on_given_range[random.randint(0,len(possible_thresholds_based_on_given_range)-1)]
    
        
    
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
    
    
    def set_required_EPR_pairs_for_each_path_each_fidelity_threshold(self):
        targets = []
        for k,target_fidelity in self.each_k_fidelity_threshold.items():
            if target_fidelity not in targets:
                targets.append(target_fidelity)
        targets.append(0.6)
        targets.sort()
        for path,path_basic_fidelity in self.each_path_basic_fidelity.items():
            #print("for path %s with lenght %s fidelity %s"%(path,self.each_path_legth[path],path_basic_fidelity))
            try:
                if path_basic_fidelity in self.global_each_basic_fidelity_target_fidelity_required_EPRs:
                    
                    for target in targets:
                        
                        #print("getting required rounds for initial F %s to target %s path length %s"%(path_basic_fidelity,target,self.each_path_legth[path]))
                        n_avg = self.global_each_basic_fidelity_target_fidelity_required_EPRs[path_basic_fidelity][target]
                        #print("we got ",n_avg)
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
    
    def get_required_edge_level_purification_EPR_pairs(self,edge,p,F):
        longest_p_lenght   = 0
        max_F_threshold = 0
        for k,users in self.each_k_user_pairs.items():
            for u in users:
                if p in self.each_k_u_paths[k][u]:
                    path = self.set_of_paths[p]
                    longest_p_lenght = len(path)
                    if self.each_k_fidelity_threshold[k] > max_F_threshold:
                        max_F_threshold = self.each_k_fidelity_threshold[k]
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





# In[10]:


# print("start ...")
# file1 = open("data/SURFnet2", "w")
# file1.writelines("Link_index	Source	Destination	Capacity(EPRps)"+"\n")
# f = open("data/SURFnet", 'r')
# header = f.readline()
# for line in f:
#     line = line.strip()
#     link = line.split('\t')
#     print(line,link)
#     i, s, d,  c = link
#     c = random.randint(1,400)
#     file1.writelines(str(i)+"\t"+str(s)+"\t"+str(d)+"\t"+str(c)+"\n")
# print("done!")


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




