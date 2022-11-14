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
import tensorflow as tf
from game import CFRRL_Game
from model import Model
from network import Network
from config import get_config
from solver import Solver
import time
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_agents',1, 'number of agents')
flags.DEFINE_string('baseline', 'avg', 'avg: use average reward as baseline, best: best reward as baseline')
flags.DEFINE_integer('num_iter', 20, 'Number of iterations each agent would run')
# print(FLAGS.num_agents)
import pdb
# pdb.set_trace()
GRADIENTS_CHECK=False


# In[ ]:





# In[ ]:





# In[ ]:


def central_agent(config, game, model_weights_queues, experience_queues):
    model = Model(config, game.state_dims, game.action_dim, game.max_moves, master=True)
    model.save_hyperparams(config)
    start_step = model.restore_ckpt()
    for step in tqdm(range(start_step, config.max_step), ncols=70, initial=start_step):
        model.ckpt.step.assign_add(1)
        model_weights = model.model.get_weights()

        for i in range(FLAGS.num_agents):
            model_weights_queues[i].put(model_weights)

        if config.method == 'actor_critic':
            #assemble experiences from the agents
            s_batch = []
            a_batch = []
            r_batch = []

            for i in range(FLAGS.num_agents):
                s_batch_agent, a_batch_agent, r_batch_agent = experience_queues[i].get()
              
                assert len(s_batch_agent) == FLAGS.num_iter, \
                    (len(s_batch_agent), len(a_batch_agent), len(r_batch_agent))

                s_batch += s_batch_agent
                a_batch += a_batch_agent
                r_batch += r_batch_agent
           
            assert len(s_batch)*game.max_moves == len(a_batch)
            #used shared RMSProp, i.e., shared g
            actions = np.eye(game.action_dim, dtype=np.float32)[np.array(a_batch)]
            value_loss, entropy, actor_gradients, critic_gradients = model.actor_critic_train(np.array(s_batch), 
                                                                    actions, 
                                                                    np.array(r_batch).astype(np.float32), 
                                                                    config.entropy_weight)
       
            if GRADIENTS_CHECK:
                for g in range(len(actor_gradients)):
                    assert np.any(np.isnan(actor_gradients[g])) == False, ('actor_gradients', s_batch, a_batch, r_batch, entropy)
                for g in range(len(critic_gradients)):
                    assert np.any(np.isnan(critic_gradients[g])) == False, ('critic_gradients', s_batch, a_batch, r_batch)

            if step % config.save_step == config.save_step - 1:
                model.save_ckpt(_print=True)
                
                #log training information
                actor_learning_rate = model.lr_schedule(model.actor_optimizer.iterations.numpy()).numpy()
                avg_value_loss = np.mean(value_loss)
                avg_reward = np.mean(r_batch)
                avg_entropy = np.mean(entropy)
            
                model.inject_summaries({
                    'learning rate': actor_learning_rate,
                    'value loss': avg_value_loss,
                    'avg reward': avg_reward,
                    'avg entropy': avg_entropy
                    }, step)
                print('lr:%f, value loss:%f, avg reward:%f, avg entropy:%f'%(actor_learning_rate, avg_value_loss, avg_reward, avg_entropy))

        elif config.method == 'pure_policy':
            #assemble experiences from the agents
            s_batch = []
            a_batch = []
            r_batch = []
            ad_batch = []

            for i in range(FLAGS.num_agents):
                s_batch_agent, a_batch_agent, r_batch_agent, ad_batch_agent = experience_queues[i].get()
              
                assert len(s_batch_agent) == FLAGS.num_iter, \
                    (len(s_batch_agent), len(a_batch_agent), len(r_batch_agent), len(ad_batch_agent))

                s_batch += s_batch_agent
                a_batch += a_batch_agent
                r_batch += r_batch_agent
                ad_batch += ad_batch_agent
           
            assert len(s_batch)*game.max_moves == len(a_batch)
            #used shared RMSProp, i.e., shared g
            actions = np.eye(game.action_dim, dtype=np.float32)[np.array(a_batch)]
            entropy, gradients = model.policy_train(np.array(s_batch), 
                                                      actions, 
                                                      np.vstack(ad_batch).astype(np.float32), 
                                                      config.entropy_weight)

            if GRADIENTS_CHECK:
                for g in range(len(gradients)):
                    assert np.any(np.isnan(gradients[g])) == False, (s_batch, a_batch, r_batch)
            
            if step % config.save_step == config.save_step - 1:
                model.save_ckpt(_print=True)
                
                #log training information
                learning_rate = model.lr_schedule(network.optimizer.iterations.numpy()).numpy()
                avg_reward = np.mean(r_batch)
                avg_advantage = np.mean(ad_batch)
                avg_entropy = np.mean(entropy)
                network.inject_summaries({
                    'learning rate': learning_rate,
                    'avg reward': avg_reward,
                    'avg advantage': avg_advantage,
                    'avg entropy': avg_entropy
                    }, step)
                print('lr:%f, avg reward:%f, avg advantage:%f, avg entropy:%f'%(learning_rate, avg_reward, avg_advantage, avg_entropy))

def agent(agent_id, config, game,network, tm_subset, model_weights_queue, experience_queue):
    random_state = np.random.RandomState(seed=agent_id)
    model = Model(config, game.state_dims, game.action_dim, game.max_moves, master=False)
    solver = Solver()
    # initial synchronization of the model weights from the coordinator 
    model_weights = model_weights_queue.get()
    model.model.set_weights(model_weights)

    idx = 0
    s_batch = []
    a_batch = []
    r_batch = []
    if config.method == 'pure_policy':
        ad_batch = []
    run_iteration_idx = 0
    num_tms = len(tm_subset)
    random_state.shuffle(tm_subset)
    run_iterations = FLAGS.num_iter
    
    while True:
        tm_idx = tm_subset[idx]
        #state
        state = game.get_state(tm_idx)
        s_batch.append(state)
        #action
        if config.method == 'actor_critic':    
            policy = model.actor_predict(np.expand_dims(state, 0)).numpy()[0]
        elif config.method == 'pure_policy':
            policy = model.policy_predict(np.expand_dims(state, 0)).numpy()[0]
        assert np.count_nonzero(policy) >= game.max_moves, (policy, state)
        actions = random_state.choice(game.action_dim, game.max_moves, p=policy, replace=False)
        for a in actions:
            a_batch.append(a)

        #reward
        reward = game.reward(tm_idx,network,actions,solver)
        r_batch.append(reward)
       
        if config.method == 'pure_policy':
            #advantage
            if config.baseline == 'avg':
                ad_batch.append(game.advantage(tm_idx, reward))
                game.update_baseline(tm_idx, reward)
            elif config.baseline == 'best':
                best_actions = policy.argsort()[-game.max_moves:]
                best_reward = game.reward(tm_idx, best_actions)
                ad_batch.append(reward - best_reward)

        run_iteration_idx += 1
        if run_iteration_idx >= run_iterations:
            # Report experience to the coordinator                          
            if config.method == 'actor_critic':    
                experience_queue.put([s_batch, a_batch, r_batch])
            elif config.method == 'pure_policy':
                experience_queue.put([s_batch, a_batch, r_batch, ad_batch])
            
            #print('report', agent_id)

            # synchronize the network parameters from the coordinator
            model_weights = model_weights_queue.get()
            model.model.set_weights(model_weights)
            
            del s_batch[:]
            del a_batch[:]
            del r_batch[:]
            if config.method == 'pure_policy':
                del ad_batch[:]
            run_iteration_idx = 0
      
        # Update idx
        idx += 1
        if idx == num_tms:
           random_state.shuffle(tm_subset)
           idx = 0

def main(_):
    #cpu only
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.get_logger().setLevel('INFO')
    #tf.debugging.set_log_device_placement(True)

    config = get_config(FLAGS) or FLAGS
    
    
    for num_paths in range(1,int(config.num_of_paths)+1):
        for network_topology in config.topology_file:
            for edge_capacity_bound in config.edge_capacity_bounds:
                network = Network(config,edge_capacity_bound,False)
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
                        game = CFRRL_Game(config,network)
    
                        
                        model_weights_queues = []
                        experience_queues = []
                        if FLAGS.num_agents == 0 or FLAGS.num_agents >= mp.cpu_count():
                            FLAGS.num_agents = mp.cpu_count() - 1
                        print('Agent num: %d, iter num: %d\n'%(FLAGS.num_agents+1, FLAGS.num_iter))
                        for _ in range(FLAGS.num_agents):
                            model_weights_queues.append(mp.Queue(1))
                            experience_queues.append(mp.Queue(1))

                        tm_subsets = np.array_split(game.wk_indexes, FLAGS.num_agents)

                        coordinator = mp.Process(target=central_agent, args=(config, game, model_weights_queues, experience_queues))

                        coordinator.start()

                        agents = []
                        for i in range(FLAGS.num_agents):
                            agents.append(mp.Process(target=agent, args=(i, config, game, network,tm_subsets[i], model_weights_queues[i], experience_queues[i])))

                        for i in range(FLAGS.num_agents):
                            agents[i].start()

                        coordinator.join()



# In[ ]:





# In[ ]:


if __name__ == '__main__':
    app.run(main)


# In[ ]:





# In[1]:


# import numpy as np
# tm_indexes = np.arange(0, 100)
# print(tm_indexes)


# In[ ]:




