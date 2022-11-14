# This is the configuration file for the experiments. 
class NetworkConfig(object):
    
  """this class would be used in the reinforcement learning implementation"""  
  version = 'RL_v31'
  project_name = 'inter_domain_egr_maximizationv31'
  testing_results = "results/juniper_path_selection_evaluationv31.csv"
  tf_ckpts = "tf_ckpts31"
  method = 'actor_critic'
#   method = 'pure_policy'
  
  model_type = 'Conv'
  scale = 100

  max_step = 1000 * scale
  
  initial_learning_rate = 0.001
  learning_rate_decay_rate = 0.94
  learning_rate_decay_step = 2 * scale
  moving_average_decay = 0.9999
  entropy_weight = 0.1

 
  save_step = 40 
  max_to_keep = 1000

  Conv2D_out = 128
  Dense_out = 128
  
  optimizer = 'RMSprop'
#   optimizer = 'Adam'
    
  logit_clipping = 10       #10 or 0, = 0 means logit clipping is disabled



  test_traffic_file = 'WK2'
  max_moves = 30            #percentage
  # For pure policy
  baseline = 'avg'          #avg, best
#   baseline = 'best'          #avg, best
class Config(NetworkConfig):
  """ this class includes all the experiments setup"""
  topology_file = 'ATT'#'SURFnet'
  work_load_file = 'WK'
  min_edge_capacity = 1# EPR per second
  max_edge_capacity = 400# EPR per second
  min_edge_fidelity = 0.94
  max_edge_fidelity = 0.98
  fidelity_threshold_ranges = [0.9]# for organizations
  edge_fidelity_ranges = [0.9]
  edge_capacity_bounds = [400]
  schemes = ["EGR","Hop","EGRSquare","Genetic"]# Set the schemes that you want to evaluate. If you want to evaluate only genetic algorithm, set only Genetic keyword in the list
  schemes = ["EGR"]
  cut_off_for_path_searching = 3# We use this cut off when we search for the set of paths that a user pair can use
  num_of_organizations = 1
  number_of_user_pairs = 6 # Number of user pairs that exist for each organization. 
  min_num_of_paths = 1
  num_of_paths = 1
  q_values = [1]# Different values of q (swap success probability) that we want to evaluate in the experiment
 
    

def get_config(FLAGS):
  config = Config

  for k, v in FLAGS.__flags.items():
    if hasattr(config, k):
      setattr(config, k, v.value)

  return config
