class NetworkConfig(object):
  scale = 10

  max_step = 1000 * scale
  
  initial_learning_rate = 0.0001
  learning_rate_decay_rate = 0.96
  learning_rate_decay_step = 5 * scale
  moving_average_decay = 0.9999
  entropy_weight = 0.1

 
  save_step = 40 
  max_to_keep = 1000

  Conv2D_out = 128
  Dense_out = 128
  
  optimizer = 'RMSprop'
  #optimizer = 'Adam'
    
  logit_clipping = 10       #10 or 0, = 0 means logit clipping is disabled

class Config(NetworkConfig):
  version = 'RL_v1'

  project_name = 'inter_domain_egr_maximization'

  method = 'actor_critic'
  #method = 'pure_policy'
  
  model_type = 'Conv'

  topology_file = 'ATT'
  traffic_file = 'WK1'
  test_traffic_file = 'WK2'
  time_intervals = 10

  max_moves = 30            #percentage
  

  # For pure policy
  baseline = 'avg'          #avg, best
  commitment_window_range = 8
  look_ahead_window_range = 10
  logging_training_epochs = True
  training_epochs_experiment = True
  printing_flag = False
  

  min_edge_capacity = 400
  max_edge_capacity = 1400
  min_edge_fidelity = 0.94
  max_edge_fidelity = 0.98
  fidelity_threshold = 0.7
  fidelity_threshold_ranges = [0.9]
  edge_fidelity_ranges = [0.9]
  edge_capacity_bounds = [400]
  link_cost_metrics = ["EGR","hop","EGRsquare"]
  num_of_organizations = 1
  number_of_user_pairs = 3
  num_of_paths = 1
  path_selection_scheme = "shortest"
  testing_results = "results/juniper_path_selection_evaluation.csv"

def get_config(FLAGS):
  config = Config

  for k, v in FLAGS.__flags.items():
    if hasattr(config, k):
      setattr(config, k, v.value)

  return config
