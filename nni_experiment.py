import nni
from nni.experiment import Experiment
search_space = {
    "batch_size": {"_type":"choice", "_value": [128]}, # 32,64,256
    # "bias_init":{"_type":"loguniform","_value":[0.5,1.5]},
    "local_weight":{"_type":"loguniform","_value":[0.0001,0.1]},
    "lr":{"_type":"loguniform","_value":[0.00001,0.0015]},
    "meta_lr":{"_type":"loguniform","_value":[0.01,0.15]},
    'repr_dims':{"_type":"choice", "_value": [320,]}, # 32,64, 256,320
    'max_train_length':{"_type":"choice", "_value": [2048]}, # 256,512,1024,1280,1536,
    'depth':{"_type":"choice", "_value": [10]},        # 3,5,7,9,
    'dropout':{"_type":"choice", "_value": [0.1]},    # 0.1,0.6, ,0.2,0.3,0.4,0.5,0.7
    # "dropout":{"_type":"uniform","_value":[0.05,0.45]}, # a*10  ,
    'hidden_dims':{"_type":"choice", "_value": [64,]}, # 8,16,32,128
    # 'ratio_step':{"_type":"choice", "_value": [1]},
    'mask_mode':{"_type":"choice", "_value": ['mask_last']}, # ''binomial','all_true','continuous',
}
experiment = Experiment('local')
experiment.config.trial_command = 'python3 train_forecasting.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
experiment.config.max_trial_number = 1000
experiment.config.trial_concurrency = 1
experiment.run(8080)
input()