import nni
from nni.experiment import Experiment
search_space = {
    "batch_size": {"_type":"choice", "_value": [32,64,256,128]}, #
    "lr":{"_type":"loguniform","_value":[0.000001,0.01]}, # 0.000003
    "meta_lr":{"_type":"loguniform","_value":[0.00001,0.5]},  # 0.03

    'mask_mode':{"_type":"choice", "_value": ['binomial','all_true','continuous','mask_last']}, # ''binomial','all_true','continuous',
    'augmask_mode': {"_type": "choice", "_value": ['binomial', 'all_true', 'continuous', 'mask_last']},
    "bias_init": {"_type": "loguniform", "_value": [0.01, 1.]},
    'local_weight':{"_type":"loguniform", "_value": [0.0001,1.0]},  # 0.72
    'reg_weight':{"_type":"loguniform", "_value": [0.0001,1.0]},

    'dropout': {"_type": "choice", "_value": [0.1]},  # 0.1,0.6, ,0.2,0.3,0.4,0.5,0.7
    'augdropout': {"_type": "choice", "_value": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]},  # 0.1,0.6, ,0.2,0.3,0.4,0.5,

    'repr_dims':{"_type":"choice", "_value": [320,256,128,64,512]}, # 32,64, 256,320
    'hidden_dims': {"_type": "choice", "_value": [256, 128, 64, 32, 16, 8]},  # 32,64, 256,320
    'max_train_length': {"_type": "choice", "_value": [2048, 256, 512, 1024]},  # 256,512,1024,1280,1536,

    'depth': {"_type": "choice", "_value": [10]},  # 3,5,7,9,
    'aug_depth':{"_type":"choice", "_value": [1]},  # 21


    # 'reg_weight':{"_type":"loguniform", "_value": [0.0001,1.0]},
    # 'local_weight':{"_type":"loguniform", "_value": [0.0001,1.0]},  # 0.72
    # "dropout":{"_type":"uniform","_value":[0.05,0.45]},

}
experiment = Experiment('local')
experiment.config.trial_command = 'python3 train_forecasting.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'SMAC'
experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
experiment.config.max_trial_number = 500
experiment.config.trial_concurrency = 2
experiment.run(8080)
input()