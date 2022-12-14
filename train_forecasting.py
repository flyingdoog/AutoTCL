import numpy as np
import argparse
import time
import datetime
import datautils
from utils import init_dl_program,dict2class
from infots import InfoTS as MetaInfoTS
# from baseline import InfoTS as baseInfoTS
import nni
from nni.utils import merge_parameter
nni_params = nni.get_next_parameter()
from models.augclass import *
all_augs = [jitter(), scaling(), time_warp(), window_slice(), window_warp(),cutout(),subsequence()]

paras = {
    'dataset':'ETTh1',  #electricity ETTh1 lora
    'batch_size': 32,
    'lr': 0.0001,
    'meta_lr':0.012,
    'mask_mode':'mask_last',
    'augmask_mode':'mask_last',
    'bias_init':0.0,
    'local_weight':0.72,
    'reg_weight':0.002,

    'dropout': 0.1,
    'augdropout': 0.1,

    'repr_dims': 128,
    'hidden_dims': 64,
    'max_train_length': 256,
    'iters': 40000,
    'epochs': 400,
    'depth': 10,
    'aug_depth': 1,

    'archive':'forecast_csv_univar',
    'gpu':0,
    'seed':42,
    'max_threads':12,
    'log_file':'forecast_csv',
    'eval':True,
    'beta':0.5,
    'split_number':8,
    'label_ratio':1.0,
    'meta_beta':0.15,
    'aug':None,
    'aug_p1':0.7,
    'aug_p2':0.,
    'supervised_meta':False,
    'ratio_step':1,
    'gamma_zeta':0.005,
}
print(paras)
print(nni_params)
# nni_params={'aug_depth': 1, 'augdropout': 0.4, 'augmask_mode': 'mask_last', 'batch_size': 128,
# 'bias_init': 0.2450533837982518, 'depth': 10, 'dropout': 0.1, 'hidden_dims': 128,
# 'local_weight': 0.003183648974376273, 'lr': 2.1231938441299486e-05, 'mask_mode': 'all_true',
# 'max_train_length': 256, 'meta_lr': 0.005769119414927489, 'reg_weight': 0.19205758994985575, 'repr_dims': 320}
params = merge_parameter(paras, nni_params)
parser = argparse.ArgumentParser()
args = dict2class(**params)


device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

if args.dataset == "lora":
    task_type = 'forecasting'
    data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv_(True)
    train_data = data[:, train_slice]

elif args.archive == 'forecast_csv':
    task_type = 'forecasting'
    data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset)
    train_data = data[:, train_slice]

elif args.archive == 'forecast_csv_univar':
    task_type = 'forecasting'
    data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, univar=True)
    train_data = data[:, train_slice]

valid_dataset = (data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)

if train_data.shape[0] == 1:
    train_slice_number = int(train_data.shape[1] / args.max_train_length)
    if train_slice_number < args.batch_size:
        args.batch_size = train_slice_number
else:
    if train_data.shape[0] < args.batch_size:
        args.batch_size = train_data.shape[0]
print("Arguments:", str(args))

config = dict(
    batch_size=args.batch_size,
    lr=args.lr,
    meta_lr = args.meta_lr,
    output_dims=args.repr_dims,
    max_train_length=args.max_train_length,
    input_dims=train_data.shape[-1],
    device=device,
    depth =  args.depth,
    hidden_dims = args.hidden_dims,
    num_cls =  args.batch_size,
    dropout = args.dropout,
    augdropout = args.augdropout,
    mask_mode = args.mask_mode,
    augmask_mode = args.augmask_mode,
    bias_init = args.bias_init,
    gamma_zeta = args.gamma_zeta
)

t = time.time()

'''
model = baseInfoTS(
    aug = args.aug,
    aug_p1= args.aug_p1,
    **config
)
'''

model = MetaInfoTS(
    aug_p1= args.aug_p1,
    eval_every_epoch =20,
    **config
)


res = model.fit(train_data,
     task_type = task_type,
     meta_beta=args.meta_beta,
     n_epochs=args.epochs,
     n_iters=args.iters,
     beta = args.beta,
     verbose=False,
     miverbose=True,
     split_number=args.split_number,
     valid_dataset = valid_dataset,
     train_labels= None,
    ratio_step= args.ratio_step,
    lcoal_weight = args.local_weight,
    reg_weight = args.reg_weight
    )

mse, mae = res
mi_info = 'mse %.5f  mae%.5f' % (mse[-1], mae[-1])

print(mi_info)

t = time.time() - t
print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")
print("Finished.")

