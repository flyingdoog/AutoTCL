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


def main(params):
    # params = dict2class(**args)

    device = init_dl_program(params.gpu, seed=params.seed, max_threads=params.max_threads)

    task_type = 'forecasting'
    data, train_slice, valid_slice, test_slice, \
    scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv_lora(params.dataset,
                                            params.time_cols,
                                            params.forecast_cols)
    train_data = data[:, train_slice]


    # valid_dataset = (data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)

    if train_data.shape[0] == 1:
        train_slice_number = int(train_data.shape[1] / params.max_train_length)
        if train_slice_number < params.batch_size:
            params.batch_size = train_slice_number
    else:
        if train_data.shape[0] < params.batch_size:
            params.batch_size = train_data.shape[0]

    config = dict(
        batch_size=params.batch_size,
        lr=params.lr,
        meta_lr=params.meta_lr,
        output_dims=params.repr_dims,
        max_train_length=params.max_train_length,
        input_dims=train_data.shape[-1],
        device=device,
        depth=params.depth,
        hidden_dims=params.hidden_dims,
        num_cls=params.batch_size,
        dropout=params.dropout,
        mask_mode=params.mask_mode,
        bias_init=params.bias_init
    )

    model = MetaInfoTS(
        eval_every_epoch=5,
        **config
    )

    res = model.fit(train_data,
                    task_type=task_type,
                    n_epochs=params.epochs,
                    n_iters=params.iters,
                    verbose=False,
                    miverbose=True,
                    valid_dataset=None,
                    train_labels=None,
                    lcoal_weight=params.local_weight,
                    reg_weight=params.reg_weight
                    )
    padding = 200
    embedding = model.casual_encode(train_data,sliding_length=1,sliding_padding=padding,batch_size=256)

    return embedding


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='20221112-lora-features', # required=True,
                        help='dataset name')
    parser.add_argument('--mask_mode', type=str, default='mask_last', # required=True,
                        help='mask mode')

    parser.add_argument('--time_cols', type=str, default='gw1-lp-PRR1,gw2-lp-PRR1,gw2-lp-PRR2,gw1-lp-BER1,gw1-lp-BER2,gw2-lp-BER1,gw1-lp-RSSI2',  # required=True,
                        help='time embedding cols')
    parser.add_argument('--forecast_cols', type=str,
                        default='WS-temper,WS-humidity,WS-wind-speed',
                        # required=True,
                        help='forecast embedding cols')

    parser.add_argument('--gpu', type=int, default=0,  # required=True,
                        help='device')
    parser.add_argument('--max_threads', type=int, default=12,  # required=True,
                        help='max_threads')
    parser.add_argument('--seed', type=int, default=42,  # required=True,
                        help='seed')
    parser.add_argument('--batch_size', type=int, default=32,  # required=True,
                        help='batch size')
    parser.add_argument('--repr_dims', type=int, default=320,  # required=True,
                        help='embedding dims')
    parser.add_argument('--hidden_dims', type=int, default=64,  # required=True,
                        help='hidden dims')
    parser.add_argument('--max_train_length', type=int, default=256,  # required=True,
                        help='train length')
    parser.add_argument('--iters', type=int, default=4000,  # required=True,
                        help='train iters')
    parser.add_argument('--epochs', type=int, default=400,  # required=True,
                        help='train epochs')
    parser.add_argument('--depth', type=int, default=10,  # required=True,
                        help='embedding layer depth')
    parser.add_argument('--aug_depth', type=int, default=10,  # required=True,
                        help='augment layer depth')


    parser.add_argument('--lr', type=float, default=0.001,  # required=True,
                        help='embedding net learning rate')
    parser.add_argument('--meta_lr', type=float, default=0.1,  # required=True,
                        help='seed')
    parser.add_argument('--dropout', type=float, default=0.1,  # required=True,
                        help='dropout rate')
    parser.add_argument('--bias_init', type=float, default=0.0,  # required=True,
                        help='bias init value')
    parser.add_argument('--reg_weight', type=float, default=0.001,  # required=True,
                        help='regulation weight')
    parser.add_argument('--local_weight', type=float, default=0.0,  # required=True,
                        help='local infoNCE weight')

    args = parser.parse_args()
    # forecast_cols = args.forecast_cols.split(',')
    # time_cols = args.time_cols.split(',')
    args.forecast_cols = args.forecast_cols.split(',')
    args.time_cols = args.time_cols.split(',')

    main(args)

