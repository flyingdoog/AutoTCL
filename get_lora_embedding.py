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


def main():
    pass


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='20221112-lora-features', # required=True,
                        help='dataset name')
    parser.add_argument('--seed', type=int, default=42,  # required=True,
                        help='seed')

    parser.add_argument('--lr', type=float, default=0.001,  # required=True,
                        help='seed')

