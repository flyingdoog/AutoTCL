# AutoTCL
## run
```commandline
python train_forecasting.py
```

## nni
```commandline
python nni_experiment.py
```

## embedding
```commandline
python get_lora_embedding.py --dataset 20221112-lora-features \
                             --mask_mode mask_last \
                             --time_cols gw1-lp-PRR1,gw2-lp-PRR1,gw2-lp-PRR2,gw1-lp-BER1,gw1-lp-BER2,gw2-lp-BER1,gw1-lp-RSSI2 \
                             --forecast_cols WS-temper,WS-humidity,WS-wind-speed \
                             --lr 0.001  \
                             --meta_lr 0.1 \
                             --reg_weight 0.001 
```

## result
