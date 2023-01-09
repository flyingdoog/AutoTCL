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

```
ETTh1
{'aug_depth': 1, 'augdropout': 0.4, 'augmask_mode': 'mask_last', 'batch_size': 128, 
'bias_init': 0.2450533837982518, 'depth': 10, 'dropout': 0.1, 'hidden_dims': 128, 
'local_weight': 0.003183648974376273, 'lr': 2.1231938441299486e-05, 'mask_mode': 'all_true', 
'max_train_length': 256, 'meta_lr': 0.005769119414927489, 'reg_weight': 0.19205758994985575, 'repr_dims': 320}
0.3213645274941347
{24: {'norm': {'MSE': 0.03973055803758698, 'MAE': 0.15079194278342928}}, 
48: {'norm': {'MSE': 0.05624999275722058, 'MAE': 0.1801996229891593}}, 
168: {'norm': {'MSE': 0.10013126218373086, 'MAE': 0.23857584428311646}}, 
336: {'norm': {'MSE': 0.11905807636909252, 'MAE': 0.2633873644813072}}, 
720: {'norm': {'MSE': 0.14911703042812166, 'MAE': 0.3095809431579089}}}
```

```
ETTh2
{'aug_depth': 1, 'augdropout': 0.4, 'augmask_mode': 'mask_last', 'batch_size': 64, 
'bias_init': 0.012643424821070729, 'depth': 7, 'dropout': 0.1, 'hidden_dims': 32, 
'local_weight': 0.00018167214056458262, 'lr': 1.2654425138639566e-06, 'mask_mode': 'all_true', 
'max_train_length': 512, 'meta_lr': 0.3778642466122285, 'reg_weight': 0.0001563464527148474, 'repr_dims': 128}
0.4664574773557473
{24: {'norm': {'MSE': 0.0901836538055397, 'MAE': 0.23012123806653165}}, 
48: {'norm': {'MSE': 0.11674894998176559, 'MAE': 0.2648018954547142}}, 
168: {'norm': {'MSE': 0.17708138762792985, 'MAE': 0.33344198265699165}}, 
336: {'norm': {'MSE': 0.19757995767442252, 'MAE': 0.35660163019392066}}, 
720: {'norm': {'MSE': 0.20161758399967705, 'MAE': 0.36410910731724333}}}
```

```
ETTm1
{'aug_depth': 5, 'augdropout': 0.3, 'augmask_mode': 'mask_last', 'batch_size': 256,
'bias_init': 0.7480803532376448, 'depth': 6, 'dropout': 0.1, 'hidden_dims': 128, 
 local_weight': 0.0003450211056905891, 'lr': 6.303278638046721e-06, 'mask_mode': 'continuous',
'max_train_length': 512, 'meta_lr': 0.4219098645784781, 'reg_weight': 0.1299673441067767, 'repr_dims': 320}

0.22025471175975384
{24: {'norm': {'MSE': 0.013984960206357828, 'MAE': 0.08639836763011087}}, 
48: {'norm': {'MSE': 0.025775542312827353, 'MAE': 0.11775946084866623}}, 
96: {'norm': {'MSE': 0.03933831794411334, 'MAE': 0.14874286769443007}}, 
288: {'norm': {'MSE': 0.07850074116284952, 'MAE': 0.2122371106592067}}, 
672: {'norm': {'MSE': 0.11704371556968554, 'MAE': 0.2614924747705217}}}
```


```
electricity
{'aug_depth': 1, 'augdropout': 0.6672541892601949, 'augmask_mode': 'continuous', 'batch_size': 64,
 'bias_init': 0.3343334441473783, 'depth': 6, 'dropout': 0.042522386356410515, 'hidden_dims': 32,
  'local_weight': 0.07188680792170277, 'lr': 0.002700792812714019, 'mask_mode': 'all_true', 
  'max_train_length': 1024, 'meta_lr': 0.27086561180729235, 'reg_weight': 0.5719949796941333, 'repr_dims': 320}

0.7214978091114395

{24: {'norm': {'MSE': 0.24256298129181852, 'MAE': 0.2705244401915481}}, 
48: {'norm': {'MSE': 0.28854915681020016, 'MAE': 0.29837168350144705}}, 
168: {'norm': {'MSE': 0.4034157572966998, 'MAE': 0.3716181435023264}}, 
336: {'norm': {'MSE': 0.5625022308900808, 'MAE': 0.44844684296163684}}}
```