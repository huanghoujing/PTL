export exp_root=${exp_root:=exp}
export gpu=${gpu:=0}
export only_test=${only_test:=False}
export dont_load_model_weight=${dont_load_model_weight:=False}
export dataset=${dataset:=market1501}

ow_str="
cfg.eval.dont_load_model_weight = ${dont_load_model_weight};
cfg.dataset.train.name = '${dataset}';
cfg.dataset.test.names = ['market1501', 'cuhk03_np_detected_jpg', 'duke'];
cfg.optim.optimizer = 'adam';
cfg.optim.ft_lr = 0.00035;
cfg.optim.new_params_lr = 0.00035;
cfg.dataloader.train.batch_type = 'pk';
cfg.dataloader.train.batch_size = 64;
cfg.dataloader.pk.k = 4;
cfg.optim.lr_decay_epochs = (160, 280);
cfg.optim.epochs = 480;
cfg.optim.epochs_per_val = 80;
cfg.optim.epochs_per_save_ckpt = 20;
cfg.id_loss.weight = 1;
cfg.tri_loss.weight = 1;
cfg.optim.lr_policy = 'warmupstep';
"
exp_dir=${exp_root}/train_baseline/${dataset}

if [ -n "${only_test}" ] && [ "${only_test}" == True ]; then
    ow_str="${ow_str}; cfg.only_test = True"
#else
#    rm -rf ${exp_dir}  # Remove results of last run
fi

CUDA_VISIBLE_DEVICES=${gpu} \
python -m package.optim.ptl_trainer \
--cfg_file package/config.py \
--ow_str "${ow_str}" \
--exp_dir ${exp_dir}