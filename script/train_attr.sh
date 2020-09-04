export exp_root=${exp_root:=exp}
export gpu=${gpu:=0}
export only_test=${only_test:=False}
export dont_load_model_weight=${dont_load_model_weight:=False}
export dataset=${dataset:=peta}

if [ ${dataset} == peta ]; then
    lr_decay_epochs='(15, 30)'
    epochs=40
    train_split=train
    test_split=val
elif [ ${dataset} == rap ]; then
    lr_decay_epochs='(8, 16)'
    epochs=20
    train_split=attr_ims_trainval
    test_split=attr_ims_test
else
    echo "dataset should be peta or rap"
    exit
fi

ow_str="
cfg.eval.dont_load_model_weight = ${dont_load_model_weight};
cfg.dataset.train.name = '${dataset}';
cfg.optim.lr_decay_epochs = ${lr_decay_epochs};
cfg.optim.epochs = ${epochs};
cfg.optim.epochs_per_val = 10;
cfg.src_attr_loss.weight = 1;
cfg.model.use_attr = True;
cfg.model.attr_format = '${dataset}';
cfg.dataset.use_attr_label = True;
cfg.dataset.attr_test.name = '${dataset}';
cfg.dataset.train.split = '${train_split}';
cfg.dataset.attr_test.split = '${test_split}';
cfg.test_tasks = ['attr'];
cfg.optim.trial_run = False;
cfg.optim.epochs_per_val = 10;
"
exp_dir=${exp_root}/train_attr/${dataset}

#echo "--------------------------"
#echo "ow_str is:"
#echo $ow_str
#echo "--------------------------"
#exit

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