export exp_root=${exp_root:=exp}
export gpu=${gpu:=0}
export dont_load_model_weight=${dont_load_model_weight:=False}
export train_on_dataset=${train_on_dataset:=peta}  # peta, rap
export predict_on_dataset=${predict_on_dataset:=market1501}  # market1501, cuhk03_np_detected_jpg, duke
export soft_or_hard=${soft_or_hard:=soft}

ow_str="
cfg.eval.dont_load_model_weight = ${dont_load_model_weight};
cfg.model.use_attr = True;
cfg.model.attr_format = '${train_on_dataset}';
cfg.simple_init = True;
"
exp_dir=${exp_root}/train_attr/${train_on_dataset}

CUDA_VISIBLE_DEVICES=${gpu} \
python script/predict_attr.py \
--cfg_file package/config.py \
--ow_str "${ow_str}" \
--exp_dir ${exp_dir} \
--train_on_dataset ${train_on_dataset} \
--predict_on_dataset ${predict_on_dataset} \
--soft_or_hard ${soft_or_hard}