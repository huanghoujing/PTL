# baseline + tgt attr
# gpu=0 src_attr_lw=0 tgt_attr_lw=1 src_ps_lw=0 tgt_ps_lw=0 use_attr=True use_ps=False bash script/train_ptl.sh

# baseline + tgt ps
# gpu=0 src_attr_lw=0 tgt_attr_lw=0 src_ps_lw=0 tgt_ps_lw=0.1 use_attr=False use_ps=True bash script/train_ptl.sh

# baseline + src attr
# gpu=0 src_attr_lw=1 tgt_attr_lw=0 src_ps_lw=0 tgt_ps_lw=0 use_attr=True use_ps=False bash script/train_ptl.sh

# baseline + src attr + tgt attr
# gpu=0 src_attr_lw=1 tgt_attr_lw=1 src_ps_lw=0 tgt_ps_lw=0 use_attr=True use_ps=False bash script/train_ptl.sh

# baseline + src ps
# gpu=0 src_attr_lw=0 tgt_attr_lw=0 src_ps_lw=0.1 tgt_ps_lw=0 use_attr=False use_ps=True bash script/train_ptl.sh

# baseline + src ps + tgt ps
# gpu=0 src_attr_lw=0 tgt_attr_lw=0 src_ps_lw=0.1 tgt_ps_lw=0.1 use_attr=False use_ps=True bash script/train_ptl.sh

# baseline + tgt attr, hard pseudo attr label
# gpu=0 src_attr_lw=0 tgt_attr_lw=1 src_ps_lw=0 tgt_ps_lw=0 use_attr=True use_ps=False soft_or_hard=hard bash script/train_ptl.sh

# baseline + tgt attr, RAP
# gpu=0 src_attr_lw=0 tgt_attr_lw=1 src_ps_lw=0 tgt_ps_lw=0 use_attr=True use_ps=False attr_format=rap bash script/train_ptl.sh

export exp_root=${exp_root:=exp}
export gpu=${gpu:=0}
export only_test=${only_test:=False}
export dont_load_model_weight=${dont_load_model_weight:=False}
export src_dset=${src_dset:=market1501}
export tgt_dset=${tgt_dset:=duke}
export src_attr_lw=${src_attr_lw:=0}
export tgt_attr_lw=${tgt_attr_lw:=1}
export use_attr=${use_attr:=True}
export soft_or_hard=${soft_or_hard:=soft}
export attr_format=${attr_format:=peta}
export src_ps_lw=${src_ps_lw:=0}
export tgt_ps_lw=${tgt_ps_lw:=0.1}
export use_ps=${use_ps:=True}


ow_str="
cfg.eval.dont_load_model_weight = ${dont_load_model_weight};
cfg.dataset.train.name = '${src_dset}';
cfg.dataset.cd_train.name = '${tgt_dset}';
cfg.dataset.test.names = ['${tgt_dset}'];
cfg.optim.optimizer = 'adam';
cfg.optim.ft_lr = 0.00035;
cfg.optim.new_params_lr = 0.00035;
cfg.dataloader.train.batch_type = 'pk';
cfg.dataloader.train.batch_size = 64;
cfg.dataloader.pk.k = 4;
cfg.optim.lr_decay_epochs = (160, 280);
cfg.optim.epochs = 480;
cfg.optim.epochs_per_val = 80;
cfg.optim.epochs_per_save_ckpt = 40;
cfg.id_loss.weight = 1;
cfg.tri_loss.weight = 1;
cfg.optim.lr_policy = 'warmupstep';
cfg.src_attr_loss.weight = ${src_attr_lw};
cfg.cd_attr_loss.weight = ${tgt_attr_lw};
cfg.model.use_attr = ${use_attr};
cfg.model.attr_format = '${attr_format}';
cfg.dataset.use_attr_label = ${use_attr};
cfg.dataset.pseudo_attr_soft_or_hard = '${soft_or_hard}';
cfg.dataset.attr_format = '${attr_format}';
cfg.src_attr_loss.soft_or_hard = '${soft_or_hard}';
cfg.cd_attr_loss.soft_or_hard = '${soft_or_hard}';
cfg.model.use_ps = ${use_ps};
cfg.dataset.use_ps_label = ${use_ps};
cfg.src_ps_loss.weight = ${src_ps_lw};
cfg.cd_ps_loss.weight = ${tgt_ps_lw};
"

exp_dir=${exp_root}/train_ptl/${attr_format}/${soft_or_hard}/src_attr_lw${src_attr_lw}-tgt_attr_lw${tgt_attr_lw}-src_ps_lw${src_ps_lw}-tgt_ps_lw${tgt_ps_lw}/${src_dset}-TO-${tgt_dset}

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