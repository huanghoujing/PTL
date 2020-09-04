from __future__ import print_function
import numpy as np
from tqdm import tqdm
from ..utils.misc import concat_dict_list
from ..utils.torch_utils import recursive_to_device
from ..utils.file import save_pickle, load_pickle
from ..eval.attr_metric import eval_accuracy, attribute_evaluate_lidw
from ..utils.log import array_str


def predict_dataloader(model, loader, cfg):
    model.eval()
    dict_list = []
    for batch in tqdm(loader, desc='Predict Attribute', miniters=20, ncols=120, unit=' batches'):
        batch = recursive_to_device(batch, cfg.device)
        pred_dict = model.predict_attr(batch)
        dict_list.append(pred_dict)
    ret_dict = concat_dict_list(dict_list)
    return ret_dict


def _print_stat(dic):
    print('=> Attribute Eval Statistics:')
    print('\tdic.keys():', dic.keys())
    print("\tdic['attr_pred'].shape:", dic['attr_pred'].shape)
    print("\tdic['attr_label'].shape:", dic['attr_label'].shape)


def eval_attr_dataloader(model, loader, cfg):
    pred_dict = predict_dataloader(model, loader, cfg)
    # save_pickle(pred_dict, cfg.attr_pred_cache_file)
    # pred_dict = load_pickle(cfg.attr_pred_cache_file)  # TODO
    print("cfg.attr_num_classes: ", cfg.attr_num_classes)
    _print_stat(pred_dict)
    attr_n_cls = cfg.attr_num_classes
    assert min(attr_n_cls) >= 2, "attr_num_classes: {}".format(cfg.attr_num_classes)
    assert pred_dict['attr_pred'].shape == pred_dict['attr_label'].shape
    assert len(cfg.attr_num_classes) == pred_dict['attr_pred'].shape[1], "attr_num_classes: {}, pred_dict['attr_pred'].shape[1]: {}".format(cfg.attr_num_classes, pred_dict['attr_pred'].shape[1])
    any_multi = any([nc > 2 for nc in cfg.attr_num_classes])
    if any_multi:
        multi_cls_pred = np.stack([pred_dict['attr_pred'][:, i] for i, n_cls in enumerate(cfg.attr_num_classes) if n_cls > 2], axis=1)
        multi_cls_label = np.stack([pred_dict['attr_label'][:, i] for i, n_cls in enumerate(cfg.attr_num_classes) if n_cls > 2], axis=1)
        acc = eval_accuracy(multi_cls_label, multi_cls_pred)
        print('Multi-class Acc | {:5.1%} | {}'.format(acc.mean(), array_str(acc, fmt='{:5.1%}')))
    any_binary = any([nc == 2 for nc in cfg.attr_num_classes])
    if any_binary:
        binary_cls_pred = np.stack([pred_dict['attr_pred'][:, i] for i, n_cls in enumerate(cfg.attr_num_classes) if n_cls == 2], axis=1) if any_multi else pred_dict['attr_pred']
        binary_cls_label = np.stack([pred_dict['attr_label'][:, i] for i, n_cls in enumerate(cfg.attr_num_classes) if n_cls == 2], axis=1) if any_multi else pred_dict['attr_label']
        score_dict = attribute_evaluate_lidw(binary_cls_label, binary_cls_pred)
        print('Label Acc    | {:5.1%}'.format(score_dict['label_acc'].mean()))
        print('Instance Acc | {:5.1%}'.format(score_dict['instance_acc']))
        print('Instance F1  | {:5.1%}'.format(score_dict['instance_F1']))
