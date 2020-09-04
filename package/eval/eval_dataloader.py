from __future__ import print_function
import numpy as np
from .extract_feat import extract_dataloader_feat
from .eval_feat import eval_feat
from ..utils.file import save_pickle


def _print_stat(dic):
    print('=> Eval Statistics:')
    print('\tdic.keys():', dic.keys())
    print("\tdic['q_feat'].shape:", dic['q_feat'].shape)
    print("\tdic['q_label'].shape:", dic['q_label'].shape)
    print("\tdic['q_cam'].shape:", dic['q_cam'].shape)
    print("\tdic['g_feat'].shape:", dic['g_feat'].shape)
    print("\tdic['g_label'].shape:", dic['g_label'].shape)
    print("\tdic['g_cam'].shape:", dic['g_cam'].shape)


def eval_dataloader(model, q_loader, g_loader, cfg):
    q_feat_dict = extract_dataloader_feat(model, q_loader, cfg)
    g_feat_dict = extract_dataloader_feat(model, g_loader, cfg)
    save_pickle([q_feat_dict, g_feat_dict], cfg.test_feat_cache_file)
    dic = {
        'q_feat': q_feat_dict['feat'],
        'q_label': np.array(q_feat_dict['label']),
        'q_cam': np.array(q_feat_dict['cam']),
        'g_feat': g_feat_dict['feat'],
        'g_label': np.array(g_feat_dict['label']),
        'g_cam': np.array(g_feat_dict['cam']),
    }
    _print_stat(dic)
    return eval_feat(dic, cfg)
