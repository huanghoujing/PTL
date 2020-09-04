from __future__ import print_function


import sys
sys.path.insert(0, '.')
from package.utils.file import walkdir
from package.utils.file import may_make_dir
import torch
import os.path as osp

ori_ckpt_paths = list(walkdir('exp', '.pth'))
# ori_ckpt_paths = [
#     'exp/train_ptl/peta/soft/src_attr_lw0-tgt_attr_lw1-src_ps_lw0-tgt_ps_lw0.1/market1501-TO-duke/ckpt.pth',
#     'exp/train_ptl/peta/soft/src_attr_lw0-tgt_attr_lw1-src_ps_lw0-tgt_ps_lw0.1/duke-TO-market1501/ckpt.pth',
#     'exp/train_ptl/peta/soft/src_attr_lw0-tgt_attr_lw1-src_ps_lw0-tgt_ps_lw0.1/cuhk03_np_detected_jpg-TO-market1501/ckpt.pth',
#     'exp/train_ptl/peta/soft/src_attr_lw0-tgt_attr_lw1-src_ps_lw0-tgt_ps_lw0.1/cuhk03_np_detected_jpg-TO-duke/ckpt.pth',
# ]
print('ori_ckpt_paths:\n' + '\n'.join(ori_ckpt_paths))
new_ckpt_paths = ori_ckpt_paths


def remove_cls_attr_ps(ori_p, new_p):
    ckpt = torch.load(ori_p, map_location=(lambda storage, loc: storage))
    sd = ckpt['state_dicts']['model']
    new_sd = {k:v for k,v in sd.items() if (k.startswith('backbone.') or k.startswith('bn.'))}
    ckpt['state_dicts']['model'] = new_sd
    may_make_dir(osp.dirname(new_p))
    torch.save(ckpt, new_p)
    print('=> Removed optimizer and lr scheduler of ckpt {} and save it to {}'.format(ori_p, new_p))


for ori_path, new_path in zip(ori_ckpt_paths, new_ckpt_paths):
    remove_cls_attr_ps(ori_path, new_path)