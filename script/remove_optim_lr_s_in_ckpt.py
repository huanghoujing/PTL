"""Remove optimizer and lr scheduler in checkpoint to reduce file size."""
from __future__ import print_function


import sys
sys.path.insert(0, '.')
from package.utils.file import walkdir
from package.utils.torch_utils import only_keep_model


# ori_ckpt_paths = list(walkdir('exp', '.pth'))
ori_ckpt_paths = [
    'exp/train_ptl/peta/soft/src_attr_lw0-tgt_attr_lw1-src_ps_lw0-tgt_ps_lw0.1/cuhk03_np_detected_jpg-TO-market1501/ckpt.pth',
    'exp/train_ptl/peta/soft/src_attr_lw0-tgt_attr_lw1-src_ps_lw0-tgt_ps_lw0.1/cuhk03_np_detected_jpg-TO-duke/ckpt.pth',
]
print('ori_ckpt_paths:\n' + '\n'.join(ori_ckpt_paths))
new_ckpt_paths = ori_ckpt_paths

for ori_path, new_path in zip(ori_ckpt_paths, new_ckpt_paths):
    only_keep_model(ori_path, new_path)
