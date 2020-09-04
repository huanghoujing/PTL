from __future__ import print_function
import sys
sys.path.insert(0, '.')

import os.path as osp
from package.utils.file import get_files_by_pattern
from package.utils.image import read_im, mask_to_im_custom_colormap, save_im


def vis_one_im(ps_path, save_path):
    ps = read_im(ps_path, convert_rgb=False, resize_h_w=None, transpose=False)
    ps = mask_to_im_custom_colormap(ps, 8, transpose=False)
    save_im(ps, save_path, transpose=False)


# root = 'dataset/market1501/Market-1501-v15.09.15_ps_label/bounding_box_train'
# new_root = 'dataset/market1501/Market-1501-v15.09.15_ps_label_vis/bounding_box_train'

# root = 'dataset/cuhk03_np_detected_jpg/cuhk03-np-jpg_ps_label/detected/bounding_box_train'
# new_root = 'dataset/cuhk03_np_detected_jpg/cuhk03-np-jpg_ps_label_vis/detected/bounding_box_train'

root = 'dataset/duke/DukeMTMC-reID_ps_label/bounding_box_train'
new_root = 'dataset/duke/DukeMTMC-reID_ps_label_vis/bounding_box_train'

pattern = '*.png'
paths = sorted(get_files_by_pattern(root, pattern=pattern, strip_root=False))
for i, p in enumerate(paths):
    new_p = osp.join(new_root, osp.basename(p))
    vis_one_im(p, new_p)
    if (i+1) % 50 == 0:
        print('{}/{} done'.format(i+1, len(paths)))
