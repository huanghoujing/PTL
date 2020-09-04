import os.path as osp
from scipy.io import loadmat
from ..dataset import Dataset


# 26 attributes
attr_names = [
    'Female',
    'AgeOver60',
    'Age18-60',
    'AgeLess18',
    'Front',
    'Side',
    'Back',
    'Hat',
    'Glasses',
    'HandBag',
    'ShoulderBag',
    'Backpack',
    'HoldObjectsInFront',
    'ShortSleeve',
    'LongSleeve',
    'UpperStride',
    'UpperLogo',
    'UpperPlaid',
    'UpperSplice',
    'LowerStripe',
    'LowerPattern',
    'LongCoat',
    'Trousers',
    'Shorts',
    'Skirt&Dress',
    'boots'
]

# 22 attributes
used_attr_names = [
    'Female',
    'AgeOver60',
    'Age18-60',
    'AgeLess18',
    'Hat',
    'Glasses',
    'HandBag',
    'ShoulderBag',
    'Backpack',
    'ShortSleeve',
    'LongSleeve',
    'UpperStride',
    'UpperLogo',
    'UpperPlaid',
    'UpperSplice',
    'LowerStripe',
    'LowerPattern',
    'LongCoat',
    'Trousers',
    'Shorts',
    'Skirt&Dress',
    'boots'
]

used_attr_indices = [attr_names.index(n) for n in used_attr_names]


class PA100K(Dataset):
    has_ps_label = False
    has_attr_label = True
    attr_num_classes = [2 for _ in used_attr_names]
    im_root = 'PA-100K/release_data/release_data'

    def get_attr_label(self, im_path):
        # a numpy array with shape [num_attr]
        attr_label = self.im_name_to_attr[osp.basename(im_path)]
        return attr_label

    def load_split(self):
        cfg = self.cfg
        mat_path = osp.join(self.root, 'PA-100K/annotation.mat')
        ann = loadmat(open(mat_path, 'r'))
        # attr_names = [ann['attributes'][i][0][0] for i in range(len(ann['attributes']))]  # 26 attr
        # np.unique(ann['train_label']), np.unique(ann['val_label']), np.unique(ann['test_label']): array([0, 1], dtype=uint8)
        # ann['train_label'].shape, ann['val_label'].shape, ann['test_label'].shape: (80000, 26), (10000, 26), (10000, 26)
        split = cfg.split
        im_names = [ann['{}_images_name'.format(split)][i][0][0] for i in range(len(ann['{}_label'.format(split)]))]
        attr = ann['{}_label'.format(split)][:, used_attr_indices].astype(int)
        self.im_name_to_attr = dict(zip(im_names, attr))
        samples = [{'im_path': osp.join(self.im_root, n)} for n in im_names]
        return samples
