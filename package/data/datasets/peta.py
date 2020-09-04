from __future__ import print_function
import os
import os.path as osp
import numpy as np
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from ...utils.file import walkdir
from ...utils.file import read_lines
from ..transform import transform


colors = 'Black, Blue, Brown, Green, Grey, Orange, Pink, Purple, Red, White, Yellow'.split(', ')
attr_names = ['upperBody' + c for c in colors] \
             + ['lowerBody' + c for c in colors] \
             + ['hair' + c for c in colors] \
             + ['footwear' + c for c in colors] \
             + 'lowerBodyCasual lowerBodyFormal'.split(' ') \
             + 'lowerBodyCapri lowerBodyHotPants lowerBodyJeans lowerBodyLongSkirt lowerBodyShorts lowerBodyShortSkirt lowerBodySuits lowerBodyTrousers'.split(' ') \
             + 'lowerBodyPlaid lowerBodyThinStripes'.split(' ') \
             + 'personalLess15 personalLess30 personalLess45 personalLess60 personalLarger60'.split(' ') \
             + 'personalFemale personalMale'.split(' ') \
             + 'upperBodyCasual upperBodyFormal'.split(' ') \
             + 'upperBodyPlaid upperBodyLogo upperBodyThinStripes'.split(' ') \
             + 'upperBodyLongSleeve upperBodyNoSleeve upperBodyShortSleeve'.split(' ') \
             + 'upperBodyTshirt upperBodyVNeck upperBodyJacket upperBodySuit upperBodySweater upperBodyThickStripes upperBodyOther'.split(' ') \
             + 'hairBald hairLong hairShort'.split(' ') \
             + 'footwearBoots footwearLeatherShoes footwearSandals footwearShoes footwearSneakers footwearStocking'.split(' ') \
             + 'carryingBabyBuggy carryingBackpack carryingShoppingTro carryingUmbrella carryingFolder carryingLuggageCase carryingMessengerBag carryingPlasticBags carryingSuitcase carryingOther carryingNothing'.split(' ') \
             + 'accessoryHeadphone accessoryHairBand accessoryHat accessoryKerchief accessoryMuffler accessorySunglasses accessoryNothing'.split(' ')
attr_name_to_ind = dict(zip(attr_names, range(len(attr_names))))
print('PETA len(attr_names):', len(attr_names))


class PETA(TorchDataset):
    has_ps_label = False
    has_attr_label = True
    attr_num_classes = [2 for _ in range(105)]
    im_root = 'PETA dataset'  # Note that `unzip PETA.zip` gives 'PETA dataset' and 'ReadMe.txt', without the top-level dir 'PETA'.

    def __init__(self, cfg, samples=None):
        self.cfg = cfg
        self.root = osp.join(cfg.root, cfg.name)
        self._pre_process_attr()
        if cfg.split == 'train':
            self.samples = self.train_im_paths
        elif cfg.split == 'val':
            self.samples = self.val_im_paths
        elif cfg.split == 'test':
            self.samples = self.test_im_paths
        elif cfg.split == 'all_attr_ims':
            self.samples = self.train_im_paths + self.val_im_paths + self.test_im_paths
        else:
            raise ValueError('Unsupported split {}'.format(cfg.split))
        print('PETA | {} | {} images'.format(cfg.split, len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        cfg = self.cfg
        im_path = self.samples[index]
        sample = {}
        sample['im_path'] = im_path
        sample['im'] = self.get_im(im_path)
        sample['attr_label'] = self.get_attr_label(im_path)
        transform(sample, cfg)
        return sample

    def get_im(self, im_path):
        return Image.open(osp.join(self.root, im_path)).convert("RGB")

    @staticmethod
    def _parse_ann_line(line):
        not_used_attrs = ['accessoryFaceMask', 'lowerBodyLogo', 'accessoryShawl', 'lowerBodyThickStripes']
        items = line.strip().split(' ')
        inds = [attr_name_to_ind[a] for a in items[1:] if a not in not_used_attrs]
        label = np.zeros([len(attr_names)], dtype=int)
        label[inds] = 1
        return items[0], label

    @staticmethod
    def parse_im_path(im_path):
        parts = im_path.split('/')
        id = parts[-3] + '_' + parts[-1].split('_')[0]
        return id

    def _pre_process_sub_dir(self, sub_dir):
        """sub_dir is one of ['3DPeS', 'CAVIAR4REID', 'CUHK', 'GRID', 'i-LID', 'MIT', 'PRID', 'SARC3D', 'TownCentre', 'VIPeR']"""
        im_dir = osp.join(self.root, self.im_root, sub_dir, 'archive')
        im_paths = list(walkdir(im_dir, ['.bmp', '.jpg', '.png', '.jpeg']))
        im_paths = [p[len(self.root) + 1:] for p in im_paths]
        lines = read_lines(osp.join(im_dir, 'Label.txt'))
        ids, labels = zip(*[self._parse_ann_line(l) for l in lines])
        ids = [sub_dir + '_' + id for id in ids]
        id_to_attr_label = dict(zip(ids, labels))
        return im_paths, id_to_attr_label

    def _pre_process_attr(self):
        cfg = self.cfg
        # sub_dirs = os.listdir(osp.join(self.root, self.im_root))
        sub_dirs = cfg.peta_sub_dirs
        im_paths = []
        id_to_attr_label = {}
        for sub_dir in sub_dirs:
            im_paths_, id_to_attr_label_ = self._pre_process_sub_dir(sub_dir)
            im_paths.extend(im_paths_)
            id_to_attr_label.update(id_to_attr_label_)
            print('sub_dir {}, #ids {}, #images {}'.format(sub_dir, len(id_to_attr_label_), len(im_paths_)))
        train_prop, val_prop, test_prop = 0.6, 0.3, 0.1
        train_num, val_num = int(len(im_paths) * train_prop), int(len(im_paths) * val_prop)
        np.random.RandomState(seed=1).shuffle(im_paths)
        self.train_im_paths, self.val_im_paths, self.test_im_paths = im_paths[:train_num], im_paths[train_num:train_num+val_num], im_paths[train_num+val_num:]
        self.id_to_attr_label = id_to_attr_label
        print('total #ids {}, #images {}, #train images {}, #val images {}, #test images {}'.format(len(id_to_attr_label), len(im_paths), len(self.train_im_paths), len(self.val_im_paths), len(self.test_im_paths)))

    def get_attr_label(self, im_path):
        attr_label = self.id_to_attr_label[self.parse_im_path(im_path)]
        return attr_label
