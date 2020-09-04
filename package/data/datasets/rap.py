import os.path as osp
import numpy as np
from scipy.io import loadmat
from ..dataset import Dataset


class RAP(Dataset):
    has_ps_label = False
    has_attr_label = True
    attr_num_classes = [2 for _ in range(96)]
    im_root = 'RAP_dataset'

    def _pre_process_all_attr_ims(self):
        ann_file = osp.join(self.root, 'RAP_annotation/RAP_annotation.mat')
        assert osp.exists(ann_file), "Annotation does not exist: {}".format(ann_file)
        ann = loadmat(open(ann_file, 'r'))
        ann = ann['RAP_annotation'][0][0]
        num_ims = 84928
        num_attrs = 152
        num_train_ids = 1295
        num_test_ids = 1294
        assert len(ann[0]) == num_ims
        assert ann[1].shape == (num_ims, num_attrs)
        assert len(ann[2]) == num_attrs
        assert len(ann[5]) == num_ims
        im_names = np.array([ann[0][i][0][0] for i in range(num_ims)])  # np.ndarray, str, shape (num_ims,)
        attr_values = ann[1].astype(int)  # np.ndarray, int32, shape (num_ims, 152) -> int
        attr_names = [ann[2][i][0][0] for i in range(num_attrs)]  # num_attrs strings
        ids = np.array([ann[5][i][0] for i in range(num_ims)], dtype=int)  # np.ndarray, int16, shape (num_ims,) -> int
        train_ids = ann[6][0, 0][0][0].astype(int)  # np.ndarray, uint16, shape (num_train_ids,) -> int
        test_ids = ann[6][0, 0][1][0].astype(int)  # np.ndarray, uint16, shape (num_test_ids,) -> int
        assert train_ids.shape == (num_train_ids,)
        assert test_ids.shape == (num_test_ids,)
        assert set(train_ids).isdisjoint(set(test_ids))
        # 26638 images with labeled id, 14947 with id -1 (distractors), 43343 with id -2 (unknown)
        num_ims_with_labeled_id, num_ims_with_neg1_id, num_ims_with_neg2_id = 26638, 14947, 43343
        assert (ids > 0).sum() == num_ims_with_labeled_id
        assert (ids == -1).sum() == num_ims_with_neg1_id
        assert (ids == -2).sum() == num_ims_with_neg2_id

        im_name_to_id = dict(zip(im_names, ids))

        # index of 'viewpoint' is 111, index of 'occlustion-TypeOther' is 119
        # 'viewpoint' is 4-class, with value 1,2,3,4
        # Others in the first 120 attributes should be binary.
        # The trailing 32 attributes are bbox coordinates.
        binary_inds = range(attr_names.index('occlustion-TypeOther') + 1)
        binary_inds.remove(attr_names.index('viewpoint'))
        # In the binary attributes, there is some noise.
        #     np.unique(attr_values[:, inds]) is [0, 1, 2]
        #     (attr_values[:, inds] == 0).sum() is 9017630
        #     (attr_values[:, inds] == 1).sum() is 1088641
        #     (attr_values[:, inds] == 2).sum() is 161
        attr_values[:, binary_inds] = np.clip(attr_values[:, binary_inds], 0, 1)
        assert set(attr_values[:, attr_names.index('viewpoint')]) == {1, 2, 3, 4}
        attr_values[:, attr_names.index('viewpoint')] -= 1

        used_attr_names = attr_names[:attr_names.index('attachment-Baby')]
        used_attr_inds = [attr_names.index(n) for n in used_attr_names]
        used_attr_values = attr_values[:, used_attr_inds]
        im_name_to_attr = dict(zip(im_names, used_attr_values))

        return {
            'im_names': im_names,
            'im_name_to_attr': im_name_to_attr
        }

    def get_attr_label(self, im_path):
        # a numpy array with shape [num_attr]
        attr_label = self.ann['im_name_to_attr'][osp.basename(im_path)]
        return attr_label

    def load_split(self):
        cfg = self.cfg
        ann = self._pre_process_all_attr_ims()
        if cfg.split != 'all_attr_ims':
            ntrain, nval = int(len(ann['im_names']) * 0.6), int(len(ann['im_names']) * 0.3)
            if cfg.split == 'attr_ims_trainval':
                ann['im_names'] = ann['im_names'][:ntrain+nval]
            elif cfg.split == 'attr_ims_train':
                ann['im_names'] = ann['im_names'][:ntrain]
            elif cfg.split == 'attr_ims_val':
                ann['im_names'] = ann['im_names'][ntrain:ntrain+nval]
            elif cfg.split == 'attr_ims_test':
                ann['im_names'] = ann['im_names'][ntrain + nval:]
            else:
                raise ValueError
        self.ann = ann
        samples = [{'im_path': osp.join(self.im_root, n)} for n in ann['im_names']]
        return samples
