from .market1501 import Market1501


class DukeMTMCreID(Market1501):
    has_ps_label = True
    has_attr_label = True
    im_root = 'DukeMTMC-reID'
    split_spec = {
        'train': {'pattern': '{}/bounding_box_train/*.jpg'.format(im_root), 'map_label': True},
        'query': {'pattern': '{}/query/*.jpg'.format(im_root), 'map_label': False},
        'gallery': {'pattern': '{}/bounding_box_test/*.jpg'.format(im_root), 'map_label': False},
    }
