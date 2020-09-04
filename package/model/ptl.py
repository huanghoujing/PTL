from __future__ import print_function
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from .backbone import create_backbone
from .global_pool import GlobalPool
from .ps_head import PartSegHead
from ..utils.model import create_embedding
from ..utils.model import init_classifier
from ..eval.torch_distance import normalize


class PTL(BaseModel):
    def __init__(self, cfg):
        super(PTL, self).__init__()
        self.cfg = cfg
        self.backbone = create_backbone(cfg.backbone)
        self.pool = GlobalPool(cfg)
        self.bn = nn.BatchNorm1d(self.backbone.out_c)
        self.bn.bias.requires_grad_(False)
        if hasattr(cfg, 'num_classes') and cfg.num_classes > 0:
            self.cls = nn.Linear(self.backbone.out_c, self.cfg.num_classes, bias=False)
            self.cls.apply(init_classifier)
        if cfg.use_ps:
            cfg.ps_head.in_c = self.backbone.out_c
            self.ps_head = PartSegHead(cfg.ps_head)
        if cfg.use_attr:
            self.create_attr_em_list()
            self.create_attr_cls_list()
        print('Model Structure:\n{}'.format(self))

    def create_attr_em_list(self):
        cfg = self.cfg
        self.attr_em_list = nn.ModuleList([create_embedding(self.backbone.out_c, cfg.attr_em_dim) for _ in cfg.attr_num_classes])

    def create_attr_cls_list(self):
        cfg = self.cfg
        self.attr_cls_list = nn.ModuleList([nn.Linear(cfg.attr_em_dim, nclass) for nclass in cfg.attr_num_classes])
        self.attr_cls_list.apply(init_classifier)

    def get_ft_and_new_params(self):
        ft_modules, new_modules = self.get_ft_and_new_modules()
        ft_params = list(chain.from_iterable([list(m.parameters()) for m in ft_modules]))
        new_params = list(chain.from_iterable([list(m.parameters()) for m in new_modules]))
        return ft_params, new_params

    def get_ft_and_new_modules(self):
        ft_modules = [self.backbone]
        new_modules = [self.bn]
        if hasattr(self, 'cls'):
            new_modules += [self.cls]
        if hasattr(self, 'ps_head'):
            new_modules += [self.ps_head]
        if hasattr(self, 'attr_em_list'):
            new_modules += [self.attr_em_list, self.attr_cls_list]
        return ft_modules, new_modules

    def set_train_mode(self):
        self.train()

    def backbone_forward(self, in_dict):
        return self.backbone(in_dict['im'])

    def reid_forward(self, in_dict):
        # print('=====> in_dict.keys() entering reid_forward():', in_dict.keys())
        pool_out_dict = self.pool(in_dict)
        out_dict = {}
        feat_before_bn = pool_out_dict['feat_list'][0]
        feat_after_bn = self.bn(feat_before_bn)
        out_dict['tri_loss_em'] = feat_before_bn
        out_dict['feat_list'] = [feat_after_bn]
        if hasattr(self, 'cls'):
            out_dict['logits_list'] = [self.cls(feat_after_bn)]
        return out_dict

    def ps_forward(self, in_dict):
        return self.ps_head(in_dict['feat'])

    def attr_forward(self, in_dict):
        N = in_dict['feat'].size(0)
        feat = F.adaptive_avg_pool2d(in_dict['feat'], 1).contiguous().view(N, -1)
        feat_list = [em(feat) for em in self.attr_em_list]
        logits_list = [cls(f) for cls, f in zip(self.attr_cls_list, feat_list)]
        return feat_list, logits_list

    def forward(self, in_dict, forward_type='reid'):
        in_dict['feat'] = self.backbone_forward(in_dict)
        out_dict = {}
        forward_type = forward_type.split('-')
        if 'reid' in forward_type:
            out_dict.update(self.reid_forward(in_dict))
        if 'ps' in forward_type:
            out_dict['ps_pred'] = self.ps_forward(in_dict)
        if 'attr' in forward_type:
            attr_feat_list, attr_logits_list = self.attr_forward(in_dict)
            out_dict.update({'attr_feat_list': attr_feat_list, 'attr_logits_list': attr_logits_list})
        return out_dict

    @torch.no_grad()
    def extract_feat(self, in_dict):
        self.eval()
        out_dict = self.forward(in_dict, forward_type='reid')
        out_dict['feat_list'] = [normalize(f) for f in out_dict['feat_list']]
        feat = torch.cat(out_dict['feat_list'], 1)
        feat = feat.cpu().numpy()
        ret_dict = {
            'im_path': in_dict['im_path'],
            'feat': feat,
        }
        if 'label' in in_dict:
            ret_dict['label'] = in_dict['label'].cpu().numpy() if isinstance(in_dict['label'], torch.Tensor) else in_dict['label']
        if 'cam' in in_dict:
            ret_dict['cam'] = in_dict['cam'].cpu().numpy() if isinstance(in_dict['cam'], torch.Tensor) else in_dict['cam']
        return ret_dict

    @torch.no_grad()
    def predict_attr(self, in_dict):
        self.eval()
        out_dict = self.forward(in_dict, forward_type='attr')
        pred = torch.cat([torch.argmax(l, dim=1, keepdim=True) for l in out_dict['attr_logits_list']], 1)
        pred = pred.cpu().numpy()
        ret_dict = {
            'im_path': in_dict['im_path'],
            'attr_pred': pred,
        }
        if 'attr_label' in in_dict:
            ret_dict['attr_label'] = in_dict['attr_label'].cpu().numpy()
        return ret_dict