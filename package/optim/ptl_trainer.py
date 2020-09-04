from __future__ import print_function
from collections import OrderedDict
import torch
from torch.optim.lr_scheduler import MultiStepLR
from ..utils.cfg import transfer_items
from .reid_trainer import ReIDTrainer
from ..model import create_model
from ..utils.torch_utils import may_data_parallel
from ..utils.torch_utils import recursive_to_device
from .optimizer import create_optimizer
from .lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR
from ..data.multitask_dataloader import MTDataLoader
from ..loss.triplet_loss import TripletLoss
from ..loss.id_loss import IDLoss
from ..loss.ps_loss import PSLoss
from ..loss.attr_loss import AttrLoss
from ..data.create_dataset import __factory as dataset_factory


class PTLTrainer(ReIDTrainer):

    def create_train_loader(self, samples=None):
        cfg = self.cfg
        self.train_loader = self.create_dataloader(mode='train', samples=samples)
        if cfg.cd_ps_loss.use or cfg.cd_attr_loss.use:
            self.cd_train_loader = self.create_dataloader(mode='cd_train', samples=samples)
            self.train_loader = MTDataLoader([self.train_loader, self.cd_train_loader], ref_loader_idx=0)

    def create_model(self):
        if hasattr(self.cfg.model, self.cfg.model.name):
            transfer_items(getattr(self.cfg.model, self.cfg.model.name), self.cfg.model)
        if hasattr(self, 'train_loader') and self.cfg.id_loss.use:
            reid_loader = self.train_loader.loaders[0] if self.cfg.cd_ps_loss.use or self.cfg.cd_attr_loss.use else self.train_loader
            self.cfg.model.num_classes = [l.dataset.num_ids for l in reid_loader.loaders] if hasattr(reid_loader, 'loaders') else reid_loader.dataset.num_ids
            # print('self.cfg.model.num_classes', self.cfg.model.num_classes)
        if self.cfg.model.use_attr:
            self.cfg.model.attr_num_classes = dataset_factory[self.cfg.model.attr_format].attr_num_classes
        self.model = create_model(self.cfg.model)
        self.model = may_data_parallel(self.model)
        self.model.to(self.device)

    def set_model_to_train_mode(self):
        self.model.set_train_mode()

    def create_optimizer(self):
        cfg = self.cfg.optim
        ft_params, new_params = self.model.get_ft_and_new_params()
        param_groups = [{'params': ft_params, 'lr': cfg.ft_lr}]
        # Some model may not have new params
        if len(new_params) > 0:
            param_groups += [{'params': new_params, 'lr': cfg.new_params_lr}]
        self.optimizer = create_optimizer(param_groups, cfg)
        recursive_to_device(self.optimizer.state_dict(), self.device)

    def create_lr_scheduler(self):
        cfg = self.cfg.optim
        if cfg.lr_policy == 'step':
            cfg.lr_decay_steps = [len(self.train_loader) * ep for ep in cfg.lr_decay_epochs]
            self.lr_scheduler = MultiStepLR(self.optimizer, cfg.lr_decay_steps)
        elif cfg.lr_policy == 'warmupstep':
            cfg.lr_decay_steps = [len(self.train_loader) * ep for ep in cfg.lr_decay_epochs]
            self.lr_scheduler = WarmupMultiStepLR(self.optimizer, cfg.lr_decay_steps, warmup_epochs=int(cfg.epochs * 0.1 * len(self.train_loader)), warmup_begin_lrs=[0 for _ in self.optimizer.param_groups])
        elif cfg.lr_policy == 'warmupcosine':
            self.lr_scheduler = WarmupCosineAnnealingLR(self.optimizer, warmup_epochs=int(cfg.epochs * 0.1 * len(self.train_loader)), warmup_begin_lrs=[0 for _ in self.optimizer.param_groups], T_max=cfg.epochs * len(self.train_loader))
        else:
            raise ValueError('Invalid lr_policy {}'.format(cfg.lr_policy))

    def create_loss_funcs(self):
        cfg = self.cfg
        self.loss_funcs = OrderedDict()
        if cfg.id_loss.use:
            self.loss_funcs[cfg.id_loss.name] = IDLoss(cfg.id_loss, self.tb_writer)
        if cfg.tri_loss.use:
            self.loss_funcs[cfg.tri_loss.name] = TripletLoss(cfg.tri_loss, self.tb_writer)
        if cfg.src_ps_loss.use:
            self.loss_funcs[cfg.src_ps_loss.name] = PSLoss(cfg.src_ps_loss, self.tb_writer)
        if cfg.cd_ps_loss.use:
            self.loss_funcs[cfg.cd_ps_loss.name] = PSLoss(cfg.cd_ps_loss, self.tb_writer)
        if cfg.src_attr_loss.use:
            self.loss_funcs[cfg.src_attr_loss.name] = AttrLoss(cfg.src_attr_loss, self.tb_writer)
        if cfg.cd_attr_loss.use:
            self.loss_funcs[cfg.cd_attr_loss.name] = AttrLoss(cfg.cd_attr_loss, self.tb_writer)

    # NOTE: To save GPU memory, our multi-domain training requires
    # [1st batch: source-domain forward and backward]-
    # [2nd batch: cross-domain forward and backward]-
    # [update model]
    # So the following three-step framework is not strictly followed.
    #     pred = self.train_forward(batch)
    #     loss = self.criterion(batch, pred)
    #     loss.backward()
    def train_forward(self, batch):
        cfg = self.cfg
        batch = recursive_to_device(batch, self.device)
        if cfg.cd_ps_loss.use or cfg.cd_attr_loss.use:
            src_batch, cd_batch = batch
        else:
            src_batch = batch
        # Source Loss
        forward_type = []
        if cfg.id_loss.use or cfg.tri_loss.use:
            forward_type.append('reid')
        if cfg.src_ps_loss.use:
            forward_type.append('ps')
        if cfg.src_attr_loss.use:
            forward_type.append('attr')
        forward_type = '-'.join(forward_type)
        pred = self.model.forward(src_batch, forward_type=forward_type)
        loss = 0
        for loss_cfg in [cfg.id_loss, cfg.tri_loss, cfg.src_ps_loss, cfg.src_attr_loss]:
            if loss_cfg.use:
                loss += self.loss_funcs[loss_cfg.name](src_batch, pred, step=self.trainer.current_step)['loss']
        if isinstance(loss, torch.Tensor):
            loss.backward()
        # Cross-Domain Loss
        if cfg.cd_ps_loss.use or cfg.cd_attr_loss.use:
            forward_type = []
            if cfg.cd_ps_loss.use:
                forward_type.append('ps')
            if cfg.cd_attr_loss.use:
                forward_type.append('attr')
            forward_type = '-'.join(forward_type)
            pred = self.model.forward(cd_batch, forward_type=forward_type)
            loss = 0
            for loss_cfg in [cfg.cd_ps_loss, cfg.cd_attr_loss]:
                if loss_cfg.use:
                    loss += self.loss_funcs[loss_cfg.name](cd_batch, pred, step=self.trainer.current_step)['loss']
            if isinstance(loss, torch.Tensor):
                loss.backward()

    def criterion(self, batch, pred):
        return 0


if __name__ == '__main__':
    from ..utils import init_path
    trainer = PTLTrainer()
    if trainer.cfg.only_test:
        if 'reid' in trainer.cfg.test_tasks:
            score_str = trainer.test()
        if 'attr' in trainer.cfg.test_tasks:
            trainer.test_attr()
    else:
        trainer.train()
