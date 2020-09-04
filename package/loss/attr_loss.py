from __future__ import print_function
import torch
import torch.nn.functional as F
from .loss import Loss
from ..utils.meter import RecentAverageMeter as Meter


class AttrLoss(Loss):
    def __init__(self, cfg, tb_writer=None):
        super(AttrLoss, self).__init__(cfg, tb_writer=tb_writer)
        self.criterion = torch.nn.CrossEntropyLoss()

    def __call__(self, batch, pred, step=0, **kwargs):
        cfg = self.cfg

        # Calculation
        if cfg.soft_or_hard == 'soft':
            loss_list = [- (F.log_softmax(logits, dim=1) * label).sum(1).mean() for logits, label in zip(pred['attr_logits_list'], batch['attr_label'])]
        elif cfg.soft_or_hard == 'hard':
            loss_list = [self.criterion(logits, label).mean() for logits, label in zip(pred['attr_logits_list'], batch['attr_label'].t())]
        else:
            raise ValueError('Invalid AttrLoss Type: {}'.format(cfg.soft_or_hard))
        # New version of pytorch allow stacking 0-dim tensors, but not concatenating.
        loss = torch.stack(loss_list).mean()

        # Meter
        if cfg.name not in self.meter_dict:
            self.meter_dict[cfg.name] = Meter(name=cfg.name)
        self.meter_dict[cfg.name].update(loss.item())
        if len(loss_list) > 1 and cfg.print_each_loss:
            attr_fmt = 'a#{}'
            for i in range(len(loss_list)):
                if attr_fmt.format(i + 1) not in self.meter_dict:
                    self.meter_dict[attr_fmt.format(i + 1)] = Meter(name=attr_fmt.format(i + 1))
                self.meter_dict[attr_fmt.format(i + 1)].update(loss_list[i].item())

        # Tensorboard
        if self.tb_writer is not None:
            self.tb_writer.add_scalars(cfg.name, {cfg.name: self.meter_dict[cfg.name].avg}, step)
            if len(loss_list) > 1 and cfg.print_each_loss:
                self.tb_writer.add_scalars('Attr Losses', {attr_fmt.format(i + 1): self.meter_dict[attr_fmt.format(i + 1)].avg for i in range(len(loss_list))}, step)

        # Scale by loss weight
        loss *= cfg.weight

        return {'loss': loss}
