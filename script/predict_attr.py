from __future__ import print_function
import sys
sys.path.insert(0, '.')
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from package.optim.ptl_trainer import PTLTrainer
from package.utils.torch_utils import recursive_to_device
from package.utils.file import save_pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_on_dataset', type=str, default='None', help='Which dataset the attributes are trained on.')
    parser.add_argument('--predict_on_dataset', type=str, default='None', help='Which dataset to predict attributes.')
    parser.add_argument('--soft_or_hard', type=str, default='soft', choices=['soft', 'hard'], help='Predict soft or hard label.')
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    trainer = PTLTrainer()
    trainer.create_model()
    trainer.load_items(model=True)
    trainer.model.eval()
    loader = trainer.create_dataloader('test', args.predict_on_dataset, 'train')
    im_paths = []
    prob = []
    for i, batch in enumerate(loader):
        print('i={}'.format(i))
        recursive_to_device(batch, trainer.device)
        with torch.no_grad():
            logits_list = trainer.model.forward(batch, forward_type='attr')['attr_logits_list']
        if args.soft_or_hard == 'soft':
            prob_list = [F.softmax(logits, 1).detach().cpu().numpy() for logits in logits_list]
        elif args.soft_or_hard == 'hard':
            prob_list = torch.cat([torch.argmax(logits, 1, keepdim=True) for logits in logits_list], 1).detach().cpu().numpy()
        else:
            raise ValueError
        prob.append(prob_list)
        im_paths.extend(batch['im_path'])
    if args.soft_or_hard == 'soft':
        prob = [np.concatenate(p, axis=0) for p in zip(*prob)]
        for p in prob:
            print('p.shape:', p.shape)
        im_path_to_prob = [dict(zip(im_paths, p)) for p in prob]
    elif args.soft_or_hard == 'hard':
        prob = np.concatenate(prob, axis=0)
        print("prob.shape:", prob.shape)
        im_path_to_prob = dict(zip(im_paths, prob))
    else:
        raise ValueError

    save_pickle(im_path_to_prob, 'dataset/predicted_attr/{}_to_{}/{}_attr.pkl'.format(args.train_on_dataset, args.predict_on_dataset, args.soft_or_hard))


if __name__ == '__main__':
    main()
