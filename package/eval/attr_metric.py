from __future__ import print_function
import numpy as np


def attribute_evaluate_lidw(gt_result, pt_result):
    """
    Input:
    gt_result, pt_result, N*L, with 0/1
    Output:
    result
    a dictionary, including label-based and instance-based evaluation
    label-based: label_pos_acc, label_neg_acc, label_acc
    instance-based: instance_acc, instance_precision, instance_recall, instance_F1
    """
    # obtain the label-based and instance-based accuracy
    # compute the label-based accuracy
    if gt_result.shape != pt_result.shape:
        print('Shape beteen groundtruth and predicted results are different')
    # compute the label-based accuracy
    result = {}
    gt_pos = np.sum((gt_result == 1).astype(float), axis=0)
    gt_neg = np.sum((gt_result == 0).astype(float), axis=0)
    pt_pos = np.sum((gt_result == 1).astype(float) * (pt_result == 1).astype(float), axis=0)
    pt_neg = np.sum((gt_result == 0).astype(float) * (pt_result == 0).astype(float), axis=0)
    label_pos_acc = 1.0*pt_pos/(gt_pos + 1e-8)
    label_neg_acc = 1.0*pt_neg/(gt_neg + 1e-8)
    label_pos_acc[gt_pos == 0] = 1
    label_neg_acc[gt_neg == 0] = 1
    label_acc = (label_pos_acc + label_neg_acc)/2
    result['label_pos_acc'] = label_pos_acc
    result['label_neg_acc'] = label_neg_acc
    result['label_acc'] = label_acc
    # compute the instance-based accuracy
    # precision
    gt_pos = np.sum((gt_result == 1).astype(float), axis=1)
    pt_pos = np.sum((pt_result == 1).astype(float), axis=1)
    floatersect_pos = np.sum((gt_result == 1).astype(float)*(pt_result == 1).astype(float), axis=1)
    union_pos = np.sum(((gt_result == 1)+(pt_result == 1)).astype(float),axis=1)
    # avoid empty label in predicted results
    cnt_eff = float(gt_result.shape[0])
    for iter, key in enumerate(gt_pos):
        if key == 0:
            union_pos[iter] = 1
            pt_pos[iter] = 1
            gt_pos[iter] = 1
            cnt_eff = cnt_eff - 1
            continue
        if pt_pos[iter] == 0:
            pt_pos[iter] = 1
    instance_acc = np.sum(floatersect_pos/union_pos)/cnt_eff
    instance_precision = np.sum(floatersect_pos/pt_pos)/cnt_eff
    instance_recall = np.sum(floatersect_pos/gt_pos)/cnt_eff
    floatance_F1 = 2*instance_precision*instance_recall/(instance_precision+instance_recall)
    result['instance_acc'] = instance_acc
    result['instance_precision'] = instance_precision
    result['instance_recall'] = instance_recall
    result['instance_F1'] = floatance_F1
    return result


def eval_accuracy(gt, pred):
    """
    Multi-class classification accuracy.
    :param gt: numpy int array, shape [num_samples, num_classification_tasks] or [num_samples]
    :param pred: numpy int array, shape [num_samples, num_classification_tasks]  or [num_samples]
    :return: numpy float array, shape [num_classification_tasks] or scalar
    """
    return (gt == pred).mean(axis=0)