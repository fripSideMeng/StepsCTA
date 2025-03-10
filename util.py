import random
import os

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
import torch


def f1_score_multilabel(true_list, pred_list):
    conf_mat = multilabel_confusion_matrix(np.array(true_list),
                                           np.array(pred_list))
    agg_conf_mat = conf_mat.sum(axis=0)
    # Note: Pos F1
    # [[TN FP], [FN, TP]] if we consider 1 as the positive class
    p = agg_conf_mat[1, 1] / agg_conf_mat[1, :].sum()
    r = agg_conf_mat[1, 1] / agg_conf_mat[:, 1].sum()
    
    micro_f1 = 2 * p * r / (p  + r) if (p + r) > 0 else 0.
    class_p = conf_mat[:, 1, 1] /  conf_mat[:, :, 1].sum(axis=1)
    class_r = conf_mat[:, 1, 1] /  conf_mat[:, 1, :].sum(axis=1)
    class_f1 = np.divide(2 * (class_p * class_r), class_p + class_r,
                         out=np.zeros_like(class_p), where=(class_p + class_r) != 0)
    
    class_freq = conf_mat.sum(axis=(1, 2))
    valid_classes = class_freq > 0

    filtered_class_f1 = class_f1[valid_classes]
    filtered_class_f1 = np.nan_to_num(filtered_class_f1)

    macro_f1 = filtered_class_f1.mean() if len(filtered_class_f1) > 0 else 0.0

    if "LOCAL_RANK" not in os.environ or os.environ["LOCAL_RANK"] == '0':
        print(filtered_class_f1)

    return (micro_f1, macro_f1, class_f1, conf_mat)


def parse_tagname(tag_name):
    """sato_bert_bert-base-uncased-bs16-ml-256"""
    if "__" in tag_name:
        # Removetraining ratio
        tag_name = tag_name.split("__")[0]
    tokens = tag_name.split("_")[-1].split("-")
    shortcut_name = "-".join(tokens[:-3])
    max_length = int(tokens[-1])
    batch_size = int(tokens[-3].replace("bs", ""))
    return shortcut_name, batch_size, max_length


def accuracy_minority_labels(y_true: list, y_pred: list) -> float:
    actual_minority_cnt = torch.tensor(y_true).sum().item()
    pred_minority_match_cnt = 0
    for pred, label in zip(y_pred, y_true):
        if label == 1 and pred == 1:
            pred_minority_match_cnt += 1
    return pred_minority_match_cnt / actual_minority_cnt


def set_seed(seed: int):
    """https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py#L58-L63"""    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    """Add the following 2 lines
    https://discuss.pytorch.org/t/how-could-i-fix-the-random-seed-absolutely/45515
    """
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

    """
    For detailed discussion on the reproduciability on multiple GPU
    https://discuss.pytorch.org/t/reproducibility-over-multigpus-is-impossible-until-randomness-of-threads-is-controled-and-yet/47079
    """
