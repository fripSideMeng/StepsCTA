import argparse
import json
import os
import gc
from time import time
from collections import OrderedDict, defaultdict, Counter
import random
import numpy as np

import pandas as pd
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, MSELoss, KLDivLoss, CosineEmbeddingLoss
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer

from dataset import (
    collate_fn,
    SatoCVTablewiseDatasetFromDF,
    SatoCVTablewiseDatasetFromDFAug
)

from model import (BertForMultiOutputClassification,
                   DistilBertForMultiOutputClassification)
from sklearn.metrics import f1_score


def u_loss_train(unlabeled_batch, device: torch.device,
                 cls_token_id, device_type: str, args, teacher, dtype: torch.dtype,
                 reg_fn, x_u_idx: defaultdict, selected_label: torch.Tensor, 
                 class_acc: torch.Tensor, u_cols_total: int, aug_type: str) -> torch.Tensor:
    inputs: torch.Tensor = unlabeled_batch["data"].to(device).T
    attn_mask: torch.Tensor = (inputs != 0.).float()
    curr_x_u_idx: list = unlabeled_batch["idx"]

    inputs_aug: torch.Tensor = unlabeled_batch["data_aug"].to(device).T
    inputs_combined = torch.cat((inputs, inputs_aug), dim=1)
    attn_mask: torch.Tensor = (inputs_combined != 0.).float()
    num_cols_list: list = unlabeled_batch["num_col"]
    if aug_type == "shuffle":
        cols_per_table = num_cols_list
        col_mappings: list = unlabeled_batch["col_mapping"]
    elif aug_type == "dropcol":
        dropped_cols: list = unlabeled_batch["dropped_col"]
        undropped_cols = [[True for _ in range(num_cols)] 
                          for num_cols in num_cols_list]
        for i, dropped_col in enumerate(dropped_cols):
            undropped_cols[i][dropped_col] = False
        undropped_cols = [flag for flags in undropped_cols for flag in flags]                              

    cls_indexes = torch.nonzero(inputs == cls_token_id)
    flatten_cls_idx = list()
    for n in range(cls_indexes.shape[0]):
        i, j = cls_indexes[n]
        flatten_cls_idx.append((i * inputs.shape[1] + j).item())

    flatten_cls_idx_s = list()
    cls_indexes_s = torch.nonzero(inputs_aug == cls_token_id)
    for n in range(cls_indexes_s.shape[0]):
        i, j = cls_indexes_s[n]
        flatten_cls_idx_s.append((i * inputs_aug.shape[1] + j).item())

    with torch.autocast(device_type=device_type, dtype=dtype):
        if args.output_hidden_states:
            logits, _ = teacher(inputs_combined, attn_mask)
        else:
            logits, = teacher(inputs_combined, attn_mask)
        split = inputs.size(1)
        logits_w = logits[:, :split].detach()
        logits_s = logits[:, split:]
    if len(logits_w.shape) == 3:
        logits_w = torch.flatten(logits_w, start_dim=0, end_dim=1)
    filtered_logits_w = logits_w[flatten_cls_idx]
    if len(logits_s.shape) == 3:
        logits_s = torch.flatten(logits_s, start_dim=0, end_dim=1)
    filtered_logits_s = logits_s[flatten_cls_idx] if aug_type == "dropout"\
                        else logits_s[flatten_cls_idx_s]

    if aug_type == "shuffle":
        start = 0
        for i, cols in enumerate(cols_per_table):
            curr_table_logits_s = filtered_logits_s[start : start + cols]
            curr_table_logits_s = curr_table_logits_s[col_mappings[i]]
            filtered_logits_s[start : start + cols] = curr_table_logits_s
            start += cols
    elif aug_type == "dropcol":
        filtered_logits_w = filtered_logits_w[undropped_cols]

    pseudo_labels = torch.softmax(filtered_logits_w / args.temp, dim=1)
    pseudo_labels = pseudo_labels.detach()
    max_probs, pred_u = torch.max(pseudo_labels, dim=-1)
    pseudo_counter = Counter(selected_label.tolist())
    if -1 in pseudo_counter:
        pseudo_counter.pop(-1)
    l_cols = list(pseudo_counter.values())
    denominator = max(l_cols) if l_cols else u_cols_total
    for i in range(78):
        class_acc[i] = pseudo_counter[i] / denominator
    # implementation of (x / (2 - x))
    convex = 0.95 * (class_acc[pred_u] / (2. - class_acc[pred_u]))
    mask = max_probs.ge(convex).float()

    select = max_probs.ge(0.95).long()
    flat_x_u_idx = list()
    for u_table_idx in curr_x_u_idx:
        flat_x_u_idx.extend(x_u_idx[u_table_idx])
    if aug_type == "dropcol":
        flat_x_u_idx = [idx for idx, keep in zip(flat_x_u_idx, undropped_cols) if keep]
    flat_x_u_idx = torch.LongTensor(flat_x_u_idx).to(device)
    if flat_x_u_idx[select == 1].nelement() != 0:
        selected_label[flat_x_u_idx[select == 1]] = pred_u.long()[select == 1]

    with torch.autocast(device_type=device_type, dtype=dtype):
        u_loss = torch.mean(reg_fn(filtered_logits_s, 
                            pseudo_labels) * mask)
        u_loss = args.u_lambda * u_loss
    del convex
    del mask
    del max_probs
    del pred_u
    del pseudo_counter
    del cls_indexes
    del attn_mask
    del l_cols
    gc.collect()
    torch.cuda.empty_cache()
    return u_loss


def eval_model(eval_dataloader: DataLoader, device_type: str, 
               model, dtype: torch.dtype, device: torch.device,
               pred_list: list, true_list: list, tokenizer: BertTokenizer,
               loss_fn, args):
    loss = 0.
    with torch.no_grad():
        for batch in eval_dataloader:

            data: torch.Tensor = batch["data"].to(device).T
            labels: torch.Tensor = batch["label"].to(device)
            attn_mask: torch.Tensor = (data != 0).float()

            with torch.autocast(device_type=device_type, dtype=dtype):
                if args.output_hidden_states:
                    logits, _ = model(data, attn_mask)
                else:
                    logits, = model(data, attn_mask)

            is_expanded = False
            if len(logits.shape) == 2:
                is_expanded = True
                logits = logits.unsqueeze(0)
            cls_indexes = torch.nonzero(data == tokenizer.cls_token_id)

            filtered_logits = torch.zeros(cls_indexes.shape[0],
                                          logits.shape[2], device=device)
            for n in range(cls_indexes.shape[0]):
                i, j = cls_indexes[n]
                if is_expanded:
                    idx = i * data.shape[0] + j
                    logit_n = logits[0, idx, :]
                else:
                    logit_n = logits[i, j, :]
                filtered_logits[n] = logit_n
            pred_list.extend(filtered_logits.argmax(
                             1).cpu().detach().numpy().tolist())
            true_list.extend(labels.cpu().detach().numpy().tolist())

            with torch.autocast(device_type=device_type, dtype=dtype):
                loss += loss_fn(filtered_logits, labels).item()

    return loss / len(eval_dataloader)


def compare_save(model, optimizer: AdamW, metric: float,
                 best_metrics: list, path, args):
    if metric > best_metrics[0] and not args.test_only:
        best_metrics[0] = metric
        model.save_pretrained(path)
        torch.save(optimizer.state_dict(), f"{path}/optimizer.pt")


def macro_micro_f1s(true_list: list, pred_list: list):
    macro_f1 = f1_score(true_list, pred_list, average="macro")
    micro_f1 = f1_score(true_list, pred_list, average="micro")
    return macro_f1, micro_f1


def train(args):
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32 if torch.cuda.is_available() else torch.bfloat16

    device = torch.device("cuda") if torch.cuda.is_available()\
             else torch.device("cpu")

    # Set random seeds for reproducibility
    torch.manual_seed(args.random_seed)
    if device_type == 'cuda':
        torch.cuda.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    args.tasks = sorted(args.tasks)

    task_num_class_dict = {
        "sato": 78,
    }

    true_ratio_dict = {}
    num_classes_list = []
    for task in args.tasks:
        num_classes_list.append(task_num_class_dict[task])
        true_ratio_dict[task] = 1.0

    for true_ratio in args.true_ratios:
        task, ratio_str = true_ratio.split("=")
        ratio = float(ratio_str)
        assert task in true_ratio_dict, f"Invalid task name: {task}"
        assert 0 < ratio <= 1
        true_ratio_dict[task] = ratio

    print(f"args={json.dumps(vars(args))}")

    batch_size = args.batch_size
    num_train_epochs = args.epoch

    true_ratio = str(round(true_ratio_dict["sato"], 2))
    if true_ratio_dict["sato"] > 1.0:
        pseudo_ratio = f"un{true_ratio}"
    else:
        pseudo_ratio = f"unsampled_{str(round(1.0 - true_ratio_dict['sato'], 2))}"
    feedback_path = "" if not args.feedback else "/feedback"
    hidden_state_path = "/pseudo" if not args.output_hidden_states\
                        else "/pseudo_hidden"
    feedback_only_path = "/no_labeled" if args.feedback_only else "/labeled"
    student_tag_name = f"model/semi/teachers_1/sato{feedback_path}/temp{args.temp}"\
                       + f"/true_{true_ratio}/bs{batch_size}{hidden_state_path}"\
                       + f"{feedback_only_path}_teacher"
    teacher_tag = f"model/ensemble/1/true_sampled_{true_ratio}_cols/sato/"\
                  + f"temp{args.temp}/bs{batch_size}{hidden_state_path}"\
                  + f"{feedback_path}{feedback_only_path}"

    print(student_tag_name)
    print(teacher_tag)

    if not args.test_only:
        if not os.path.exists(student_tag_name):
            print(f"{student_tag_name} does not exist. Created")
            os.makedirs(student_tag_name)
        if not os.path.exists(teacher_tag):
            print(f"{teacher_tag} does not exist. Created")
            os.makedirs(teacher_tag)

    tokenizer = BertTokenizer.from_pretrained("bert")

    # Initialize models and move to device
    student_model = DistilBertForMultiOutputClassification.from_pretrained(
        "distil-bert",
        num_labels=78,
        output_attentions=False,
        output_hidden_states=args.output_hidden_states,
    ).to(device)

    teacher_model = BertForMultiOutputClassification.from_pretrained(
        "bert",
        num_labels=78,
        output_attentions=False,
        output_hidden_states=args.output_hidden_states,
        hidden_dropout_prob=args.teacher_dropout
    ).to(device)

    if args.use_augmentation:
        sd = torch.load("model/watchog/pytorch_model_sato_shuffle.bin",
                        map_location=torch.device("cpu"),
                        weights_only=True)
        teacher_model.bert.load_state_dict(sd)

    true_cls = SatoCVTablewiseDatasetFromDF
    if args.use_augmentation:
        pseudo_cls = SatoCVTablewiseDatasetFromDFAug
    valid_cls = SatoCVTablewiseDatasetFromDF

    true_df_path = f"data/sato_col_type_serialized_grouped_sampled_{true_ratio}.pkl"
    if args.use_augmentation:
        aug = args.aug_type
        pseudo_df_path = "data/sato_col_type_serialized_grouped_"\
                                 + f"{aug}_{pseudo_ratio}.pkl"
    valid_df_path = "data/sato_col_type_serialized_grouped_valid.pkl"

    true_df: pd.DataFrame = pd.read_pickle(true_df_path)
    if args.use_augmentation:
        # pseudo_df_dropcol: pd.DataFrame = pd.read_pickle(pseudo_df_path_dropcol)
        pseudo_df: pd.DataFrame = pd.read_pickle(pseudo_df_path)
    valid_df: pd.DataFrame = pd.read_pickle(valid_df_path)

    train_dataset_true_label = true_cls(table_df=true_df,
                                        device=device)
    if args.use_augmentation:
        # train_dataset_pseudo_label_dropcol = pseudo_cls(table_df=pseudo_df_dropcol,
        #                                                 device=device,
        #                                                 aug_type="dropcol")
        train_dataset_pseudo_label = pseudo_cls(table_df=pseudo_df,
                                                device=device,
                                                aug_type=aug)
    valid_dataset = valid_cls(table_df=valid_df,
                              device=device)

    train_sampler_true_label = RandomSampler(train_dataset_true_label)
    if args.use_augmentation:
        train_sampler_pseudo_label = RandomSampler(train_dataset_pseudo_label)
    
    train_dataloader_true_label = DataLoader(train_dataset_true_label,
                                             sampler=train_sampler_true_label,
                                             batch_size=batch_size,
                                             collate_fn=collate_fn,
                                             pin_memory=True)
    
    if args.use_augmentation:
        train_dataloader_pseudo_label = DataLoader(train_dataset_pseudo_label,
                                                   sampler=train_sampler_pseudo_label,
                                                   batch_size=batch_size * args.u_ratio,
                                                   collate_fn=collate_fn,
                                                   pin_memory=True)

    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=batch_size,
                                  collate_fn=collate_fn,
                                  pin_memory=True)

    no_decay = ["bias", "LayerNorm.weight"]
    
    optimizer_grouped_parameters_student = [
        {
            "params": [
                p for n, p in student_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0
        },
        {
            "params": [
                p for n, p in student_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0
        },
    ]
    optimizer_student_u = AdamW(optimizer_grouped_parameters_student, lr=args.lr)

    optimizer_grouped_parameters_teacher = [
        {
            "params": [
                p for n, p in teacher_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0
        },
        {
            "params": [
                p for n, p in teacher_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0
        },
    ]
    optimizer_teacher_u = AdamW(optimizer_grouped_parameters_teacher, lr=args.lr)

    # Loss functions
    t_loss_l_fn = CrossEntropyLoss().to(device)
    s_loss_l_fn = CrossEntropyLoss().to(device)
    if args.pl_loss == "ce":
        s_loss_pseudo_fn = CrossEntropyLoss(reduction="none").to(device)
    elif args.pl_loss == "mse":
        s_loss_pseudo_fn = MSELoss(reduction="none").to(device)
    else:
        s_loss_pseudo_fn = KLDivLoss(reduction="none").to(device)
    if args.reg_loss == "ce":
        t_reg_fn = CrossEntropyLoss(reduction="none").to(device)
    elif args.reg_loss == "mse":
        t_reg_fn = MSELoss(reduction="none").to(device)
    else:
        t_reg_fn = KLDivLoss(reduction="none").to(device)
    loss_fn_embedding = CosineEmbeddingLoss().to(device)
    valid_loss_fn = CrossEntropyLoss().to(device)

    best_vl_micro_f1s_s = [-1. for _ in range(len(args.tasks))]
    best_vl_macro_f1s_s = [-1. for _ in range(len(args.tasks))]
    best_vl_micro_f1s_t = [-1. for _ in range(len(args.tasks))]
    best_vl_macro_f1s_t = [-1. for _ in range(len(args.tasks))]
    loss_info_lists_s = [[] for _ in range(len(args.tasks))]
    loss_info_lists_t = [[] for _ in range(len(args.tasks))]

    if args.use_augmentation:
        u_cols_total = pseudo_df["num_col"].sum().item()
        selected_label = torch.ones((u_cols_total,), dtype=torch.long, 
                                    device=device) * -1
        class_acc = torch.zeros((78,), device=device)
        x_u_idx = defaultdict(list)
        col_idx = 0
        for u_table_idx in range(pseudo_df.shape[0]):
            num_cols = pseudo_df.iloc[u_table_idx]["num_col"]
            x_u_idx[u_table_idx].extend(list(range(col_idx, col_idx + num_cols)))
            col_idx += num_cols
        assert col_idx == u_cols_total
        if aug == "dropcol":
            u_cols_total -= pseudo_df.shape[0]
        print("Unlabeled in total: ", u_cols_total)

    for epoch in range(num_train_epochs):
        t1 = time()

        teacher, student = teacher_model, student_model

        tr_loss_t = 0.
        vl_loss_t = 0.
        vl_loss_s = 0.
        ts_loss_s = 0.

        tr_true_list_t = []
        tr_pred_list_t = []
        vl_true_list_s = []
        vl_pred_list_s = []
        vl_true_list_t = []
        vl_pred_list_t = []

        if args.use_augmentation:
            reg_iter = iter(train_dataloader_pseudo_label)

        for batch_idx, labeled_batch in enumerate(train_dataloader_true_label):
            if batch_idx % args.report_period == 0:
                print(f"Batch {batch_idx}: {round(time() - t1, 3)} seconds.")

            labeled_data: torch.Tensor = labeled_batch["data"].to(device).T
            attn_mask_l = (labeled_data != 0).float()
            labels: torch.Tensor = labeled_batch["label"].to(device)
            cls_indexes = torch.nonzero(labeled_data == tokenizer.cls_token_id)

            student.eval()
            teacher.train()

            with torch.autocast(device_type=device_type, dtype=dtype):
                if args.output_hidden_states:
                    teacher_logits, teacher_hidden_states = teacher(labeled_data,
                                                                    attn_mask_l)
                else:
                    teacher_logits, = teacher(labeled_data,
                                              attn_mask_l)

            is_expanded = False
            if len(teacher_logits.shape) == 2:
                is_expanded = True
                teacher_logits = teacher_logits.unsqueeze(0)
            
            teacher_filtered_logits = torch.zeros(cls_indexes.shape[0],
                                                  teacher_logits.shape[2]).to(device)

            for n in range(cls_indexes.shape[0]):
                i, j = cls_indexes[n]
                if is_expanded:
                    idx = i * labeled_data.shape[0] + j
                    logit_n = teacher_logits[0, idx, :]
                else:
                    logit_n = teacher_logits[i, j, :]
                teacher_filtered_logits[n] = logit_n

            with torch.autocast(device_type="cuda", dtype=torch.float32):
                if args.output_hidden_states:
                    student_logits, student_hidden_states = student(labeled_data, 
                                                                    attn_mask_l)
                else:
                    student_logits, = student(labeled_data, attn_mask_l)
            
            is_expanded = False
            if len(student_logits.shape) == 2:
                is_expanded = True
                student_logits = student_logits.unsqueeze(0)
            student_filtered_logits = torch.zeros(cls_indexes.shape[0], 
                                                    student_logits.shape[2])\
                                                    .to(device)
            
            for n in range(cls_indexes.shape[0]):
                i, j = cls_indexes[n]
                if is_expanded:
                    idx = i * labeled_data.shape[0] + j
                    logit_n = student_logits[0, idx, :]
                else:
                    logit_n = student_logits[i, j, :]
                student_filtered_logits[n] = logit_n

            with torch.autocast(device_type=device_type, dtype=dtype):
                soft_pseudo_labels = torch.softmax(teacher_filtered_logits / args.temp,
                                                   dim=1)
                max_probs, _ = torch.max(soft_pseudo_labels, dim=1)
                mask = max_probs.ge(args.teacher_confidence).float()
                if args.pl_loss == "kl" or args.pl_loss == "mse":
                    mask = mask.unsqueeze(1).expand(-1, soft_pseudo_labels.size(1))
                    if args.pl_loss == "kl":
                        loss_s = s_loss_pseudo_fn(torch.log_softmax(student_filtered_logits, 
                                                                    dim=1), 
                                                soft_pseudo_labels) * mask
                        loss_s = loss_s.sum() / soft_pseudo_labels.size(0) * (args.temp ** 2)
                    else:
                        loss_s = s_loss_pseudo_fn(student_filtered_logits, 
                                                  teacher_filtered_logits)
                        loss_s = torch.mean(loss_s * mask)
                else:
                    loss_s = s_loss_pseudo_fn(student_filtered_logits, soft_pseudo_labels)
                    loss_s = torch.mean(loss_s * mask)
                mask = mask.int()
                if args.pl_loss == "kl" or args.pl_loss == "mse":
                    mask = mask.sum(dim=1)
                mask = (mask > 0)
                if args.output_hidden_states > 0 and mask.sum().item() > 0:
                    flat_hidden_s = torch.flatten(student_hidden_states[-1], 
                                                  start_dim=0, end_dim=1)
                    flat_hidden_t = torch.flatten(teacher_hidden_states[-1], 
                                                  start_dim=0, end_dim=1)
                    selected_cls = cls_indexes[mask][:, 0] * labeled_data.size(1)\
                                   + cls_indexes[mask][:, 1]
                    cls_embeds_s: torch.Tensor = torch.cat([flat_hidden_s[selected_cls]], dim=0)
                    cls_embeds_t: torch.Tensor = torch.cat([flat_hidden_t[selected_cls]], dim=0)
                    # cls_embeds_t = cls_embeds_t.detach()
                    cos_embedding_loss = loss_fn_embedding(cls_embeds_s, cls_embeds_t,
                                                           torch.ones(cls_embeds_s.size(0),
                                                                      device=device))
                    if torch.isnan(cos_embedding_loss).sum().item() == 0:
                        loss_s = loss_s + cos_embedding_loss
                        if batch_idx % 50 == 0:
                            print(selected_cls.tolist())
                
                student_grads = torch.autograd.grad(loss_s, student.parameters(), 
                                                    create_graph=True, retain_graph=True)
                updated_student_params = OrderedDict()
                for name_param, grad in zip(student.named_parameters(), student_grads):
                    name, param = name_param
                    updated_student_params[name] = (param - args.lr * grad)

            teacher.train()
            student.eval()

            with torch.autocast(device_type=device_type, dtype=dtype):
                if args.output_hidden_states:
                    student_logits, _ = torch.func.functional_call(student, 
                                                                   updated_student_params, 
                                                                   (labeled_data, attn_mask_l))
                else:
                    student_logits, = torch.func.functional_call(student, 
                                                                 updated_student_params, 
                                                                 (labeled_data, attn_mask_l))

            is_expanded = False
            if len(student_logits.shape) == 2:
                is_expanded = True
                student_logits = student_logits.unsqueeze(0)
            student_filtered_logits = torch.zeros(cls_indexes.shape[0], 
                                                  student_logits.shape[2]).to(device)
            for n in range(cls_indexes.shape[0]):
                i, j = cls_indexes[n]
                if is_expanded:
                    idx = i * labeled_data.shape[0] + j
                    logit_n = student_logits[0, idx, :]
                else:
                    logit_n = student_logits[i, j, :]
                student_filtered_logits[n] = logit_n

            with torch.autocast(device_type=device_type, dtype=dtype):
                loss_s_new = s_loss_l_fn(student_filtered_logits, labels)
            
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                if args.output_hidden_states:
                    teacher_logits_l, _ = teacher(labeled_data, attn_mask_l)
                else:
                    teacher_logits_l, = teacher(labeled_data, attn_mask_l)
            is_expanded = False
            if len(teacher_logits_l.shape) == 2:
                is_expanded = True
                teacher_logits_l = teacher_logits_l.unsqueeze(0)
            teacher_filtered_logits_l = torch.zeros(cls_indexes.shape[0], 
                                                    teacher_logits_l.shape[2])\
                                                    .to(device)
            for n in range(cls_indexes.shape[0]):
                i, j = cls_indexes[n]
                if is_expanded:
                    idx = i * labeled_data.shape[0] + j
                    logit_n = teacher_logits_l[0, idx, :]
                else:
                    logit_n = teacher_logits_l[i, j, :]
                teacher_filtered_logits_l[n] = logit_n

            tr_pred_list_t += teacher_filtered_logits_l.argmax(
                              1).cpu().detach().numpy().tolist()
            tr_true_list_t += labels.cpu().detach().numpy().tolist()
            
            if not args.feedback_only:
                with torch.autocast(device_type=device_type, dtype=dtype):
                    t_loss_l = t_loss_l_fn(teacher_filtered_logits_l, labels)
            loss_t = loss_s_new
            if args.pl_loss == "kl":
                loss_t = loss_t  * (1.0 / (args.temp ** 2))
            tr_loss_t += loss_t.item()
            teacher_grads = torch.autograd.grad(loss_t,
                                                teacher.parameters())
            for tg, tp in zip(teacher_grads, teacher.parameters()):
                tp.grad = tg
            if not args.feedback_only:
                loss_t = t_loss_l
                tr_loss_t += loss_t.item()
                teacher_grads = torch.autograd.grad(loss_t,
                                                    teacher.parameters())
                for tg, tp in zip(teacher_grads, teacher.parameters()):
                    tp.grad = tp.grad + tg
            
            if args.use_augmentation:
                try:
                    reg_batch = next(reg_iter)
                    u_loss_reg = u_loss_train(reg_batch, device, 
                                                  tokenizer.cls_token_id, 
                                                  device_type, args, 
                                                  teacher, dtype, t_reg_fn, 
                                                  x_u_idx, selected_label, 
                                                  class_acc, u_cols_total,
                                                  aug)
                    loss_t = u_loss_reg
                    tr_loss_t += loss_t.item()
                    teacher_grads = torch.autograd.grad(loss_t,
                                                        teacher.parameters())
                    for tg, tp in zip(teacher_grads, teacher.parameters()):
                        tp.grad = tp.grad + tg
                except StopIteration:
                    pass

            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            optimizer_teacher_u.step()
            optimizer_teacher_u.zero_grad()
            for tp in teacher.parameters():
                tp.grad = None

            del loss_t
            del loss_s
            del loss_s_new
            del student_grads
            del teacher_grads
            del updated_student_params
            gc.collect()
            torch.cuda.empty_cache()

            teacher.eval()
            student.train()

            with torch.autocast(device_type=device_type, dtype=dtype):
                with torch.no_grad():
                    if args.output_hidden_states:
                        teacher_logits, teacher_hidden_states = teacher(labeled_data,
                                                                        attn_mask_l)
                    else:
                        teacher_logits, = teacher(labeled_data,
                                                  attn_mask_l)

            is_expanded = False
            if len(teacher_logits.shape) == 2:
                is_expanded = True
                teacher_logits = teacher_logits.unsqueeze(0)
            
            teacher_filtered_logits = torch.zeros(cls_indexes.shape[0],
                                                  teacher_logits.shape[2]).to(device)

            for n in range(cls_indexes.shape[0]):
                i, j = cls_indexes[n]
                if is_expanded:
                    idx = i * labeled_data.shape[0] + j
                    logit_n = teacher_logits[0, idx, :]
                else:
                    logit_n = teacher_logits[i, j, :]
                teacher_filtered_logits[n] = logit_n

            with torch.autocast(device_type="cuda", dtype=torch.float32):
                if args.output_hidden_states:
                    student_logits, student_hidden_states = student(labeled_data, attn_mask_l)
                else:
                    student_logits, = student(labeled_data, attn_mask_l)
            
            is_expanded = False
            if len(student_logits.shape) == 2:
                is_expanded = True
                student_logits = student_logits.unsqueeze(0)
            student_filtered_logits = torch.zeros(cls_indexes.shape[0], 
                                                  student_logits.shape[2])\
                                                  .to(device)
            
            for n in range(cls_indexes.shape[0]):
                i, j = cls_indexes[n]
                if is_expanded:
                    idx = i * labeled_data.shape[0] + j
                    logit_n = student_logits[0, idx, :]
                else:
                    logit_n = student_logits[i, j, :]
                student_filtered_logits[n] = logit_n

            with torch.autocast(device_type=device_type, dtype=dtype):
                soft_pseudo_labels = torch.softmax(teacher_filtered_logits / args.temp,
                                                   dim=1).detach()
                max_probs, _ = torch.max(soft_pseudo_labels, dim=1)
                mask = max_probs.ge(args.teacher_confidence).float()
                if args.pl_loss == "kl" or args.pl_loss == "mse":
                    mask = mask.unsqueeze(1).expand(-1, soft_pseudo_labels.size(1))
                    if args.pl_loss == "kl":
                        loss_s = s_loss_pseudo_fn(torch.log_softmax(student_filtered_logits, dim=1), 
                                                  soft_pseudo_labels) * mask
                        loss_s = loss_s.sum() / student_filtered_logits.size(0) * (args.temp ** 2)
                    else:
                        loss_s = s_loss_pseudo_fn(student_filtered_logits,
                                                  teacher_filtered_logits)
                        loss_s = torch.mean(loss_s * mask)
                else:
                    loss_s = s_loss_pseudo_fn(student_filtered_logits, soft_pseudo_labels)
                    loss_s = torch.mean(loss_s * mask)
                mask = mask.int()
                if args.pl_loss == "kl" or args.pl_loss == "mse":
                    mask = mask.sum(dim=1)
                mask = (mask > 0)
                if args.output_hidden_states and mask.sum().item() > 0:
                    flat_hidden_s = torch.flatten(student_hidden_states[-1], 
                                                  start_dim=0, end_dim=1)
                    flat_hidden_t = torch.flatten(teacher_hidden_states[-1], 
                                                  start_dim=0, end_dim=1)
                    selected_cls = cls_indexes[mask][:, 0] * labeled_data.size(1)\
                                   + cls_indexes[mask][:, 1]
                    cls_embeds_s: torch.Tensor = torch.cat([flat_hidden_s[selected_cls]], dim=0)
                    cls_embeds_t: torch.Tensor = torch.cat([flat_hidden_t[selected_cls]], dim=0)
                    cls_embeds_t = cls_embeds_t.detach()
                    cos_embedding_loss = loss_fn_embedding(cls_embeds_s, cls_embeds_t,
                                                           torch.ones(cls_embeds_s.size(0),
                                                                      device=device))
                    if torch.isnan(cos_embedding_loss).sum().item() == 0:
                        loss_s = loss_s + cos_embedding_loss
                        if batch_idx % 50 == 0:
                            print(selected_cls.tolist())

            loss_s.backward()
            optimizer_student_u.step()
            student.zero_grad()
            optimizer_student_u.zero_grad()

        student.eval()
        teacher.eval()
        
        print("Evaluation starts...")
        tr_loss_t /= len(train_dataloader_true_label)
        vl_loss_s = eval_model(valid_dataloader, device_type, student,
                               dtype, device, vl_pred_list_s, vl_true_list_s, 
                               tokenizer, valid_loss_fn, args)
        vl_loss_t = eval_model(valid_dataloader, device_type, teacher,
                               dtype, device, vl_pred_list_t, vl_true_list_t, 
                               tokenizer, valid_loss_fn, args)

        tr_macro_f1_t, tr_micro_f1_t = macro_micro_f1s(tr_true_list_t,
                                                       tr_pred_list_t)
        vl_macro_f1_t, vl_micro_f1_t = macro_micro_f1s(vl_true_list_t,
                                                       vl_pred_list_t)
        vl_macro_f1_s, vl_micro_f1_s = macro_micro_f1s(vl_true_list_s,
                                                       vl_pred_list_s)

        tr_loss_avg_t = tr_loss_t
        vl_loss_avg_t = vl_loss_t

        vl_loss_avg_s = vl_loss_s
        ts_loss_avg_s = ts_loss_s

        tr_micro_f1_avg_t = tr_micro_f1_t
        tr_macro_f1_avg_t = tr_macro_f1_t
        vl_micro_f1_avg_t = vl_micro_f1_t
        vl_macro_f1_avg_t = vl_macro_f1_t

        vl_micro_f1_avg_s = vl_micro_f1_s
        vl_macro_f1_avg_s = vl_macro_f1_s

        t2 = time()
        loss_info_lists_t[0].append([
            tr_loss_avg_t, tr_macro_f1_avg_t, tr_micro_f1_avg_t, 
            vl_loss_avg_t, vl_macro_f1_avg_t, vl_micro_f1_avg_t
        ])

        loss_info_lists_s[0].append([ 
            vl_loss_avg_s, vl_macro_f1_avg_s, vl_micro_f1_avg_s
        ])

        compare_save(student, optimizer_student_u, vl_micro_f1_avg_s,
                     best_vl_micro_f1s_s, f"{student_tag_name}/best_micro_f1", 
                     args)
        compare_save(student, optimizer_student_u, vl_macro_f1_avg_s,
                     best_vl_macro_f1s_s, f"{student_tag_name}/best_macro_f1", 
                     args)
        compare_save(teacher, optimizer_teacher_u, vl_micro_f1_avg_t,
                     best_vl_micro_f1s_t, f"{teacher_tag}/best_micro_f1", 
                     args)
        compare_save(teacher, optimizer_teacher_u, vl_macro_f1_avg_t,
                     best_vl_macro_f1s_t, f"{teacher_tag}/best_macro_f1", 
                     args)

        print(
            "Teacher Epoch {} ({}): "
            "tr_loss={:.7f} tr_macro_f1={:.4f} tr_micro_f1={:.4f} ({:.2f} sec.)"
            .format(epoch, "sato", tr_loss_avg_t, 
                    tr_macro_f1_avg_t, tr_micro_f1_avg_t, (t2 - t1)),
        )
        print(
            "Teacher Epoch {} ({}): "
            "vl_loss={:.7f} vl_macro_f1={:.4f} vl_micro_f1={:.4f} ({:.2f} sec.)"
            .format(epoch, "sato", vl_loss_avg_t, 
                    vl_macro_f1_avg_t, vl_micro_f1_avg_t, (t2 - t1)),
        )
        print(
            "Student Epoch {} ({}): "
            "vl_loss={:.7f} vl_macro_f1={:.4f} vl_micro_f1={:.4f} ({:.2f} sec.)"
            .format(epoch, "sato", vl_loss_avg_s, vl_macro_f1_avg_s, 
                    vl_micro_f1_avg_s, (t2 - t1))
        )

    if not args.test_only:
        for task, loss_info_list_s, loss_info_list_t in zip(args.tasks, 
                                                            loss_info_lists_s,
                                                            loss_info_lists_t):
            loss_info_df = pd.DataFrame(loss_info_list_s,
                                        columns=[
                                            "vl_loss", "vl_macro_f1", "vl_micro_f1"
                                        ])
            loss_info_df.to_csv(f"{student_tag_name}/loss_info.csv")

            loss_info_df = pd.DataFrame(loss_info_list_t,
                                        columns=[
                                            "tr_loss", "tr_macro_f1", "tr_micro_f1",
                                            "vl_loss", "vl_macro_f1", "vl_micro_f1"
                                        ])
            loss_info_df.to_csv(f"{teacher_tag}/loss_info.csv")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shortcut_name",
        default="bert",
        type=str,
        help="Huggingface model shortcut name ",
    )
    parser.add_argument(
        "--aug_type",
        default="dropcol",
        type=str,
        choices=["shuffle", "dropcol", "dropout"],
        help="Augmentation type ",
    )
    parser.add_argument(
        "--max_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--epoch",
        default=30,
        type=int,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--random_seed",
        default=4649,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="Local Rank. Necessary for using the torch.distributed.launch utility.",
    )
    parser.add_argument(
        "--num_classes",
        default=78,
        type=int,
        help="Number of classes",
    )
    parser.add_argument("--test_only",
                        action="store_true",
                        default=False,
                        help="Test without saving results")
    parser.add_argument("--output_hidden_states",
                        action="store_true",
                        default=False,
                        help="Output model hidden states")
    parser.add_argument("--use_augmentation",
                        action="store_true",
                        default=False,
                        help="Use augmentation and regularization")
    parser.add_argument("--feedback",
                        action="store_true",
                        default=False,
                        help="Train with feedback")
    parser.add_argument("--feedback_only",
                        action="store_true",
                        default=False,
                        help="Train with feedback but no labeled loss")
    parser.add_argument(
        "--report_period",
        default=3000,
        type=int,
        help="Report period",
    )
    parser.add_argument("--fp16",
                        action="store_true",
                        default=False,
                        help="Use FP16")
    parser.add_argument("--warmup",
                        type=float,
                        default=0.,
                        help="Warmup ratio")
    parser.add_argument("--temp",
                        type=float,
                        default=0.5,
                        help="Temperature for softmax")
    parser.add_argument("--lr",
                        type=float,
                        default=5e-5,
                        help="Learning rate")
    parser.add_argument("--tasks",
                        type=str,
                        nargs="+",
                        default=["turl"],
                        choices=[
                            "sato", "sato0", "sato1", "sato2", "sato3", "sato4",
                            "msato0", "msato1", "msato2", "msato3", "msato4",
                            "turl", "turl-re"
                        ],
                        help="Task names}")
    parser.add_argument("--colpair",
                        action="store_true",
                        help="Use column pair embedding")
    parser.add_argument("--true_ratios",
                        type=str,
                        nargs="+",
                        default=[],
                        help="e.g., --true_ratios turl=0.8 turl-re=0.1")
    parser.add_argument("--from_scratch",
                        action="store_true",
                        help="Training from scratch")
    parser.add_argument("--single_col",
                        default=False,
                        action="store_true",
                        help="Training with single column model")
    parser.add_argument(
        "--reg_loss",
        default="ce",
        type=str,
        choices=["ce", "kl", "mse"],
        help="Choose regularization loss"
    )
    parser.add_argument(
        "--teacher_confidence",
        default=0.5,
        type=float,
        help="Confidence value of teacher's predictions",
    )
    parser.add_argument(
        "--reg_confidence",
        default=0.95,
        type=float,
        help="Confidence to select pseudo labels in regularization",
    )
    parser.add_argument(
        "--u_ratio",
        default=8,
        type=int,
        help="Unlabeled to labeled ratio per batch",
    )
    parser.add_argument(
        "--u_lambda",
        default=0.005,
        type=float,
        help="Weight of unlabeled loss to other losses",
    )
    parser.add_argument("--pl_loss",
                        default="ce",
                        type=str,
                        choices=["ce", "kl", "mse"],
                        help="Pseudo label loss function")
    parser.add_argument(
        "--teacher_dropout",
        default=0.1,
        type=float,
        help="Dropout rate of teacher model",
    )

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()