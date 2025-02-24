import argparse
import json
import math
import os
from time import time
from collections import OrderedDict
import random
import numpy as np
import gc

import pandas as pd
import torch
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer

from dataset import (
    collate_fn,
    TURLColTypeTablewiseDatasetFromDF
)

from model import BertForMultiOutputClassification,\
                  DistilBertForMultiOutputClassification,\
                  ASLOptimized
from util import f1_score_multilabel


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
            pred_list.extend((filtered_logits >= math.log(0.5))
                             .cpu().detach().numpy().tolist())
            true_list.extend(labels.cpu().detach().numpy().tolist())

            with torch.autocast(device_type=device_type, dtype=dtype):
                loss += loss_fn(filtered_logits, labels.float()).item()

    return loss / len(eval_dataloader)


def compare_save(model, optimizer: AdamW, metric: float,
                 best_metrics: list, path, args):
    if metric > best_metrics[0] and not args.test_only:
        best_metrics[0] = metric
        model.save_pretrained(path)
        torch.save(optimizer.state_dict(), f"{path}/optimizer.pt")


def macro_micro_f1s(true_list: list, pred_list: list):
    micro_f1, macro_f1, _, _ = f1_score_multilabel(true_list, pred_list)
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

    if args.feedback_only:
        assert args.feedback == True

    args.tasks = sorted(args.tasks)

    task_num_class_dict = {
        "turl": 255,
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

    true_ratio = str(round(true_ratio_dict["turl"], 2))
    feedback_path = "" if not args.feedback else "/feedback"
    hidden_state_path = "/pseudo" if not args.output_hidden_states\
                        else "/pseudo_hidden"
    feedback_only_path = "/no_asl" if args.feedback_only else "/asl"
    student_tag_name = f"model/semi/teachers_1{feedback_path}"\
                       + f"/true_{true_ratio}/bs{batch_size}{hidden_state_path}"\
                       + f"{feedback_only_path}_teacher"
    teacher_tag = f"model/ensemble/1/true_sampled_{true_ratio}_cols/bs{batch_size}"\
                  + f"{hidden_state_path}{feedback_path}{feedback_only_path}"

    print(student_tag_name)
    print(teacher_tag)

    if not os.path.exists(student_tag_name) and not args.test_only:
        print(f"{student_tag_name} does not exist. Created")
        os.makedirs(student_tag_name)

    if not os.path.exists(teacher_tag) and not args.test_only:
        print(f"{teacher_tag} does not exist. Created")
        os.makedirs(teacher_tag)

    tokenizer = BertTokenizer.from_pretrained("bert")

    # Initialize models and move to device
    student_model = DistilBertForMultiOutputClassification.from_pretrained(
        'distil-bert',
        num_labels=255,
        output_attentions=False,
        output_hidden_states=args.output_hidden_states,
    ).to(device)

    teacher_model = BertForMultiOutputClassification.from_pretrained(
        'bert',
        num_labels=255,
        output_attentions=False,
        output_hidden_states=args.output_hidden_states,
        hidden_dropout_prob=args.teacher_dropout
    ).to(device)

    true_cls = TURLColTypeTablewiseDatasetFromDF
    valid_cls = TURLColTypeTablewiseDatasetFromDF
    test_cls = TURLColTypeTablewiseDatasetFromDF

    true_df: pd.DataFrame = pd.read_pickle(f"data/table_col_type_serialized_grouped_sampled_{true_ratio}.pkl")
    valid_df: pd.DataFrame = pd.read_pickle("data/table_col_type_serialized_grouped_valid.pkl")
    test_df: pd.DataFrame = pd.read_pickle("data/table_col_type_serialized_grouped_test.pkl")
    
    train_dataset_true_label = true_cls(table_df=true_df,
                                        device=device)
    valid_dataset = valid_cls(table_df=valid_df,
                              device=device)
    test_dataset = test_cls(table_df=test_df,
                            device=device)

    train_sampler_true_label = RandomSampler(train_dataset_true_label)
    
    train_dataloader_true_label = DataLoader(train_dataset_true_label,
                                             sampler=train_sampler_true_label,
                                             batch_size=batch_size,
                                             collate_fn=collate_fn,
                                             pin_memory=True)

    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=batch_size,
                                  collate_fn=collate_fn,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 collate_fn=collate_fn,
                                 pin_memory=True)

    no_decay = ["bias", "LayerNorm.weight"]

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
    optimizer_teacher_u = AdamW(optimizer_grouped_parameters_teacher, lr=1e-5)
    
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

    # Loss functions
    l1 = BCEWithLogitsLoss(reduction="none").to(device)
    l3 = BCEWithLogitsLoss().to(device)
    teacher_loss_labeled = ASLOptimized().to(device)
    loss_fn_embedding = MSELoss().to(device)

    valid_loss_fn = BCEWithLogitsLoss()

    # Best validation score could be zero
    best_vl_micro_f1s_s = [-1 for _ in range(len(args.tasks))]
    best_vl_macro_f1s_s = [-1 for _ in range(len(args.tasks))]
    loss_info_lists_s = [[] for _ in range(len(args.tasks))]
    loss_info_lists_t = [[] for _ in range(len(args.tasks))]
    best_vl_micro_f1s_t = [-1 for _ in range(len(args.tasks))]
    best_vl_macro_f1s_t = [-1 for _ in range(len(args.tasks))]

    for epoch in range(num_train_epochs):
        t1 = time()

        teacher, student = teacher_model, student_model

        vl_true_list_s = []
        vl_pred_list_s = []
        vl_true_list_t = []
        vl_pred_list_t = []

        ts_true_list_s = []
        ts_pred_list_s = []
        ts_pred_list_t = []
        ts_true_list_t = []

        for idx, batch in enumerate(train_dataloader_true_label):

            if idx % args.report_period == 0:
                print(f"Batch {idx}: {round(time() - t1, 3)} seconds.", flush=True)

            labeled_data: torch.Tensor = batch["data"].to(device).T
            true_labels: torch.Tensor = batch["label"].to(device)
            attn_mask: torch.Tensor = (labeled_data != 0).float()

            student.train()
            teacher.eval()

            input_data = labeled_data
            input_attn_mask = attn_mask

            cls_indexes = torch.nonzero(input_data == tokenizer.cls_token_id)

            with torch.autocast(device_type=device_type, dtype=dtype):
                if args.output_hidden_states:
                    teacher_logits, teacher_hidden_states = teacher(input_data, input_attn_mask)
                else:
                    teacher_logits, = teacher(input_data, input_attn_mask)

            is_expanded = False
            if len(teacher_logits.shape) == 2:
                is_expanded = True
                teacher_logits = teacher_logits.unsqueeze(0)
            
            teacher_filtered_logits = torch.zeros(cls_indexes.shape[0],
                                                  teacher_logits.shape[2]).to(device)

            for n in range(cls_indexes.shape[0]):
                i, j = cls_indexes[n]
                if is_expanded:
                    idx = i * input_data.shape[0] + j
                    logit_n = teacher_logits[0, idx, :]
                else:
                    logit_n = teacher_logits[i, j, :]
                teacher_filtered_logits[n] = logit_n

            if args.feedback:
                with torch.autocast(device_type=device_type, dtype=dtype):
                    if args.output_hidden_states:
                        student_logits, student_hidden_states = student(input_data, input_attn_mask)
                    else:
                        student_logits, = student(input_data, input_attn_mask)

                is_expanded = False
                if len(student_logits.shape) == 2:
                    is_expanded = True
                    student_logits = student_logits.unsqueeze(0)
                
                student_filtered_logits = torch.zeros(cls_indexes.shape[0],
                                                      student_logits.shape[2]).to(device)
                
                for n in range(cls_indexes.shape[0]):
                    i, j = cls_indexes[n]
                    if is_expanded:
                        idx = i * input_data.shape[0] + j
                        logit_n = student_logits[0, idx, :]
                    else:
                        logit_n = student_logits[i, j, :]
                    student_filtered_logits[n] = logit_n

                with torch.autocast(device_type=device_type, dtype=dtype):
                    soft_pseudo_labels = torch.sigmoid(teacher_filtered_logits)
                    hard_pseudo_labels = soft_pseudo_labels.ge(args.teacher_confidence).detach()
                    row_sums = hard_pseudo_labels.sum(dim=1)
                    row_masks = (row_sums > 0).float()
                    mask = row_masks.unsqueeze(1).expand(-1, hard_pseudo_labels.size(1))
                    # hard_pseudo_labels = hard_pseudo_labels.float()
                    
                    loss_1 = torch.mean((l1(student_filtered_logits, soft_pseudo_labels) * mask))
                    
                    if args.output_hidden_states:
                        flat_hidden_s = torch.flatten(student_hidden_states[-1], 
                                                      start_dim=0, end_dim=1)
                        flat_hidden_t = torch.flatten(teacher_hidden_states[-1], 
                                                      start_dim=0, end_dim=1)
                        selected_cls = cls_indexes.clone()
                        selected_cls[:, 0] = selected_cls[:, 0] * input_data.size(1)
                        selected_cls[:, 1] = (selected_cls[:, 0] + selected_cls[:, 1])
                        cls_embeds_s: torch.Tensor = torch.cat([flat_hidden_s[s : e] 
                                                                for s, e in selected_cls], dim=0)
                        cls_embeds_t: torch.Tensor = torch.cat([flat_hidden_t[s : e] 
                                                                for s, e in selected_cls], dim=0)
                        mse_embedding_loss = loss_fn_embedding(cls_embeds_s, cls_embeds_t)
                        if torch.isnan(mse_embedding_loss).sum().item() == 0:
                            loss_1 = loss_1 + mse_embedding_loss

                    student_grads = torch.autograd.grad(loss_1, student.parameters(), 
                                                        create_graph=True, retain_graph=True)
                    updated_student_params = OrderedDict()
                    for name_param, grad in zip(student.named_parameters(), student_grads):
                        name, param = name_param
                        updated_student_params[name] = (param - args.lr * grad)
            
                student.eval()
                teacher.train()

                with torch.autocast(device_type=device_type, dtype=dtype):
                    if args.output_hidden_states:
                        student_logits, _ = torch.func.functional_call(student, 
                                                                       updated_student_params, 
                                                                       (input_data, input_attn_mask))
                    else:
                        student_logits, = torch.func.functional_call(student, 
                                                                     updated_student_params, 
                                                                     (input_data, input_attn_mask))

                is_expanded = False
                if len(student_logits.shape) == 2:
                    is_expanded = True
                    student_logits = student_logits.unsqueeze(0)
                student_filtered_logits = torch.zeros(cls_indexes.shape[0], 
                                                      student_logits.shape[2]).to(device)
                for n in range(cls_indexes.shape[0]):
                    i, j = cls_indexes[n]
                    if is_expanded:
                        idx = i * input_data.shape[0] + j
                        logit_n = student_logits[0, idx, :]
                    else:
                        logit_n = student_logits[i, j, :]
                    student_filtered_logits[n] = logit_n

                with torch.autocast(device_type=device_type, dtype=dtype):
                    loss_3 = l3(student_filtered_logits, true_labels.float())
            
            if not args.feedback_only:
                with torch.autocast(device_type=device_type, dtype=dtype):
                    loss_teacher_labeled = teacher_loss_labeled(teacher_filtered_logits, 
                                                                true_labels.float())
                if args.feedback:
                    teacher_loss = loss_3 + loss_teacher_labeled
                else:
                    teacher_loss = loss_teacher_labeled
            else:
                teacher_loss = loss_3
            teacher_grads = torch.autograd.grad(teacher_loss,
                                                teacher.parameters())
            for tp, tg in zip(teacher.parameters(), teacher_grads):
                tp.grad = tg

            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            optimizer_teacher_u.step()
            optimizer_teacher_u.zero_grad()
            for tp in teacher.parameters():
                tp.grad = None

            del loss_1
            del loss_3
            del teacher_loss
            del student_grads
            del teacher_grads
            del updated_student_params
            gc.collect()
            torch.cuda.empty_cache()

            if args.feedback:
                teacher.eval()
                student.train()

                with torch.autocast(device_type=device_type, dtype=dtype):
                    with torch.no_grad():
                        if args.output_hidden_states:
                            teacher_logits, teacher_hidden_states = teacher(input_data, 
                                                                            input_attn_mask)
                        else:
                            teacher_logits, = teacher(input_data, input_attn_mask)

                is_expanded = False
                if len(teacher_logits.shape) == 2:
                    is_expanded = True
                    teacher_logits = teacher_logits.unsqueeze(0)
                
                teacher_filtered_logits = torch.zeros(cls_indexes.shape[0],
                                                      teacher_logits.shape[2]).to(device)

                for n in range(cls_indexes.shape[0]):
                    i, j = cls_indexes[n]
                    if is_expanded:
                        idx = i * input_data.shape[0] + j
                        logit_n = teacher_logits[0, idx, :]
                    else:
                        logit_n = teacher_logits[i, j, :]
                    teacher_filtered_logits[n] = logit_n
                
                with torch.autocast(device_type=device_type, dtype=dtype):
                    if args.output_hidden_states:
                        student_logits, student_hidden_states = student(input_data, input_attn_mask)
                    else:
                        student_logits, = student(input_data, input_attn_mask)

                is_expanded = False
                if len(student_logits.shape) == 2:
                    is_expanded = True
                    student_logits = student_logits.unsqueeze(0)
                
                student_filtered_logits = torch.zeros(cls_indexes.shape[0],
                                                      student_logits.shape[2]).to(device)
                
                for n in range(cls_indexes.shape[0]):
                    i, j = cls_indexes[n]
                    if is_expanded:
                        idx = i * input_data.shape[0] + j
                        logit_n = student_logits[0, idx, :]
                    else:
                        logit_n = student_logits[i, j, :]
                    student_filtered_logits[n] = logit_n

                with torch.autocast(device_type=device_type, dtype=dtype):
                    soft_pseudo_labels = torch.sigmoid(teacher_filtered_logits)
                    hard_pseudo_labels = soft_pseudo_labels.ge(args.teacher_confidence).detach()
                    row_sums = hard_pseudo_labels.sum(dim=1)
                    row_masks = (row_sums > 0).float()
                    mask = row_masks.unsqueeze(1).expand(-1, hard_pseudo_labels.size(1))
                    
                    loss_1 = torch.mean((l1(student_filtered_logits, soft_pseudo_labels) * mask))
                    
                    if args.output_hidden_states:
                        flat_hidden_s = torch.flatten(student_hidden_states[-1], 
                                                      start_dim=0, end_dim=1)
                        flat_hidden_t = torch.flatten(teacher_hidden_states[-1], 
                                                      start_dim=0, end_dim=1)
                        selected_cls = cls_indexes.clone()
                        selected_cls[:, 0] = selected_cls[:, 0] * input_data.size(1)
                        selected_cls[:, 1] = (selected_cls[:, 0] + selected_cls[:, 1])
                        cls_embeds_s: torch.Tensor = torch.cat([flat_hidden_s[s : e] 
                                                                for s, e in selected_cls], dim=0)
                        cls_embeds_t: torch.Tensor = torch.cat([flat_hidden_t[s : e] 
                                                                for s, e in selected_cls], dim=0)
                        cls_embeds_t = cls_embeds_t.detach()
                        mse_embedding_loss = loss_fn_embedding(cls_embeds_s, cls_embeds_t)
                        if torch.isnan(mse_embedding_loss).sum().item() == 0:
                            loss_1 = loss_1 + mse_embedding_loss

                    loss_1.backward()
                    optimizer_student_u.step()
                    optimizer_student_u.zero_grad()
                    student.zero_grad()

        # Validation
        student.eval()
        teacher.eval()

        print("Evaluation starts...")
        vl_loss_s = eval_model(valid_dataloader, device_type, student,
                               dtype, device, vl_pred_list_s, vl_true_list_s, 
                               tokenizer, valid_loss_fn, args)
        vl_loss_t = eval_model(valid_dataloader, device_type, teacher,
                               dtype, device, vl_pred_list_t, vl_true_list_t, 
                               tokenizer, valid_loss_fn, args)
        ts_loss_s = eval_model(test_dataloader, device_type, student,
                               dtype, device, ts_pred_list_s, ts_true_list_s, 
                               tokenizer, valid_loss_fn, args)
        ts_loss_t = eval_model(test_dataloader, device_type, teacher,
                               dtype, device, ts_pred_list_t, ts_true_list_t, 
                               tokenizer, valid_loss_fn, args)

        vl_macro_f1_t, vl_micro_f1_t = macro_micro_f1s(vl_true_list_t,
                                                       vl_pred_list_t)
        vl_macro_f1_s, vl_micro_f1_s = macro_micro_f1s(vl_true_list_s,
                                                       vl_pred_list_s)
        ts_macro_f1_t, ts_micro_f1_t = macro_micro_f1s(ts_true_list_t,
                                                       ts_pred_list_t)
        ts_macro_f1_s, ts_micro_f1_s = macro_micro_f1s(ts_true_list_s,
                                                       ts_pred_list_s)

        vl_loss_avg_t = vl_loss_t
        ts_loss_avg_t = ts_loss_t

        vl_loss_avg_s = vl_loss_s
        ts_loss_avg_s = ts_loss_s

        vl_micro_f1_avg_t = vl_micro_f1_t
        vl_macro_f1_avg_t = vl_macro_f1_t
        ts_micro_f1_avg_t = ts_micro_f1_t
        ts_macro_f1_avg_t = ts_macro_f1_t

        vl_micro_f1_avg_s = vl_micro_f1_s
        vl_macro_f1_avg_s = vl_macro_f1_s
        ts_micro_f1_avg_s = ts_micro_f1_s
        ts_macro_f1_avg_s = ts_macro_f1_s

        t2 = time()
        loss_info_lists_t[0].append([
            vl_loss_avg_t, vl_macro_f1_avg_t, vl_micro_f1_avg_t,
            ts_loss_avg_t, ts_macro_f1_avg_t, ts_micro_f1_avg_t
        ])

        loss_info_lists_s[0].append([ 
            vl_loss_avg_s, vl_macro_f1_avg_s, vl_micro_f1_avg_s,
            ts_loss_avg_s, ts_macro_f1_avg_s, ts_micro_f1_avg_s
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
            "vl_loss={:.7f} vl_macro_f1={:.4f} vl_micro_f1={:.4f} ({:.2f} sec.)"
            .format(epoch, "sato", vl_loss_avg_t, 
                    vl_macro_f1_avg_t, vl_micro_f1_avg_t, (t2 - t1)),
        )
        print(
            "Teacher Epoch {} ({}): "
            "ts_loss={:.7f} ts_macro_f1={:.4f} ts_micro_f1={:.4f} ({:.2f} sec.)"
            .format(epoch, "sato", ts_loss_avg_t, 
                    ts_macro_f1_avg_t, ts_micro_f1_avg_t, (t2 - t1)),
        )
        print(
            "Student Epoch {} ({}): "
            "vl_loss={:.7f} vl_macro_f1={:.4f} vl_micro_f1={:.4f} ({:.2f} sec.)"
            .format(epoch, "sato", vl_loss_avg_s, vl_macro_f1_avg_s, 
                    vl_micro_f1_avg_s, (t2 - t1))
        )
        print(
            "Student Epoch {} ({}): "
            "ts_loss={:.7f} ts_macro_f1={:.4f} ts_micro_f1={:.4f} ({:.2f} sec.)"
            .format(epoch, "sato", ts_loss_avg_s, 
                    ts_macro_f1_avg_s, ts_micro_f1_avg_s, (t2 - t1))
        )

    if not args.test_only:
        for task, loss_info_list_s, loss_info_list_t in zip(args.tasks, 
                                                            loss_info_lists_s,
                                                            loss_info_lists_t):
            loss_info_df = pd.DataFrame(loss_info_list_s,
                                        columns=[
                                            "vl_loss", "vl_macro_f1", "vl_micro_f1",
                                            "ts_loss", "ts_macro_f1", "ts_micro_f1"
                                        ])
            loss_info_df.to_csv(f"{student_tag_name}/loss_info.csv")

            loss_info_df = pd.DataFrame(loss_info_list_t,
                                        columns=[
                                            "vl_loss", "vl_macro_f1", "vl_micro_f1",
                                            "ts_loss", "ts_macro_f1", "ts_micro_f1"
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
        "--f1_type",
        default="macro",
        type=str,
        choices=["micro", "macro"],
        help="Choose best micro or macro teacher model ",
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
        default=40,
        type=int,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--period",
        default=8,
        type=int,
        help="Number of epochs per pseudo-labeling/feedback period",
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
    parser.add_argument("--feedback",
                        action="store_true",
                        default=False,
                        help="Train with feedback")
    parser.add_argument("--feedback_only",
                        action="store_true",
                        default=False,
                        help="Train with feedback but no labeled loss")
    parser.add_argument("--output_hidden_states",
                        action="store_true",
                        default=False,
                        help="Output model hidden states")
    parser.add_argument("--use_augmented_teacher",
                        action="store_true",
                        default=False,
                        help="Teacher with augmented regularization")
    parser.add_argument(
        "--report_period",
        default=3000,
        type=int,
        help="Report period.",
    )
    parser.add_argument("--warmup",
                        type=float,
                        default=0.,
                        help="Warmup ratio")
    parser.add_argument("--lr",
                        type=float,
                        default=5e-5,
                        help="Learning rate")
    parser.add_argument("--tasks",
                        type=str,
                        nargs="+",
                        default=["turl"],
                        choices=[
                            "sato0", "sato1", "sato2", "sato3", "sato4",
                            "msato0", "msato1", "msato2", "msato3", "msato4",
                            "turl", "turl-re"
                        ],
                        help="Task names}")
    parser.add_argument("--colpair",
                        action="store_true",
                        help="Use column pair embedding")
    parser.add_argument("--train_ratios",
                        type=str,
                        nargs="+",
                        default=[],
                        help="e.g., --train_ratios turl=0.8 turl-re=0.1")
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
        "--teacher_confidence",
        default=0.7,
        type=float,
        help="Confidence value of teacher's predictions",
    )
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