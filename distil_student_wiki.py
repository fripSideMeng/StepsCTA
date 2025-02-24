import argparse
import json
import math
import os
from time import time
import random
import numpy as np

import pandas as pd
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss, MSELoss, CosineEmbeddingLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import BertTokenizer

from dataset import (
    collate_fn,
    TURLColTypeTablewiseDatasetFromDF
)

from model import BertForMultiOutputClassification,\
                  DistilBertForMultiOutputClassification
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
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32 if torch.cuda.is_available() else torch.bfloat16

    dist.init_process_group(backend=backend, init_method='env://')
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    if device_type == 'cuda':
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available()\
             else torch.device("cpu")
    world_size = dist.get_world_size()
    if rank == 0:
        print(world_size)

    # Set random seeds for reproducibility
    torch.manual_seed(args.random_seed + rank)
    if device_type == 'cuda':
        torch.cuda.manual_seed(args.random_seed + rank)
    random.seed(args.random_seed + rank)
    np.random.seed(args.random_seed + rank)

    if device_type == 'cuda':
        scaler = torch.GradScaler()

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

    if rank == 0:
        print(f"args={json.dumps(vars(args))}")

    batch_size = args.batch_size // world_size
    num_train_epochs = args.epoch

    true_ratio = str(round(true_ratio_dict["turl"], 2))
    hidden_state_path = "/pseudo" if not args.output_hidden_states\
                        else "/pseudo_hidden"
    feedback_only_path = "/no_asl" if args.teacher_feedback_only else "/asl"
    student_tag_name = f"model/semi/teachers_1/feedback"\
                       + f"/true_{true_ratio}/bs{batch_size * world_size}"\
                       + f"{hidden_state_path}"
    teacher_tag = f"model/ensemble/1/true_sampled_{true_ratio}_cols/"\
                  + f"bs{batch_size * world_size}"\
                  + f"/pseudo_hidden/feedback{feedback_only_path}"
    pseudo_ratio = str(round(1.0 - true_ratio_dict["turl"], 2))

    if rank == 0:
        print(student_tag_name)
        print(teacher_tag)

    tokenizer = BertTokenizer.from_pretrained("bert")

    # Initialize models and move to device
    student_model = DistilBertForMultiOutputClassification.from_pretrained(
        f"{student_tag_name}/best_macro_f1",
        num_labels=255,
        output_attentions=False,
        output_hidden_states=args.output_hidden_states,
    ).to(device)

    teacher_model = BertForMultiOutputClassification.from_pretrained(
        f"{teacher_tag}/best_macro_f1",
        output_hidden_states=args.output_hidden_states
    ).to(device)

    if device_type == 'cuda':
        student_model = DDP(student_model, device_ids=[local_rank], output_device=local_rank, 
                            broadcast_buffers=False)

        teacher_model = DDP(teacher_model, device_ids=[local_rank], output_device=local_rank, 
                            broadcast_buffers=False)
    else:
        student_model = DDP(student_model, broadcast_buffers=False)

        teacher_model = DDP(teacher_model, broadcast_buffers=False)

    pseudo_cls = TURLColTypeTablewiseDatasetFromDF
    valid_cls = TURLColTypeTablewiseDatasetFromDF
    test_cls = TURLColTypeTablewiseDatasetFromDF

    true_df: pd.DataFrame = pd.read_pickle(f"data/table_col_type_serialized_grouped_sampled_{true_ratio}.pkl")
    pseudo_df: pd.DataFrame = pd.read_pickle(f"data/table_col_type_serialized_grouped_unsampled_{pseudo_ratio}.pkl")
    valid_df: pd.DataFrame = pd.read_pickle("data/table_col_type_serialized_grouped_valid.pkl")
    test_df: pd.DataFrame = pd.read_pickle("data/table_col_type_serialized_grouped_test.pkl")

    pseudo_df = pd.concat([true_df, pseudo_df], ignore_index=True)

    train_dataset_pseudo_label = pseudo_cls(table_df=pseudo_df,
                                            device=device)
    valid_dataset = valid_cls(table_df=valid_df,
                              device=device)
    test_dataset = test_cls(table_df=test_df,
                            device=device)

    train_sampler_pseudo_label = DistributedSampler(train_dataset_pseudo_label, 
                                                    num_replicas=world_size, 
                                                    rank=rank, shuffle=True)

    train_dataloader_pseudo_label = DataLoader(train_dataset_pseudo_label,
                                               sampler=train_sampler_pseudo_label,
                                               batch_size=batch_size,
                                               collate_fn=collate_fn,
                                               pin_memory=True)

    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=batch_size * world_size,
                                  collate_fn=collate_fn,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size * world_size,
                                 collate_fn=collate_fn,
                                 pin_memory=True)

    no_decay = ["bias", "LayerNorm.weight"]
    
    optimizer_grouped_parameters_student = [
        {
            "params": [
                p for n, p in student_model.module.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0
        },
        {
            "params": [
                p for n, p in student_model.module.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0
        },
    ]
    optimizer_student_u = AdamW(optimizer_grouped_parameters_student, lr=args.lr)

    # Loss functions
    l1 = BCEWithLogitsLoss(reduction="none").to(device)
    if args.embed_loss == "cos":
        loss_fn_embedding = CosineEmbeddingLoss().to(device)
    else:
        loss_fn_embedding = MSELoss().to(device)
    valid_loss_fn = BCEWithLogitsLoss().to(device)

    best_vl_micro_f1s = [-1. for _ in range(len(args.tasks))]
    best_vl_macro_f1s = [-1. for _ in range(len(args.tasks))]
    loss_info_lists = [[] for _ in range(len(args.tasks))]

    for epoch in range(num_train_epochs):
        t1 = time()

        train_sampler_pseudo_label.set_epoch(epoch)

        teacher, student = teacher_model, student_model

        vl_true_list_s = []
        vl_pred_list_s = []
        ts_true_list_s = []
        ts_pred_list_s = []

        student.train()
        teacher.eval()

        for idx, batch in enumerate(train_dataloader_pseudo_label):

            if idx % args.report_period == 0 and rank == 0:
                print(f"Batch {idx}: {round(time() - t1, 3)} seconds.", flush=True)

            unlabeled_data: torch.Tensor = batch["data"].to(device).T
            attn_mask_u: torch.Tensor = (unlabeled_data != 0).float()

            cls_indexes = torch.nonzero(unlabeled_data == tokenizer.cls_token_id)

            with torch.autocast(device_type=device_type, dtype=dtype):
                with torch.no_grad():
                    if args.output_hidden_states:
                        teacher_logits, teacher_hidden_states = teacher(unlabeled_data,
                                                                        attn_mask_u)
                    else:
                        teacher_logits, = teacher(unlabeled_data,
                                                  attn_mask_u)

            is_expanded = False
            if len(teacher_logits.shape) == 2:
                is_expanded = True
                teacher_logits = teacher_logits.unsqueeze(0)
            
            teacher_filtered_logits = torch.zeros(cls_indexes.shape[0],
                                                  teacher_logits.shape[2]).to(device)

            for n in range(cls_indexes.shape[0]):
                i, j = cls_indexes[n]
                if is_expanded:
                    idx = i * unlabeled_data.shape[0] + j
                    logit_n = teacher_logits[0, idx, :]
                else:
                    logit_n = teacher_logits[i, j, :]
                teacher_filtered_logits[n] = logit_n

            with torch.autocast(device_type=device_type, dtype=dtype):
                if args.output_hidden_states:
                    student_logits, student_hidden_states = student(unlabeled_data,
                                                                    attn_mask_u)
                else:
                    student_logits, = student(unlabeled_data,
                                              attn_mask_u)

            is_expanded = False
            if len(student_logits.shape) == 2:
                is_expanded = True
                student_logits = student_logits.unsqueeze(0)
            
            student_filtered_logits = torch.zeros(cls_indexes.shape[0],
                                                  student_logits.shape[2]).to(device)
            
            for n in range(cls_indexes.shape[0]):
                i, j = cls_indexes[n]
                if is_expanded:
                    idx = i * unlabeled_data.shape[0] + j
                    logit_n = student_logits[0, idx, :]
                else:
                    logit_n = student_logits[i, j, :]
                student_filtered_logits[n] = logit_n

            with torch.autocast(device_type=device_type, dtype=dtype):
                soft_pseudo_labels = torch.sigmoid(teacher_filtered_logits)
                row_sums = (soft_pseudo_labels > args.teacher_confidence).sum(dim=1)
                row_masks = (row_sums > 0).float()
                mask = row_masks.unsqueeze(1).expand(-1, soft_pseudo_labels.size(1))
                
                loss_1 = torch.mean(l1(student_filtered_logits, soft_pseudo_labels) * mask)

                mask = (mask.int().sum(dim=1) > 0)
                
                if args.output_hidden_states and mask.sum().item() > 0:
                    hidden_states_lists_s = list()
                    hidden_states_lists_t = list()
                    student_hidden_states = student_hidden_states[1:]
                    teacher_hidden_states = teacher_hidden_states[1:]
                    for layer_idx in range(len(student_hidden_states)):
                        flat_hidden_s = torch.flatten(student_hidden_states[layer_idx], 
                                                      start_dim=0, end_dim=1)
                        flat_hidden_t = torch.flatten(teacher_hidden_states[layer_idx * 2 + 1], 
                                                      start_dim=0, end_dim=1)
                        selected_cls = cls_indexes[mask][:, 0] * unlabeled_data.size(1)\
                                       + cls_indexes[mask][:, 1]
                        cls_embeds_s: torch.Tensor = torch.cat([flat_hidden_s[selected_cls]], dim=0)
                        cls_embeds_t: torch.Tensor = torch.cat([flat_hidden_t[selected_cls]], dim=0)
                        hidden_states_lists_s.append(cls_embeds_s)
                        hidden_states_lists_t.append(cls_embeds_t)
                    
                    cls_embeds_s = torch.cat(hidden_states_lists_s)
                    cls_embeds_t = torch.cat(hidden_states_lists_t)
                    if args.embed_loss == "cos":
                        embedding_loss = loss_fn_embedding(cls_embeds_s, cls_embeds_t,
                                                           torch.ones(cls_embeds_s.size(0),
                                                                      device=device))
                    else:
                        embedding_loss = loss_fn_embedding(cls_embeds_s, cls_embeds_t)
                    if torch.isnan(embedding_loss).sum().item() == 0:
                        loss_1 = loss_1 + embedding_loss
                
                if device_type == 'cuda':
                    scaler.scale(loss_1).backward()
                    scaler.step(optimizer_student_u)
                else:
                    loss_1.backward()
                    optimizer_student_u.step()
                optimizer_student_u.zero_grad()
                student.zero_grad()

            if device_type == 'cuda':
                scaler.update()

        # Validation
        student.eval()
        teacher.eval()

        if rank == 0:
            print("Evaluation starts...")
            vl_loss_s = eval_model(valid_dataloader, device_type, student.module,
                                   dtype, device, vl_pred_list_s, vl_true_list_s, 
                                   tokenizer, valid_loss_fn, args)
            ts_loss_s = eval_model(test_dataloader, device_type, student.module,
                                   dtype, device, ts_pred_list_s, ts_true_list_s, 
                                   tokenizer, valid_loss_fn, args)

            vl_macro_f1_s, vl_micro_f1_s = macro_micro_f1s(vl_true_list_s,
                                                           vl_pred_list_s)
            ts_macro_f1_s, ts_micro_f1_s = macro_micro_f1s(ts_true_list_s,
                                                           ts_pred_list_s)

            vl_loss_avg_s = vl_loss_s
            ts_loss_avg_s = ts_loss_s

            vl_micro_f1_avg_s = vl_micro_f1_s
            vl_macro_f1_avg_s = vl_macro_f1_s
            ts_micro_f1_avg_s = ts_micro_f1_s
            ts_macro_f1_avg_s = ts_macro_f1_s

            t2 = time()

            loss_info_lists[0].append([ 
                vl_loss_avg_s, vl_macro_f1_avg_s, vl_micro_f1_avg_s,
                ts_loss_avg_s, ts_macro_f1_avg_s, ts_micro_f1_avg_s
            ])

            compare_save(student.module, optimizer_student_u, vl_micro_f1_avg_s,
                         best_vl_micro_f1s, f"{student_tag_name}/best_micro_f1_pl", 
                         args)
            compare_save(student.module, optimizer_student_u, vl_macro_f1_avg_s,
                         best_vl_macro_f1s, f"{student_tag_name}/best_macro_f1_pl", 
                         args)

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
        dist.barrier()

    if not args.test_only and rank == 0:
        for task, loss_info_list in zip(args.tasks, loss_info_lists):
            loss_info_df = pd.DataFrame(loss_info_list,
                                        columns=[
                                            "vl_loss", "vl_macro_f1", "vl_micro_f1",
                                            "ts_loss", "ts_macro_f1", "ts_micro_f1"
                                        ])

            loss_info_df.to_csv(f"{student_tag_name}/loss_info_pl.csv")


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
    parser.add_argument("--output_hidden_states",
                        action="store_true",
                        default=False,
                        help="Output model hidden states")
    parser.add_argument("--teacher_feedback_only",
                        action="store_true",
                        default=False,
                        help="Teacher trained with feedback only")
    parser.add_argument(
        "--report_period",
        default=3000,
        type=int,
        help="Report period.",
    )
    parser.add_argument("--fp16",
                        action="store_true",
                        default=False,
                        help="Use FP16")
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
    parser.add_argument("--embed_loss",
                        default="cos",
                        type=str,
                        choices=["cos", "mse"],
                        help="Hidden state alignment loss function")

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()