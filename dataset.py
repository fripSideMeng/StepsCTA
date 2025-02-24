from functools import reduce
import operator
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import transformers


def collate_fn(samples):
    data = torch.nn.utils.rnn.pad_sequence([sample["data"] for sample in samples])
    label = torch.cat([sample["label"] for sample in samples])
    batch = {"data": data, "label": label}
    if "idx" in samples[0]:
        batch["idx"] = [sample["idx"] for sample in samples]
    if "num_col" in samples[0]:
        batch["num_col"] = [sample["num_col"] for sample in samples]
    if "dropped_col" in samples[0]:
        batch["dropped_col"] = [sample["dropped_col"] for sample in samples]
    if "col_mapping" in samples[0]:
        batch["col_mapping"] = [sample["col_mapping"] for sample in samples]
    if "data_aug" in samples[0]:
        data_aug = torch.nn.utils.rnn.pad_sequence([sample["data_aug"] for sample in samples])
        batch["data_aug"] = data_aug
    return batch


class SatoCVTablewiseDatasetFromDF(Dataset):

    def __init__(self,
                 table_df: pd.DataFrame,
                 device: torch.device = None,
                 multicol_only: bool = False):
        self.table_df = table_df
        self.device = device if device is not None\
                      else torch.device('cpu')
        self.loaded_rows = dict()

        if multicol_only:
            self.table_df = self.table_df[self.table_df["num_col"] > 1]


    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "idx": idx,
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }
    

class SatoCVTablewiseDatasetFromDFAug(Dataset):

    def __init__(self,
                 table_df: pd.DataFrame,
                 device: torch.device = None,
                 aug_type: str = "dropout"):
        self.table_df = table_df
        self.device = device if device is not None\
                      else torch.device('cpu')
        self.aug_type = aug_type


    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        item = {
            "idx": idx,
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"],
            "num_col": self.table_df.iloc[idx]["num_col"]
        }
        if self.aug_type == "dropcol":
            item["dropped_col"] = self.table_df.iloc[idx]["dropped_col"]
        elif self.aug_type == "shuffle":
            item["col_mapping"] = self.table_df.iloc[idx]["col_mapping"]
        if self.aug_type != "dropout":
            item["data_aug"] = self.table_df.iloc[idx]["data_tensor_aug"]
        return item


class TURLColTypeTablewiseDatasetFromDF(Dataset):

    def __init__(self,
                 table_df: pd.DataFrame,
                 device: torch.device = None,
                 multicol_only: bool = False):
        self.table_df = table_df
        self.device = device if device is not None\
                      else torch.device('cpu')
        self.loaded_rows = dict()

        if multicol_only:
            self.table_df = self.table_df[self.table_df["num_col"] > 1]


    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "idx": idx,
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }
    

class TURLColTypeTablewiseDatasetFromDFAug(Dataset):

    def __init__(self,
                 table_df: pd.DataFrame,
                 device: torch.device = None,
                 aug_type: str = "dropout"):
        self.table_df = table_df
        self.device = device if device is not None\
                      else torch.device('cpu')
        self.aug_type = aug_type


    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        item = {
            "idx": idx,
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"],
            "data_aug": self.table_df.iloc[idx]["data_tensor_aug"],
            "num_col": self.table_df.iloc[idx]["num_col"]
        }
        if self.aug_type == "dropcol":
            item["dropped_col"] = self.table_df.iloc[idx]["dropped_col"]
        elif self.aug_type == "shuffle":
            item["col_mapping"] = self.table_df.iloc[idx]["col_mapping"]
        return item