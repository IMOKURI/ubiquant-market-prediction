import lightgbm as lgb
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def make_dataset(c, df, label=True):
    if c.params.dataset == "ump_1":
        ds = BaseDataset(c, df, label)

    else:
        raise Exception("Invalid dataset.")
    return ds


def make_dataset_lightgbm(c, train_df, valid_df):
    train_labels = train_df[c.params.label_name]
    valid_labels = valid_df[c.params.label_name]

    for col in ["row_id", "investment_id", "time_id", c.params.label_name, "fold", "group_fold", "time_fold"]:
        try:
            train_df = train_df.drop(col, axis=1)
            valid_df = valid_df.drop(col, axis=1)
        except KeyError:
            pass

    train_ds = lgb.Dataset(data=train_df, label=train_labels)
    valid_ds = lgb.Dataset(data=valid_df, label=valid_labels)

    return train_ds, valid_ds


def make_dataset_general(c, train_df, valid_df):
    train_labels = train_df[c.params.label_name].values
    valid_labels = valid_df[c.params.label_name].values

    for col in ["row_id", "investment_id", "time_id", c.params.label_name, "fold", "group_fold", "time_fold"]:
        try:
            train_df = train_df.drop(col, axis=1)
            valid_df = valid_df.drop(col, axis=1)
        except KeyError:
            pass

    train_ds = train_df.values
    valid_ds = valid_df.values

    return train_ds, train_labels, valid_ds, valid_labels


def make_dataloader(c, ds, shuffle, drop_last):
    dataloader = DataLoader(
        ds,
        batch_size=c.params.batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=drop_last,
    )
    return dataloader


class BaseDataset(Dataset):
    def __init__(self, c, df, label=True):
        # self.df = df
        self.use_label = label
        if self.use_label:
            self.labels = df[c.params.label_name].values

        for col in ["row_id", "investment_id", "time_id", c.params.label_name, "fold", "group_fold", "time_fold"]:
            try:
                df = df.drop(col, axis=1)
            except KeyError:
                pass

        self.features = df.values

    def __len__(self):
        # return len(self.df)
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx]).float()
        if self.use_label:
            label = torch.tensor(self.labels[idx]).float()
            return feature, label
        return feature
