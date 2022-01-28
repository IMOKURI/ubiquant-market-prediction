import torch
from torch.utils.data import DataLoader, Dataset


def make_dataset(c, df, label=True):
    if c.params.dataset == "ump_1":
        ds = BaseDataset(c, df, label)

    else:
        raise Exception("Invalid dataset.")
    return ds


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
        # df = df.drop(["row_id", "investment_id"], axis=1)

        self.use_label = label
        if self.use_label:
            self.labels = df[c.params.label_name].values
            # df = df.drop(["time_id", c.params.label_name, "fold"], axis=1)
            df = df.drop([c.params.label_name], axis=1)

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
