import json
from pprint import pprint
from typing import Callable

import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset

from utils import load_config


# https://github.com/c12mind/cvml


device = torch.device("mps")
class DatasetHandler:
    def __init__(self, config):
        self.config = config
        self.dataset = pd.read_csv(self.config["data_path"])
        self.dataset_properties = {}
        self.normalise_dataset()
    

    def norm_continuous_data(self, header):
        mean = self.dataset[header].mean()
        std = self.dataset[header].std()
        if std == 0:
            self.dataset[header] = 0
        else:
            self.dataset[header] = (self.dataset[header] - mean) / std
        return mean, std
    
    def create_lut(self, header):
        uniq_values = self.dataset[header].unique()
        lut = {
            k: uniq_values[k] for k in range(len(uniq_values))
        }
        inv_lut = {v: k for k, v in lut.items()}
        self.dataset[header] = self.dataset[header].map(inv_lut)
        return lut

    def normalise_dataset(self):
        for col_name, data in self.dataset.items():
            if data.dtype != "object":
                mean, std = self.norm_continuous_data(col_name)
                self.dataset_properties[col_name] = {
                    "mean": mean,
                    "std": std
                }
            else:
                lookup_table = self.create_lut(col_name)
                self.dataset_properties[col_name] = {
                    "lookup": lookup_table
                }
        
        with open("ds_stats.json", "w") as fh:
            json.dump(self.dataset_properties, fh, indent=4)
            

class AntennaDataset(Dataset):
    def __init__(self, ds_handler):
        super().__init__()
        self.dataset = ds_handler.dataset
        self.ds_stats = ds_handler.dataset_properties

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        discrete_cols = self.dataset.iloc[:, :8].iloc[index]
        cont_cols = self.dataset.iloc[:, 8:].iloc[index]
        return discrete_cols, cont_cols
    

class CollateFunction(Callable):
    def __init__(self, num_choices):
        self.handler = ds_handler

    def __call__(self, batch):
        header_names = self.dataset.columns

        discrete_list = []
        cont_list = []
        target_list = []
        for row_discrete, row_cont in batch:
            row_cont_tensor = torch.FloatTensor(row_cont[:-1]).unsqueeze(0)
            cont_list.append(row_cont_tensor)
            target_list.append(row_cont[-1])

            for header in header_names:
                header_onehot = F.one_hot()
                # iterate thru the headers and create onehot encodings for each one via 
                # F.one_hot. 

            # Concatenate the one hot encodings together across the horizontal axis to get one feature vector.
            # Append feature vector to discrete_list


        cont_tensor = torch.cat(cont_list)
        target_tensor = torch.FloatTensor(target_list)
        disc_tensor = torch.cat(discrete_list)


        return {
            "discrete_features": disc_tensor,
            "cont_features": cont_tensor,
            "targets": target_tensor,
        }

        
def load_data(config):
    ds_handler = DatasetHandler(config)
    train_df = ds_handler.dataset

    discrete_cols = train_df.iloc[:, :8]
    cont_cols = train_df.iloc[:, 8:]
    disc_header_names = train_df.columns[:8]
    discrete_list = []
    cont_list = []
    target_list = []
    cont_tensor = torch.FloatTensor(cont_cols.iloc[:, :-1].values)
    target_tensor = torch.FloatTensor(cont_cols.iloc[:, -1].values)

    discrete_cols_onehot = []
    for header in disc_header_names:
        num_choices = len(ds_handler.dataset_properties[header]["lookup"].keys())
        header_values = torch.tensor(discrete_cols[header].values)
        header_onehot = F.one_hot(header_values, num_classes=num_choices)
        discrete_cols_onehot.append(header_onehot)
    
    disc_tensor = torch.cat(discrete_cols_onehot, dim=1)
    feature_tensor = torch.cat((disc_tensor, cont_tensor), dim=1)
    target_tensor = torch.FloatTensor(target_list)
    data = {
        "features": feature_tensor,
        "targets": target_tensor,
        "ds_handler": ds_handler,
    }
    return data


if __name__ == "__main__":
    config = load_config("config.json")
    data_obj = load_data(config)
