import json
from pprint import pprint

import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset


# https://github.com/c12mind/cvml


device = torch.device("mps")
class DatasetHandler:
    def __init__(self, data_path):
        self.dataset = pd.read_csv(data_path)
        self.dataset_properties = {}
    

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
    

class CollateFunction(callable):
    def __init__(self, ds_handler):
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
                num_choices = len(self.handler.dataset_properties[header]["lookup"].keys())
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

handler = DatasetHandler("data.csv")
handler.normalise_dataset()
# antenna_dataset = AntennaDataset(handler)

