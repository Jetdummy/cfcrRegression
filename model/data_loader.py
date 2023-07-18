import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class PIDNDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, selected_indices, window=10, shift=5):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            selected_indices: (array) rows in data
            window: (int) number of columns in a torch
            shift: (int) how many shift to the next data sample
        """
        if selected_indices is None:
            selected_indices = [8, 9, 10, 11, 12, 13]
        # 0 Ax 1 Ay 2 Vx 3 Yawrate 4 Vy_dot 5 Steer 6 Vy_dot_RT 7 Vy_RT 8 Ax_normal 9 Ay_normal 10 Vx_normal
        # 11 Yawrate_normal 12 Vy_dot_normal 13 Steer_normal 14 Vy_dot_RTx_normal 15 Vy_RT_normal
        self.data_dir = data_dir
        self.window = window
        self.shift = shift
        self.selected_indices = selected_indices
        self.data = [] # each data in folder
        self.index = [] # each data size

        # Read and concatenate all text files
        file_list = [file for file in os.listdir(data_dir) if file.endswith('.txt')]
        for file in file_list:
            file_path = os.path.join(data_dir, file)
            data = pd.read_csv(file_path, delimiter=',')  # Modify delimiter as per your file format
            length = (len(data) - self.window) - (len(data) - self.window) % self.shift
            self.index.append(length // self.shift - 1)
            #selected_data = data.iloc[:, self.selected_indices]
            self.data.append(data)
            # self.data = pd.concat([self.data, selected_data], ignore_index=True)

    def __len__(self):
        # return size of dataset
        return sum(self.index)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            input: (Tensor) selected indices dataset
            label: (float) not yet
            data: (Tensor) all indices data
        """
        dataset_index = 0
        sum_data = 0
        # Find which dataset of the files in the folder is
        for i in range(len(self.index)):
            if idx - sum_data < self.index[i]:
                dataset_index = i
                break
            sum_data += self.index[i]

        #
        start_idx = (idx - sum_data) * self.shift
        end_idx = start_idx + self.window
        selected_data = self.data[dataset_index].iloc[start_idx:end_idx, :]
        selected_data_x = self.data[dataset_index].iloc[start_idx:end_idx, self.selected_indices]
        selected_data_y = self.data[dataset_index].iloc[end_idx+1, 6]  # 6 or 7

        return torch.tensor(selected_data_x.to_numpy()), torch.tensor(selected_data_y), torch.tensor(selected_data.to_numpy())


def fetch_dataloader(types, data_dir, params, selected_indices, window, shift):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}_datas".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(PIDNDataset(path, selected_indices, window, shift), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
            elif split == 'val':
                dl = DataLoader(PIDNDataset(path, selected_indices, window, shift), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)
            else:
                dl = DataLoader(PIDNDataset(path, selected_indices, window, shift), batch_size=1,
                                shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)
            dataloaders[split] = dl

    return dataloaders