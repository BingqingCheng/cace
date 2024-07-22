# Description: Utility functions for saving and loading datasets

#import h5py
import numpy as np
import torch

__all__ = ['tensor_to_numpy', 'numpy_to_tensor', 'save_dataset', 'load_dataset']

# Function to convert tensors to numpy arrays
def tensor_to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.numpy()
    elif isinstance(data, dict):
        return {k: tensor_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [tensor_to_numpy(item) for item in data]
    else:
        return data

# Function to convert numpy arrays back to tensors
def numpy_to_tensor(data):
    if isinstance(data, np.ndarray):
        return torch.tensor(data)
    elif isinstance(data, dict):
        return {k: numpy_to_tensor(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [numpy_to_tensor(item) for item in data]
    else:
        return data

# Function to save the dataset to an HDF5 file
def save_dataset(data, filename, shuffle=False):
    import h5py
    if shuffle:
        index = np.random.permutation(len(data)) # Shuffle the data
    else:
        index = np.arange(len(data))
    with h5py.File(filename, 'w') as f:
        for i, index_now in enumerate(index):
            item = data[index_now]
            grp = f.create_group(str(i))
            serializable_item = tensor_to_numpy(item)
            for k, v in serializable_item.items():
                grp.create_dataset(k, data=v)
    print(f"Saved dataset with {len(data)} records to {filename}")

def load_dataset(filename):
    import h5py
    all_data = []
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            grp = f[key]
            item = {}
            for k in grp.keys():
                data = grp[k]
                if data.shape == ():  # Check if the data is a scalar
                    item[k] = torch.tensor(data[()])  # Access scalar value
                else:
                    item[k] = torch.tensor(data[:])  # Access array value
            all_data.append(numpy_to_tensor(item))
    print(f"Loaded dataset with {len(all_data)} records from {filename}")
    return all_data
