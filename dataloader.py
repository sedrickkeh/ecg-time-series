import torch
from torch.utils.data import DataLoader
import pandas as pd
import pyts
import pyts.image


def get_transformed_datasets(X, y, transform):
    if transform=="none":
        pass
    elif transform=="gaf":
        gaftransformer = pyts.image.GramianAngularField()
        X = gaftransformer.fit_transform(X)
    elif transform=="mtf":
        mtftransformer = pyts.image.MarkovTransitionField(n_bins=5)
        X = mtftransformer.fit_transform(X)
    else:
        assert(transform in ["none", "gaf", "mtf"])

    return torch.utils.data.TensorDataset(torch.from_numpy(X).float(),
                                                   torch.from_numpy(y).long(),)

def get_dataloaders(train_path, test_path, batch_size, image_transform, val_split_factor=0.2):
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    train_data = train_df.to_numpy()
    test_data = test_df.to_numpy()

    train_X, train_y = train_data[:, :-1], train_data[:, -1]
    test_X, test_y = test_data[:, :-1], test_data[:, -1]

    train_dataset = get_transformed_datasets(train_X, train_y, image_transform)
    test_dataset = get_transformed_datasets(test_X, test_y, image_transform)

    train_len = train_data.shape[0]
    val_len = int(train_len * val_split_factor)
    train_len -= val_len

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_len, val_len])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader
