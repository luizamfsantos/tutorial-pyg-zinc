import torch_geometric.transforms as T
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader

def get_data():
    transform = T.AddRandomWalkPE(
        walk_length=20,
        attr_name='pe',
    )
    train_dataset = ZINC(
        root='data',
        subset=True,
        split='train',
        pre_transform=transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )
    validation_dataset = ZINC(
        root='data',
        subset=True,
        split='val',
        pre_transform=transform,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=64,
        shuffle=False,
    )
    test_dataset = ZINC(
        root='data',
        subset=True,
        split='test',
        pre_transform=transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
    )
    return train_loader, validation_loader, test_loader