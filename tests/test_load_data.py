from load_data import get_data
import pytest 

def test_get_data():
    train_loader, validation_loader, test_loader = get_data()
    assert train_loader
    assert validation_loader
    assert test_loader
    assert len(train_loader) > 0
    assert len(validation_loader) > 0
    assert len(test_loader) > 0
    assert len(train_loader.dataset) > 0
    assert len(validation_loader.dataset) > 0
    assert len(test_loader.dataset) > 0
    assert len(train_loader.dataset[0]) == 5
    assert len(validation_loader.dataset[0]) == 5
    assert len(test_loader.dataset[0]) == 5
    assert train_loader.batch_size == 32
    assert validation_loader.batch_size == 64
    assert test_loader.batch_size == 64
    assert train_loader.dataset[0].x.shape[1] == 1
    assert validation_loader.dataset[0].x.shape[1] == 1
    assert test_loader.dataset[0].x.shape[1] == 1
    assert train_loader.dataset[0].pe.shape[1] == 20
    assert validation_loader.dataset[0].pe.shape[1] == 20
    assert test_loader.dataset[0].pe.shape[1] == 20
    assert train_loader.dataset[0].edge_index.shape[1] == 64
    assert validation_loader.dataset[0].edge_index.shape[1] == 78
    assert test_loader.dataset[0].edge_index.shape[1] == 34
    assert train_loader.dataset[0].edge_attr.shape[0] == train_loader.dataset[0].edge_index.shape[1]
    assert validation_loader.dataset[0].edge_attr.shape[0] == validation_loader.dataset[0].edge_index.shape[1]
    assert test_loader.dataset[0].edge_attr.shape[0] == test_loader.dataset[0].edge_index.shape[1]