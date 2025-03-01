import pytest
import torch 
from torch_geometric.data import Data, Batch
from models import GPS, RedrawProjection

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.ModuleList()

    def modules(self):
        return self.attn

def test_redraw_projection():
    model = DummyModel()
    redraw = RedrawProjection(model, redraw_interval=2)
    redraw.redraw_projections()
    assert redraw.num_last_redraw == 1, "Should increment by 1"
    redraw.redraw_projections()
    redraw.redraw_projections()
    assert redraw.num_last_redraw == 0, "Should reset after reaching interval"

def test_gps_forward():
    channels = 16
    pe_dim = 4
    num_layers = 2
    attn_type = "performer"
    attn_kwargs = {}

    model = GPS(channels, pe_dim, num_layers, attn_type, attn_kwargs)
    assert model
    model.eval()

    num_nodes = 10
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    edge_attr = torch.randint(0, 4, (edge_index.shape[1],))
    x = torch.randint(0, 28, (num_nodes, 1))
    pe = torch.randn(num_nodes, 20)
    batch = torch.zeros(num_nodes, dtype=torch.long)

    out = model(x, pe, edge_index, edge_attr, batch)
    assert out.shape == (1, 1), "Global pooling should return batch-size outputs"