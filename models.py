import torch
from torch.nn import (
    Linear,
    ReLU,
    Sequential,
    Embedding,
    BatchNorm1d,
)
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn import (
    GINEConv,
    GPSConv,
    global_add_pool,
)

class RedrawProjection:
    def __init__(
        self, 
        model: torch.nn.Module,
        redraw_interval: int | None = None,
    ):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules() 
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
        else:
            self.num_last_redraw += 1

class GPS(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        pe_dim: int, # positional encoding dimension
        num_layers: int,
        attn_type: str,
        attn_kwargs: dict,
    ):
        super().__init__()
        # Embeddings
        self.node_emb = Embedding(28, channels - pe_dim)
        # Positional encoding
        self.pe_lin = Linear(20, pe_dim)
        # Normalization
        self.pe_norm = BatchNorm1d(20)
        # Edge embeddings
        self.edge_emb = Embedding(4, channels)
        # Convolutions
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(
                channels,
                GINEConv(nn),
                heads=4,
                attn_type=attn_type,
                attn_kwargs=attn_kwargs,
            )
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 1),
        )
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 
            if attn_type == 'performer' else None,
        )

    def forward(
        self,
        x: torch.Tensor,
        pe: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ):
        x_pe = self.pe_norm(pe)
        x = torch.cat((
            self.node_emb(x.squeeze(-1)),
            self.pe_lin(x_pe)),
            dim=1,
        )
        edge_attr = self.edge_emb(edge_attr)

        for conv in self.convs:
            x = conv(
                x, 
                edge_index, 
                batch, 
                edge_attr=edge_attr)
        x = global_add_pool(x, batch)
        return self.mlp(x)