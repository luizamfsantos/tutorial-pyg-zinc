import argparse
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import GPS
from load_data import get_data
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s [[%(filename)s:%(lineno)d]]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def timeit(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug(f'{func.__name__} took {end - start:.2f} seconds')
        return result
    return wrapper

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--channels',
        type=int,
        default=64,
        help='Number of channels in the model'
    )
    parser.add_argument(
        '--pe_dim',
        type=int,
        default=8,
        help='Dimension of positional encoding'
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=10,
        help='Number of layers in the model'
    )
    parser.add_argument(
        '--attn_type',
        type=str,
        default='performer',
        help='Type of attention mechanism'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs to train the model'
    )
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Enable debug mode'
    )

    return parser.parse_args()

@timeit
def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        model.redraw_projection.redraw_projections()
        out = model(
            data.x,
            data.pe,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        loss = (out.squeeze() - data.y).abs().mean()
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
):
    model.eval()

    total_error = 0
    for data in test_loader:
        data = data.to(device)
        out = model(
            data.x,
            data.pe,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        total_error += (
            out.squeeze() - data.y
        ).abs().sum().item()
    return total_error / len(test_loader.dataset)

@timeit
def main():
    args = parse_arguments()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attn_kwargs = {'dropout': 0.5}

    model = GPS(
        channels=args.channels, 
        pe_dim=args.pe_dim, 
        num_layers=args.num_layers, 
        attn_type=args.attn_type,
        attn_kwargs=attn_kwargs,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=20,
        min_lr=1e-6,
    )

    train_loader, validation_loader, test_loader = get_data()

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, optimizer, train_loader, device)
        val_mae = test(model, validation_loader, device)
        scheduler.step(val_mae)
        logger.info(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val MAE: {val_mae:.4f}')
        
    test_mae = test(model, test_loader, device)
    logger.info(f'Test MAE: {test_mae:.4f}')

if __name__ == '__main__':
    main()