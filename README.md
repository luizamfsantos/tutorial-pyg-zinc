# PyG Graph Transformers Tutorial

Disclaimer: I did not create this tutorial. You can find the original tutorial [here](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/graph_transformer.html).

In this tutorial, we implemented Graph Transformers on the ZINC dataset. We used the `torch_geometric` library to implement the Graph Transformer model. The ZINC dataset contains 250,000 molecular graphs. Each graph represents a molecule, and each node represents an atom. The task is to predict the solubility of a molecule.

My contributions:
- Added some pytest tests to the code
- Removed the global variables and added them as arguments to the functions
- Added type hints to the functions
- Added arguments to the `main` function to make it more flexible
- Added logging to the code
- Added a `requirements.txt` file
- Separated models and load data functions into different files
- Added a `README.md` file with instructions on how to run the code

## How to run the code

### Option 1: Run the code locally
*I only recommend this option if you have a powerful machine with a GPU.*

1. Clone the repository
```bash
git clone https://github.com/luizamfsantos/tutorial-pyg-zinc.git
cd tutorial-pyg-zinc
```

2. Install the requirements:
(recommended to create a virtual environment) 
```bash
pip install -r requirements.txt
```

3. Run the code with 1 epoch
This will test if the code is running correctly. It will
also give you an idea of how long it will take to run the code.
```bash
python main.py --epoch 1 -d
```

4. Run the code
```bash
python main.py
```

### Option 2: Run the code on Google Colab

1. Create a new notebook on Google Colab
2. Change the runtime to GPU
3. Clone the repository
```python
!git clone https://github.com/luizamfsantos/tutorial-pyg-zinc.git
```
4. Install the requirements
```python
!pip install -r tutorial-pyg-zinc/requirements.txt
import sys
sys.path.append('tutorial-pyg-zinc')
```
5. Optional: load data
This will take a while to run for the first time.
```python
from load_data import get_data
get_data()
```
6. Run the code with 1 epoch.
This will test if the code is running correctly. It will
also give you an idea of how long it will take to run the code.
```python
sys.argv = ['', '--epoch', '1', '--debug']
from main import main
main()
```
7. Run the code
```python
sys.argv = ['']
main()
```