# Frontrow Training Module

## Installation
```{bash}
git clone https://git.brsc.local/stgeorge/fr_train.git
cd fr_train
pip install .
```

## Basic Usage
```{python}
import fr_train as fr

# Train a model with a given FLAGS file
fpath = './flags.yaml'
FLAGS = fr.import_flags()
fr.train(FLAGS)

# Restore a model and evaluate it
mpath = './saved_models/model_A
model = fr.restore_model(mpath, init_compile=True)

fr.eval(model)
```