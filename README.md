# Deep Learning-Derived Optimal Aviation Strategy to Quench Pandemics
Official PyTorch implementation of DCSAGE and DCGAT from *Deep Learning-Derived Optimal Aviation Strategy to Quench Pandemics*.

<!-- Insert Link in brackets here, e.g.: [blogpost] [arXiv] [Yannic Kilcher's video] -->

<!-- Insert figure link -->

## Training
### Getting Started
This codebase was developed using Python 3.8.12 and PyTorch 1.8.0. To reproduce the environment, install dependencies through either anaconda or pip:

```
conda env create -f environment.yml
```

or

```
pip install requirements.txt
```


### DCSAGE Training
To train a single DCSAGE model, run:

```
cd training
python train.py
```


Note: For directory imports to work correctly, be sure to set your current 
working directory to the top directory in this repository.
