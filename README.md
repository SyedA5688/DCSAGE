# Deep Learning-Derived Optimal Aviation Strategy to Quench Pandemics
Official PyTorch implementation of DCSAGE and DCGAT from *Deep Learning-Derived Optimal Aviation Strategy to Quench Pandemics*.

<!-- Insert Link in brackets here, e.g.: [blogpost] [arXiv] [Yannic Kilcher's video] -->

<!-- Insert figure link -->

## Training
### Getting Started
This codebase was developed using Python 3.8.12 and PyTorch 1.8.0. To reproduce the environment, install dependencies through either anaconda (preferred) or pip (just use pip install command):

```
conda env create -n DCSAGE python=3.8
conda activate DCSAGE
conda install pytorch==1.8.0 -c pytorch
pip install -r requirements.txt
```

and then set the PYTHONPATH to the base directory of the repository (important for imports to work correctly):
```
export PYTHONPATH="/path/to/base/DCSAGE/directory" 
```


### DCSAGE Training
To train a single DCSAGE model, run:

```
cd training
python train.py
```

For training multiple DCSAGE models, define the number of models (e.g. 100) in training/train_multiple_models.py and run:

```
cd training
python train_multiple_models.py
```


### DCSAGE Node Perturbation
To run the node perturbation experiment on trained DCSAGE models, first update the directory of the trained models in ./explainability/node_perturbation/node_perturb_analysis_config.json. Then run:

```
cd explainability/node_perturbation
python node_perturbation_analysis.py
```


