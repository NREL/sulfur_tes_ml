# Sulfur TES ML
##### *Build, train, and test models with customizable features and targets*
This repository contains code originally created to develop surrogate ML models to mimic the utility of CFD simultations of Sulfur TES systems. It has been adapted to provide a comprehensive package for building, training, validating, testing, and optimizing regression models with customzable features and targets.

There is native support for the followng model types:
- Neural Network (Tensorflow with Keras)
- XGBoost
- RandomForest

## Getting Started
#### 1 - *Clone this repository*
#### 2 -  *Create the conda environment from the `envSulfurTES.yaml` file*
 - Navigate to the `sulfur_tes_ml` directory
 - Execute the following command:
```sh
conda env create -p ./envSulfurTES -f envSulfurTES.yaml
```
This will create the conda environment within a new `envSulfurTES` directory.

** *Note: You may get an error involving pip installation. You can ignore this error.*
#### 3 - *Activate the conda environment*
From the `sulfur_tes_ml` directory, run:
```
conda activate ./envSulfurTES -f envSulfurTES.yaml
```
#### 4 - *Build the `stesml` package*
From the `sulfur_tes_ml` directory, run:
```
python setup.py install
```
The `stesml` package contains the utility functions used in the Jupyter notebooks.
#### 5 - *Open Jupyter Lab*
From the `sulfur_tes_ml` directory, run:
```
jupyter lab
```
This will open Jupyter Lab in your default browser.
#### 6 - *Using Jupyter Lab*
- A detailed Jupyter Lab tutorial can be found [here](https://jupyterlab.readthedocs.io/en/stable/).
- Open the `notebooks` folder
- The following notebooks are stable:
    - ***STES_Predictions***: Use this notebook to make predictions for a charging scenario.
    - ***STES_Train_and_Validate***: Use this notebook to train and validate a model.
    - ***STES_Optuna_Studies***: Use this notebook to optimize the hyperparameters of your model with Optuna.
    - ***STES_Optuna_Visualization***: Use this notebook to see visualizations of an Optuna study.
    - ***STES_Final_Model***: Once you are satisfied with a model design, use this notebook to train, test, and save a final model.
    - ***TF_Tutorial***: Use this notebook to get a better understanding of how Tensorflow is used _"under the hood"_ of the `stesml` package

- The following notebooks are experimental:
    - ***STES_PySINDy***: This notebook uses [PySINDY](https://pysindy.readthedocs.io/en/latest/) to learn governing equations from data.
    - ***STES_LSTM***: This notebook implements a Long Short-Term Memory recurrent neural network model.
    - ***STES_PINN***: This notebook implements a physics-informed neural network with a custom loss function.
    - ***Fluent_Out_to_CSV***: This notebook takes data from a FLuent .out file, calculates secondary properties, and outputs data to a .csv file
#### 7 - *Using PySINDy*
If you want too use PySINDy to learn the governing equations from your dataset, perform the following steps:
- Ensure the `python` you are using is local to your conda environment by running:
```
which python
```
The response should end with `sulfur_tes_ml/envSulfurTES/bin/python`. If it doesn't there may be an issue with your `$PATH` variable. Read [this article](https://towardsdatascience.com/python-the-system-path-and-how-conda-and-pyenv-manipulate-it-234f8e8bbc3e) to get a better understanding of this issue and how to resolve it.
- Ensure the `pip` you are using is local to your conda environment by running:
```
which pip
```
The response should end with `sulfur_tes_ml/envSulfurTES/bin/pip`. If it doesn't, install `pip` to the conda environment with:
```
conda install pip
```
- From the `sulfur_tes_ml` directory, run:
```
python -m pip install pysindy
```
This process ensures you are installing the PySINDy package locally within the conda environment.

#### 8 - *Deactivate the conda environment*
To deactivate the environment, run:
```
conda deactivate
```
