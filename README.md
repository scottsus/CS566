# CSCI-566 Assignment 1

## Installation instructions
Working on the assignment in a virtual environment (`venv` or `conda`) is highly encouraged.
Please use Python `3.10`.

### venv

Below are instructions for `venv` based virtual environments
```shell
pip3 install virtualenv # If you didn't install it

# replace <your_virtual_env> with the virtual env name you want
virtualenv -p $(which python3) <your_virtual_env>
source <your_virtual_env>/bin/activate

# install dependencies
pip3 install -r requirements.txt

# work on the assignment
```

### IPython Notebook
To start working on the assignment, simply run the following command to start an ipython kernel.
```shell
# add your virtual environment to jupyter notebook (replace <your_virtual_env> with the virtual env name that you created in the previous step)
source your_virtual_env/bin/activate
python -m ipykernel install --user --name=<your_virtual_env>

# port is only needed if you want to work on more than one notebooks
jupyter notebook --port=your_port_number
```
and then work on each problem with their corresponding `.ipynb` notebooks.
Check the python environment you are using on the top right corner.
If the name of environment doesn't match, change it to your virtual environment in "Kernel>Change kernel".

## Problems
In each of the notebook file, we indicate `TODO` or `Your Code` for you to fill in with your implementation.
Majority of implementation in problem 2 will also be required under `lib` with specified tags.

### Problem 1: Decision Trees and Random Forests
The IPython Notebook `problem_1_DT.ipynb` will walk you through implementing the basics of decision trees and random forests.

### Problem 2: Basics of Neural Networks
The IPython Notebook `problem_2_NN.ipynb` will walk you through implementing the basics of neural networks.

## How to submit

After completing the required code, run all the code cells in both the notebooks and make sure the outputs are visible.

Create a zip file with the following files in it:
- `problem_1_DT.ipynb`
- `problem_2_NN.ipynb`
- `lib/`
  - `mlp/`
    - `fully_conn.py`
    - `layer_utils.py`
    - `train.py`
  - `grad_check.py`
  - `optim.py`

## Questions?
If you have any question or find a bug in this assignment (or even any suggestions), we are more than welcome to assist through Piazza (post questions under folder `hw1`).

## FAQ

- **What is a good starting learning rate?**\
There is a good article: https://medium.com/octavian-ai/which-optimizer-and-learning-rate-should-i-use-for-deep-learning-5acb418f9b2

- **I experimented with the hyperparameters and tried many different combinations, which ones should I report?**\
The usual rule of thumb is to report results with the best hyperparameters you found.

- **Am I allowed to change code outside the TODO blocks?**\
Unless specified otherwise please do not change any code outside TODO blocks. **Doing so may negatively affect your grade**.

- **My %reload_ext autoreload command does not work, how to fix it?**\
This has been observed in the past and, whenever it was a problem, could be fixed by downgrading IPython to version 7.5.0: `pip3 install ipython==7.5`.

- **The zip file to submit is too large**\
Make sure you do not include your virtual environment, checkpoints, or datasets.

- **General debugging tips**
1. Make sure your implementations matches the specified function descriptions perfectly.
2. Put print statements at various places inside your implementation code to make sure every module is working as it should (but please remove any additional print statements for submission).