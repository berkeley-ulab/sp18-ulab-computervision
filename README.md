# Spring 2018 ULab Computer Vision

Please fork this repository.

## Running on your own computer
1. Fork this github repository (make sure git is setup!)
1. `git clone [repo url]`
1. `git remote add ulab https://github.com/berkeley-ulab/sp18-ulab-computervision.git`
1. Setup Conda via (this link)[https://conda.io/miniconda.html]
1. `cd setup` and execute `setup.sh`. 
1. Run `source activate ulab`.
1. Run `jupyter lab` in this repository.


### Updating Your Forked Repository
1. ```git pull ulab master```

### Navigating Jupyter Notebooks
This link has a quick start on [Keyboard navigation shortcuts](http://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Notebook%20Basics.html#Keyboard-Navigation).

## Curriculum Agenda
* Numpy
    * Manipulating matrices and vectors
    * Indexing
    * Broadcasting
* Matplotlib
    * Plotting images
    * Line/scatter plots
* PyTorch
    * Creating and manipulating Tensors - very similar to Numpy
    * nn.Linear
    * Variable and autograd
    * Learn how to train a linear model and visualize results
* Machine Learning (basics)
    * Data - how data is stored and accessed - with particular focus on images
    * Model
    * Training/Testing
    * Train a model on MNIST

## Order of notebooks
0. NeuralStyleTransfer (demo)
1. numpy_matplotlib
2. pytorch
3. autograd_tutorial
4. introml