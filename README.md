# Spring 2018 ULab Computer Vision

Please fork this repository.

## Running on your own computer
1. Fork this github repository (make sure git is setup!)
1. `git clone [your forked repo url]`
1. `cd sp18-ulab-computervision`
1. `git remote add ulab https://github.com/berkeley-ulab/sp18-ulab-computervision.git`
1. Setup Conda via [this link](https://conda.io/miniconda.html)
1. `cd setup` and execute `bash setup.sh`. 
1. Run `source activate ulab`.
1. Run `jupyter lab` in this repository.


### Updating Your Forked Repository
1. ```git pull ulab master```

### Navigating Jupyter Notebooks
This link has a quick start on [Keyboard navigation shortcuts](http://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Notebook%20Basics.html#Keyboard-Navigation).

## Curriculum
* Numpy
    * Manipulating matrices and vectors
    * Indexing
    * Broadcasting
* Matplotlib
    * Plotting images
    * Line/scatter plots
* Neural Networks/Deep Learning
    * What is a neural network?
    * Building out your own
* PyTorch
    * Creating and manipulating Tensors - very similar to Numpy
* PyTorch Details: Automatic Differentitation
    * Calculating Gradients
* (In Progress) Conv Neural Networks
    * ...
* Gradient Descent
    * Understanding Loss Functions
    * Understanding Loss functions for Style Transfer
    * Understanding Gradient Descent
* Machine Learning (basics)
    * Data - how data is stored and accessed - with particular focus on images
    * Model
    * Training/Testing
    * Train a model on MNIST

## Order of notebooks
0. NeuralStyleTransfer (demo)
1. numpy_matplotlib
2. Neural Networks
3. pytorch
4. autograd_tutorial
5. (In progress) Convolutional Neural Networks
6. Gradient Descent
