import time
import os

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision
from torchvision import transforms

from PIL import Image
from collections import OrderedDict
import matplotlib.pyplot as plt
import utils
from utils import postp


def load_images(image_dir=None):
    """
    1) Load each image in img_paths. Use Image in the PIL library.
    2) Apply the prep transformation to each image (see the variables above)
    3) Convert each image from size C x W x H to 1 x C x W x H (same as batch size 1)
    4) Wrap each Tensor in a Variable
    5) Return a tuple of (style_image_tensor, content_image_tensor)
    """
    image_dir = image_dir or utils.image_dir
    img_paths = [image_dir + 'vangogh_starry_night.jpg', image_dir + 'Tuebingen_Neckarfront.jpg']

    raise NotImplementedError


def generate_pastiche(content_image):
    """
    Clone the content_image and return wrapped as a Variable with
    requires_grad=True
    """
    raise NotImplementedError


class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        """ Saves input variables and initializes objects

        Keyword arguments:
        target - the feature matrix of content_image
        weight - the weight applied to this loss (refer to the formula)
        """
        super(ContentLoss, self).__init__()

        raise NotImplementedError

    def forward(self, x):
        """ Calculate the content loss. Refer to the notebook for the formula.

        Keyword arguments:
        x -- a selected output layer of feeding the pastiche through the cnn
        """
        raise NotImplementedError


class GramMatrix(nn.Module):
    def forward(self, x):
        """ Calculates the batchwise Gram Matrices of x. You will want to
        divide the Gram Matrices by W*H in the end. This will help keep
        values normalized and small.

        Keyword arguments:
        x - a B x C x W x H sized tensor, it should be resized to B x C x W*H
        """

        raise NotImplementedError


class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        """ Saves input variables and initializes objects

        Keyword arguments:
        target - the Gram Matrix of an arbitrary layer of the cnn output for style_image
        weight - the weight applied to this loss (refer to the formula)
        """
        super(StyleLoss, self).__init__()

        raise NotImplementedError


    def forward(self, x):
        """Calculates the weighted style loss. Note that we are comparing STYLE,
        so you will need to find the Gram Matrix for x. You will not need to do so
        for target, since it is stated that it is already a Gram Matrix.

        Keyword arguments:
        x - features of an arbitrary cnn layer by feeding the pastiche
        """
        raise NotImplementedError


def construct_style_loss_fns(vgg_model, style_image, style_layers):
    """Constructs and returns a list of StyleLoss instances - one for each given style layer.
    See vgg.py to see how to extract the given layers from the vgg model. After you've calculated
    the targets, make sure to detach the results by calling detach(). Also make sure your output
    StyleLoss objects are in order of the style_layers. For the contents of 
    style_layers, see the style_layers object in main()

    Keyword arguments:
    vgg_model - the pretrained vgg model. See vgg.py for more details
    style_image - the style image
    style_layers - a list of layers of the cnn output we want.

    """
    raise NotImplementedError


def construct_content_loss_fns(vgg_model, content_image, content_layers):
    """Constructs and returns a list of ContentLoss instances - one for each given content layer.
    See vgg.py to see how to extract the given layers from the vgg model. After you've calculated
    the targets, make sure to detach the results by calling detach(). For info on the
    contents of content_layer, see content_layers in main()

    Keyword arguments:
    vgg_model - the pretrained vgg model. See vgg.py for more details
    content_image - the content image
    content_layers - a list of layers of the cnn output we want.

    """
    raise NotImplementedError


def main():
    """The main method for performing style transfer"""
    max_iter, show_iter = 40, 2
    style_layers = ['r11','r21','r31','r41', 'r51']
    content_layers = ['r42']
    loss_layers = style_layers + content_layers

    vgg_model = utils.load_vgg()
    # Load up all of the style, content, and pastiche Image
    # Construct the loss functions
    raise NotImplementedError

    optimizer = optim.LBFGS([pastiche])

    for itr in range(max_iter):
        def closure():

            # Implement the optimization step
            raise NotImplementedError

            if itr % show_iter == 0:
                print('Iteration: %d, loss: %f' % (itr, loss.data[0]))
            return loss
        optimizer.step(closure)

    out_img = postp(pastiche.data[0].cpu().squeeze())
    plt.imshow(out_img)
    plt.show()


if __name__ == '__main__':
    main()


