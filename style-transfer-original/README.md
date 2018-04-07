# Style Transfer
Now, we should have all of the tool and concepts we need to implement style transfer!

## Brief Overview

1. Pick a content image and a style image
2. Intialize the pastiche (the resulting image with the content of the content image and the style of the style image) - usually random noise
3. Run the content and style image through a pretrained VGG model
    - For the content image, extract the convolutional features from ONE layer
    - For the style image, extract the convolutional features from several layers - this will help with encapsulating global and local features of the style image
4. For each iteration of gradient descent
    - Run the pastiche through the VGG model and extrat features from specified layers
    - Construct the loss, total loss = content loss + style loss
    - Update the pastiche to minimize loss
    - Repeat

Everything you will need is in this directory, as well as the methods you will complete.

## Part One: Initial Steps

Implement `style_transfer.load_images`

Implement `style_transfer.generate_pastiche`

## Part Two: Content Loss

Remember that `total_loss = content_loss + style_loss`. In this part, we will focus on implementing the `ContentLoss` class in `style_transfer.py`. We compute the content loss by taking the mean-squared error between the feature map of the pastiche and the feature map of the content image. We choose a specific layer 'r42' of the vgg net and only use that layer to compute the MSE. It may help to use `nn.MSELoss` when implementing `ContentLoss`.

## Part Three: Gram Matrix + Style Loss


## Part Four: Optimization