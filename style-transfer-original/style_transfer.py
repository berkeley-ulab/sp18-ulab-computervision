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

from vgg import load_vgg

image_dir = os.getcwd() + '/Images/'
model_dir = os.getcwd() + '/Models/'
img_size = 512

prep = transforms.Compose([transforms.Resize(img_size),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ])
postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                           transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                           ])
postpb = transforms.Compose([transforms.ToPILImage()])
def postp(tensor): # to clip results in the range [0,1]
    t = postpa(tensor)
    t[t>1] = 1    
    t[t<0] = 0
    img = postpb(t)
    return img

def load_images():
    """
    1) Load each image in img_paths. Use Image in the PIL library. 
    2) Apply the prep transformation to each image
    3) Convert each image from size C x W x H to 1 x C x W x H
    4) Wrap each Tensor in a Variable
    5) Return a tuple of (style_image_tensor, content_image_tensor)
    """
    img_paths = [image_dir + 'vangogh_starry_night.jpg', image_dir + 'Tuebingen_Neckarfront.jpg']
    imgs = [Image.open(path) for path in img_paths]
    imgs = [prep(img).unsqueeze(0) for img in imgs]
    imgs = [Variable(img, requires_grad=False) for img in imgs]
    return imgs

def generate_pastiche(content_image):
    """
    Clone the content_image and return wrapped as a Variable
    """
    return Variable(content_image.data.clone(), requires_grad=True)


class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        """
        Your Code Here
        """
        self.target = target
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        """ Your Code Here """
        return self.criterion(input, self.target) * self.weight


class GramMatrix(nn.Module):
    def forward(self, input):
        """
        Calculate the Gram Matrix of the input features
        The input will be of size B x C x W x H. You want to resize
        the input to B x C x W*H, and find the Gram Matrix for each batch
        """

        """ Your Code Here """
        b, c, w, h = input.size()
        features = input.view(b, c, w*h)
        G = torch.bmm(features, features.transpose(1, 2))
        return G.div_(w*h)


class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

        """ Your Code Here """

    def forward(self, input):
        """ Your Code Here """
        return self.criterion(self.gram(input), self.target) * self.weight


def construct_style_loss_fns(vgg_model, style_image, style_layers):
    style_targets = [GramMatrix()(A).detach() for A in vgg_model(style_image, style_layers)]
    style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
    return [StyleLoss(st, sw) for st, sw in zip(style_targets, style_weights)]


def construct_content_loss_fns(vgg_model, content_image, content_layers):
    content_targets = [A.detach() for A in vgg_model(content_image, content_layers)]
    content_weights = [1e0]
    return [ContentLoss(ct, cw) for ct, cw in zip(content_targets, content_weights)]


def main():
    vgg_model = load_vgg(model_dir)

    style_image, content_image = load_images()
    pastiche = generate_pastiche(content_image)

    style_layers = ['r11','r21','r31','r41', 'r51'] 
    content_layers = ['r42']
    loss_layers = style_layers + content_layers

    style_loss_fns = construct_style_loss_fns(vgg_model, style_image, style_layers) 
    content_loss_fns = construct_content_loss_fns(vgg_model, content_image, content_layers)    
    loss_fns = style_loss_fns + content_loss_fns

    max_iter, show_iter = 40, 2
    optimizer = optim.LBFGS([pastiche])
    n_iter = [0]

    while n_iter[0] <= max_iter:
        def closure():
            optimizer.zero_grad()
            out = vgg_model(pastiche, loss_layers)
            layer_losses = [loss_fn(A) for loss_fn, A in zip(loss_fns, out)]
            style_loss, content_loss = sum(layer_losses[:-1]), sum(layer_losses[-1:])
            print(style_loss.data[0], content_loss.data[0])
            loss = style_loss + content_loss
            loss.backward()
            n_iter[0] += 1
            if n_iter[0] % show_iter == 0:
                print('Iteration: %d, loss: %f' % (n_iter[0], loss.data[0]))
            return loss
        optimizer.step(closure)

    out_img = postp(pastiche.data[0].cpu().squeeze())
    plt.imshow(out_img)
    plt.show()

if __name__ == '__main__':
    main()

    