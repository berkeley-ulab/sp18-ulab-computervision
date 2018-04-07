import sys

import torch
from torch.autograd import Variable
import numpy as np

import style_transfer as st
from vgg import *


def test_load_images():
    style, content = st.load_images()

    assert isinstance(style, Variable), 'style_image must be of type Variable'
    assert isinstance(content, Variable), 'content_image must be of typeVariable'

    style_test = np.load('test_data/style_image_test.npy')
    content_test = np.load('test_data/content_image_test.npy')

    style_test, content_test = torch.FloatTensor(style_test), torch.FloatTensor(content_test)
    style, content = style.data, content.data

    assert style_test.size() == style.size(), 'Expected: %s, Actual: %s' % (style_test.size(), style.size())
    assert content_test.size() == content.size(), 'Expected: %s, Actual: %s' % (content_test.size(), content.size())

    assert np.allclose(style_test.numpy(), style.numpy())
    assert np.allclose(content_test.numpy(), content.numpy())


def test_generate_pastiche():
    _, content = st.load_images()
    pastiche = st.generate_pastiche(content)

    assert isinstance(pastiche, Variable), 'pastice must be of type Variable'
    assert pastiche.size() == content.size()
    assert np.allclose(content.data.numpy(), pastiche.data.numpy())
    assert content is not pastiche
    assert content.data is not pastiche.data


def test_ContentLoss():
    torch.manual_seed(0)
    target, weight = Variable(torch.randn(1, 32, 128, 128)), 0.75
    input = Variable(torch.randn(1, 32, 128, 128))
    loss = st.ContentLoss(target, weight)(input)

    actual, expected = loss.data[0], 1.4979565143585205
    assert np.abs(actual - expected) < 0.01, 'Expected: %s, Actual: %s' % (expected, actual) 


def test_GramMatrix():
    torch.manual_seed(0)
    input = Variable(torch.randn(1, 32, 128, 256))
    G = st.GramMatrix()(input).data
    G_exp = torch.FloatTensor(np.load('test_data/test_GramMatrix.npy'))

    assert G.size() == G_exp.size(), 'Expected: %s, Actual: %s' % (G_exp.size(), G.size())
    assert np.allclose(G.numpy(), G_exp.numpy())


def test_StyleLoss():
    torch.manual_seed(0)
    target, weight = Variable(torch.randn(1, 32, 32)), 0.75
    input = Variable(torch.randn(1, 32, 128, 128))
    loss = st.StyleLoss(target, weight)(input)

    actual, expected = loss.data[0], 0.8064331412315369
    assert np.allclose(actual, expected), 'Expected: %s, Actual: %s' % (expected, actual)


def test_construct_style_loss_fns():
    vgg_model = st.load_vgg()
    style_image, content_image = st.load_images()
    pastiche = st.generate_pastiche(content_image)
    style_layers = ['r11','r21','r31','r41', 'r51'] 

    out = vgg_model(pastiche, style_layers)
    loss_fns = st.construct_style_loss_fns(vgg_model, style_image, style_layers)
    assert len(loss_fns) == len(style_layers)
    assert all([isinstance(loss_fn, st.StyleLoss) for loss_fn in loss_fns])

    losses = [loss_fn(A).data[0] for loss_fn, A in zip(loss_fns, out)]
    expected = [95157.6953125, 8318182.5, 4280054.5, 213536288.0, 26124.064453125]
    assert np.allclose(losses, expected), 'Expected: %s, Actual: %s' % (expected, losses)


def test_construct_content_loss_fns():
    torch.manual_seed(0)
    vgg_model = st.load_vgg()
    style_image, content_image = st.load_images()
    pastiche = st.generate_pastiche(content_image) + Variable(torch.randn(*content_image.size()))
    content_layers = ['r42']

    out = vgg_model(pastiche, content_layers)
    loss_fns = st.construct_content_loss_fns(vgg_model, content_image, content_layers)
    assert len(loss_fns) == len(content_layers)
    assert all([isinstance(loss_fn, st.ContentLoss) for loss_fn in loss_fns])

    losses = [loss_fn(A).data[0] for loss_fn, A in zip(loss_fns, out)]
    expected = [324.38946533203125]
    assert np.allclose(losses, expected), 'Expected: %s, Actual: %s' % (expected, losses)

TEST_METHODS = ['load_images', 'generate_pastiche', 'ContentLoss', 'GramMatrix', 'StyleLoss', 'construct_style_loss_fns', 'construct_content_loss_fns']

if __name__ == '__main__':
    assert len(sys.argv) > 1, 'Need method name to test'
    method_name = sys.argv[1]

    if method_name == 'all':
        for method in TEST_METHODS:
            print('Testing %s...' % method, end='')
            globals()['test_%s' % method]()
            print(' Passed')
    else:
        test_method_name = 'test_%s' % method_name
        assert test_method_name in globals(), 'No such method: %s' % method_name
        globals()[test_method_name]()
        print('%s: Passed' % method_name)
