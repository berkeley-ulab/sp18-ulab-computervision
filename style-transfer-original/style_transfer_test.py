import sys

import torch
import numpy as np

import style_transfer as st


def test_load_images():
    style, content = st.load_images()

    assert isinstance(style, torch.FloatTensor), 'style_image must be FloatTensor'
    assert isinstance(content, torch.FloatTensor), 'content_image must be FloatTensor'

    style_test = np.load('test_data/style_image_test.npy')
    content_test = np.load('test_data/content_image_test.npy')

    style_test, content_test = torch.FloatTensor(style_test), torch.FloatTensor(content_test)

    assert style_test.size() == style.size(), 'Expected: %s, Actual: %s' % (style_test.size(), style.size())
    assert content_test.size() == content.size(), 'Expected: %s, Actual: %s' % (content_test.size(), content.size())

    assert np.allclose(style_test.numpy(), style.numpy())
    assert np.allclose(content_test.numpy(), content.numpy())


def test_generate_pastiche():
    _, content = st.load_images()
    pastiche = st.generate_pastiche(content)
    assert pastiche.size() == content.size()


def test_extract_vgg_features():
    style, content = st.load_images()
    st.extract_vgg_features()

if __name__ == '__main__':
    assert len(sys.argv) > 1, 'Need method name to test'
    method_name = sys.argv[1]
    test_method_name = 'test_%s' % method_name
    assert test_method_name in globals(), 'No such method: %s' % method_name
    globals()[test_method_name]()
    print('Passed!')
