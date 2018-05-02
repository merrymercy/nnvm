"""
Symbol of super resolution

Shi, Wenzhe, et al.
"Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network."
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.
"""

from .. import symbol as sym
from . utils import create_workload

def get_symbol(image_shape, upscale_factor, **kwargs):
    data = sym.Variable(name='data')

    net = sym.conv2d(data, kernel_size=(5, 5), strides=(1, 1), padding=(2, 2),
                     channels=64, name='conv1')
    net = sym.relu(net, name='relu1')
    net = sym.conv2d(net, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1),
                     channels=64, name='conv2')
    net = sym.relu(net, name='relu2')
    net = sym.conv2d(net, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1),
                     channels=32, name='conv3')
    net = sym.relu(net, name='relu3')
    net = sym.conv2d(net, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1),
                     channels=upscale_factor ** 2, name='conv4')

    net = sym.reshape(net, shape=(-1, image_shape[0], upscale_factor, upscale_factor,
                                  image_shape[1], image_shape[2]))
    net = sym.transpose(net, axes=[0, 1, 4, 2, 5, 3])
    net = sym.reshape(net, shape=(-1, image_shape[0], image_shape[1] * upscale_factor,
                                  image_shape[2] * upscale_factor))

    return net


def get_workload(batch_size, upscale_factor=3, image_shape=(1, 224, 224), dtype="float32"):
    """Get benchmark workload for a simple multilayer perceptron

    Parameters
    ----------
    batch_size : int
        The batch size used in the model
    upscale_factor: int
        The upscale factor
    image_shape : tuple, optional
        The input image shape
    dtype : str, optional
        The data type

    Returns
    -------
    net : nnvm.symbol
        The computational graph
    params : dict of str to NDArray
        The parameters.
    """
    net = get_symbol(image_shape=image_shape, upscale_factor=upscale_factor)
    return create_workload(net, batch_size, image_shape, dtype)


