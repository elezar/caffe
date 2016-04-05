import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt
import caffe
import caffe.proto.caffe_pb2 as pb
import argparse


""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    H, W = img_shape
    Hs, Ws = tile_spacing

    # generate a matrix to store the output
    dt = X.dtype
    if output_pixel_vals:
        dt = 'uint8'
    out_array = np.zeros(out_shape, dtype=dt)

    for tile_row in xrange(tile_shape[0]):
        for tile_col in xrange(tile_shape[1]):
            if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                this_x = X[tile_row * tile_shape[1] + tile_col]
                if scale_rows_to_unit_interval:
                    # if we should scale values to be between 0 and 1
                    # do this by calling the `scale_to_unit_interval`
                    # function
                    this_img = scale_to_unit_interval(
                        this_x.reshape(img_shape))
                else:
                    this_img = this_x.reshape(img_shape)
                # add the slice to the corresponding position in the
                # output array
                c = 1
                if output_pixel_vals:
                    c = 255
                out_array[
                    tile_row * (H + Hs): tile_row * (H + Hs) + H,
                    tile_col * (W + Ws): tile_col * (W + Ws) + W
                ] = this_img * c
    return out_array


def plot(filename, layer_id, output_file=None, cmap=None):
    with open(filename) as f:
        net = pb.NetParameter.FromString(f.read())

    print net.layers

    shape = net.layer[layer_id].blobs[0].shape.dim
    W = np.array(net.layer[layer_id].blobs[0].data).reshape(shape)

    output_size = int(np.sqrt(shape[0]))
    input_size = int(np.sqrt(shape[1]))

    image_array = tile_raster_images(
        X=W,
        img_shape=(input_size, input_size),
        tile_shape=(output_size, output_size),
        tile_spacing=(1, 1)
    )

    plt.imshow(image_array, interpolation='none', cmap=cmap, aspect=None)

    if output_file is not None:
        plt.imsave(
            output_file, image_array, cmap=cmap)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    # Required argument: trace file
    parser.add_argument(
        "-f", "--file", required=True,
        help=".caffemodel file to load"
    )
    parser.add_argument(
        "--layer-id", type=int, required=True,
        help="The ID of the layer for which the filters must be generated"
    )
    parser.add_argument(
        "-o", "--output-file", default=None,
        help="The output filename to save the image as. If not specified, the \
        plot is shown"
    )
    parser.add_argument(
        "--cmap", default=None,
        help="The matplotlib colormap to use for the image")
    args = parser.parse_args()

    plot(args.file, args.layer_id, args.output_file, args.cmap)


if __name__ == "__main__":
    main()
