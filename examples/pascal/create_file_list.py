# Create a file list for the pascal multi-label classification.
# This should be run from the $CAFFE_ROOT folder as:
# python ./examples/pascal/create_file_list.py
#
# This creates three files (each in the $CAFFE_ROOT/examples/pascal folder):
#   trainval.list.txt, train.list.txt, and val.list.txt
#
# Note that the folder $CAFFE_ROOT/data/pascal/VOC2012 must contain the
# relevant data. The dataset can be downloaded by running:
#   ./data/pascal/get_pascal.sh
# in the $CAFFE_ROOT folder.
#

import datetime
import os
import numpy as np
import sets
import xml.dom.minidom


def get_pascal_classes(index, pascal_root, ignore_list):
    """
    This code is adapted from Ross Girshick's FAST-RCNN code
    (https://github.com/rbgirshick/fast-rcnn).
    It parses the PASCAL .xml metadata files.
    See publication for further details: (http://arxiv.org/abs/1504.08083).

    Thanks Ross!

    """
    classes = ('__background__',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
    class_to_ind = dict(zip(classes, xrange(21)))
    filename = os.path.join(pascal_root, 'Annotations', index + '.xml')
    # print 'Loading: {}'.format(filename)

    def get_data_from_tag(node, tag):
        return node.getElementsByTagName(tag)[0].childNodes[0].data

    with open(filename) as f:
        data = xml.dom.minidom.parseString(f.read())

    objs = data.getElementsByTagName('object')

    gt_classes = []

    # Load  the object classes
    for obj in objs:
        cls = class_to_ind[
            str(get_data_from_tag(obj, "name")).lower().strip()]
        gt_classes.append(cls)

    return sets.Set(gt_classes) - sets.Set(ignore_list)


def get_index_list(root, phase):
    """
    Return the list of indices for the specified phase.
    """
    set_root = os.path.join(root, 'ImageSets')
    # Get list of image indexes.
    indexlist = [line.rstrip('\n') for line in open(
        os.path.join(set_root, 'Main', phase + '.txt'))]

    return indexlist


def create_label_list(labels, separator):
    return separator.join(str(l) for l in labels)


def create_list_file(root, output_path, phase,
                     label_separator, ignore_list, ignore_separator):
    """
    Create a list file for the specified phase
    """
    i_list = get_index_list(root, phase)

    image_root = os.path.join('JPEGImages')
    image_ext = '.jpg'

    filename = os.path.join(output_path, '%s.list.txt' % phase)

    if ignore_list and ignore_separator:
        ignore_string = '%s%s' % (
            ignore_separator, create_label_list(ignore_list, label_separator))
    else:
        ignore_string = ''

    with open(filename, 'w') as f:
        f.write('# %s file list generated on %s\n' %
                (phase, datetime.datetime.now()))
        for i in i_list:
            classes = get_pascal_classes(i, root, ignore_list)

            image_file = os.path.join(image_root, i + image_ext)
            label_list = create_label_list(classes, label_separator)
            line = '%s %s%s\n' % (image_file, label_list, ignore_string)
            f.write(line)


def main():
    pascal_root = 'data/pascal/VOC2012'
    output_path = 'examples/pascal'
    label_separator = ' '
    ignore_separator = ';'
    # Ignore the background label.
    ignore_list = [0]

    for phase in ['train', 'trainval', 'val']:
        print("Creating list for %s" % phase)
        train_list = create_list_file(
            pascal_root, output_path, phase, label_separator, ignore_list,
            ignore_separator)

if __name__ == "__main__":
    main()
