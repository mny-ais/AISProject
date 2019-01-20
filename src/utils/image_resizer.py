# -*- coding: utf-8 -*-
"""Image Resizer.

Resizes images into into a 320 x 64 pixel image.

Authors:
    Yvan Satyawan <ys88@saturn.uni-freiburg.de>
"""
import argparse
from PIL import Image
from imageresize import imageresize
from os import path
from os import listdir


def parse_args():
    """Parses terminal arguments"""
    description = "Resize all images in the path in the argument to 320 x 64."
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('dir', metavar='D', type=str, nargs=1,
                        help="directory of the images to be resized.")

    arguments = parser.parse_args()

    return arguments

def resize_image(image_path):
    """Resize images in the path in the argument to 320 x 64."""
    with open(image_path, 'r+b') as f:
        with Image.open(f) as image:
            cover = imageresize.resize_cover(image, [320, 64])
            new_save_path = path.join(path.dirname(image_path), "resized",
                                      path.basename(image_path))
            cover.save(new_save_path)

def batch_resize(dir_path):
    """Batch resize all images in the path in the argument"""
    for filename in listdir(dir_path):
        if filename.endswith(".png"):
            resize_image(filename)

if __name__ == "__main__":
    batch_resize(parse_args())
