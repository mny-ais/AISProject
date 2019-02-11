# -*- coding: utf-8 -*-
"""Image Resizer.

Resizes images into into a 320 x 64 pixel image.

Authors:
    Yvan Satyawan <ys88@saturn.uni-freiburg.de>
"""
import argparse
from PIL import Image
from resizeimage import resizeimage as imageresize
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
            cover.save(image_path)


def batch_resize(dir_path):
    """Batch resize all images in the path in the argument"""
    counter = 0
    for filename in listdir(dir_path):
        if filename.endswith(".png"):
            image = path.join(dir_path, filename)
            resize_image(image)
            counter += 1


def batch_folder(dir_path):
    """Batch resize all subfolders in the path."""
    # First get all files
    total = 0
    for dir in listdir(dir_path):
        if path.isdir(path.join(dir_path, dir)):
            total += len(listdir(path.join(dir_path, dir)))

    counter = 0  # reset counter to 0
    for dir in listdir(dir_path):
        if path.isdir(path.join(dir_path, dir)):
            for file in listdir(path.join(dir_path, dir)):
                if file.endswith(".png"):
                    image = path.join(dir_path, dir, file)
                    resize_image(image)
                    counter += 1
                    print("Resized: {0} of {1}".format(counter, total))


def main(arguments):
    """Main function that runs everything."""
    batch_folder(arguments.dir[0])


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
