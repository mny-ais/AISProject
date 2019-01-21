"""Image Mover.

Moves around the images so we have one big folder of images and one super long
csv file instead of a lot of smaller folders

Authors:
    Yvan Satyawan <ys88@saturn.uni-freiburg.de>
"""
import argparse
import csv
import re
from os import listdir, mkdir, path, rename
"""
First make a new folder called "all"

Then iterate through each subdirectory of the existing recordings and move
the image to the new folder, while also making a new csv file in that folder

While it is doing this, also rename each file consecutively
image_xxxxx-cam_x.png or seg_xxxxx-cam_x.png

The csv file should be done once for every folder

This is meant to also work if the folder all already has contents. In that case
it just keeps on appending to the end of it.
"""


def move_images(parent_dir):
    """Moves all images recursively to a subfolder "all" within the parent_dir.

    Args:
        parent_dir (str): Parent directory which contains all the recordings.
    """
    count = 0
    csv_line_count = 0
    total = 0

    print("Processing {:0>5d}/{}".format(count, total))

    dir_list = []  # List of directories with images, excluding "all"

    # Create regex stuff
    p = re.compile("\d+")


    # Count total so we can see progress
    for dir in listdir(parent_dir):
        if path.isdir(path.join(parent_dir, dir)):
            if dir != "all":
                cur_dir = path.join(parent_dir, dir)
                dir_list.append(cur_dir)
                total += len(listdir(cur_dir))

    all_dir = path.join(parent_dir, 'all')


    # Check to see if the all dir already exists
    if not path.isdir(all_dir):
        # If not, make it
        mkdir(all_dir)
    else:
        # If it is, make sure the counter adds to the end of it.
        count = len(listdir(all_dir)) / 4

    # Check to see if all_dir already contains a csv file
    csv_data = []
    if path.exists(path.join(all_dir, "control_input.csv")):
        write_mode = "a"  # Append if it already exists
    else:
        write_mode = "w"  # Write new if non existent

        csv_data.append(
            ["No.",
             "Steering",
             "Throttle",
             "Brakes",
             "Gear",
             "Requested_Direction"]
        )  # Add header if non existant

    for dir in dir_list:
        # Get a 0th index for the current folder
        file_list = listdir(dir)
        num_files = len(file_list)
        image_count = 0

        for file in file_list:
            # File is currently only the file name, without the whole path
            # dir contains the whole path to the current working directory
            if file.endswith(".csv"):
                with open(path.join(dir, file), mode='r') as old_csv:
                    reader = csv.reader(old_csv)
                    line_list = list(reader)[1:]
                    for line in line_list:
                        data = line
                        data[0] = int(csv_line_count) + int(data[0])
                        csv_data.append(data)

                    csv_line_count += len(line_list)
                old_csv.close()

                with open(path.join(all_dir, "control_input.csv"),
                                    mode=write_mode, newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerows(csv_data)
                csv_file.close()
                csv_data = []
                write_mode = 'a'

            if file.endswith(".png"):
                new_name = ""

                # First extract the number values and convert them to int
                num_vals = re.findall(p, file)
                num_vals[0] = int(num_vals[0]) + int(count)
                num_vals[1] = int(num_vals[1])

                # put the max value as image_count
                image_count = max(image_count, num_vals[0])
                # Now the 2 cases of seg or image
                if file.startswith("image"):
                    new_name = "image_{:0>5d}-cam_{}.png".format(num_vals[0],
                                                                 num_vals[1])
                elif file.startswith("seg"):
                    new_name = "seg_{:0>5d}-cam_{}.png".format(num_vals[0],
                                                               num_vals[1])

                rename(path.join(dir, file), path.join(all_dir, new_name))

        count += image_count
        print("Processing {:0>5d}/{}".format(count, total))




def parse_args():
    """Parses command line arguments."""
    description = "Moves all recorded images and associated data to one folder."
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('dir', metavar='D', type=str, nargs='?',
                        help='path of the parent recordings directory.')
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    move_images(arguments.dir)
