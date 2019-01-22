"""Image Mover.

Rename the images properly according to their command. Images should now have a
prefix "left-", "right-", or "forward-" appended.

Authors:
    Yvan Satyawan <ys88@saturn.uni-freiburg.de>
"""
"""
First read the csv.
Then turn it into a dict with the image number as the key and the command as the
value.
Then iterate through every file, renaming it according to its number value.
"""
import argparse
import csv
import re
from os import listdir, path, rename


def rename_images(dir):
    """Renames images according to command.

    Args:
        dir (str): The directory to execute the command.
    """
    command_data = {}

    # Compile regex pattern
    p = re.compile("\d+")

    # Read the csv data
    with open (path.join(dir, "control_input.csv"), mode='r') as csv_file:
        reader = csv.reader(csv_file)
        line_list = list(reader)[1:]  # Put all the lines into a list

        for line in line_list:
            command_data[int(line[0])] = int(line[5])

    csv_file.close()

    for file in listdir(dir):
        if file.endswith(".png"):
            # Get the number value
            num_val = re.search(p, file)
            num_val = num_val[0].lstrip("0")
            if num_val == "":
                num_val = "0"
            num_val = int(num_val)
            print(num_val)

            # Compare the number value to the dict
            try:
                curr_command = command_data[num_val + 1]
                # Add the value to the new name
                if curr_command == -1:
                    new_name = "left-"
                elif curr_command == 0:
                    new_name = "forward-"
                else:
                    new_name = "right-"

                new_name = new_name + file
                rename(path.join(dir, file), path.join(dir, new_name))
            except KeyError:
                print("No key for: {}".format(num_val + 1))
                prompt = input("Skip? [y or n]")
                if prompt == "y":
                    continue
                else:
                    raise KeyError





def parse_args():
    """Parses command line arguments."""
    description = "Renames files according to high-level command."
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('dir', metavar='D', type=str, nargs=1,
                        help='path of the directory to execute the operation')
    return parser.parse_args()


if __name__ == "__main__":
    rename_images(parse_args().dir[0])
