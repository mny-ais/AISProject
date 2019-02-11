"""Data Fixer.

The purpose of this script is to fix and clean up data in the old recordings.
It does this be renaming each image so that they are now sequential, and by
cleaning up the csv file.

This is a purpose built script and can only be used in this specific scenario
and can not be used for other folders.
"""

import csv
import os

dir_path = "C:\\Users\\y_sat\\Offline Docs\\ais_processed\\all-old-track"

# First get the data from the csv
csv_file = open(os.path.join(dir_path, "control_input.csv"), mode='r')
reader = csv.reader(csv_file)
csv_data = list(reader)
csv_file.close()

header = csv_data[0]
del csv_data[0]

non_existing_images_cam_0 = []
non_existing_images_cam_1 = []

# Then figure out all images that don't exist.
for line in csv_data:

    # Make the file names first
    cam_0_file = "image_{:0>5d}-cam_0.png".format(int(line[0]))
    cam_1_file = "image_{:0>5d}-cam_1.png".format(int(line[0]))

    if not os.path.isfile(os.path.join(dir_path, cam_0_file)):
        non_existing_images_cam_0.append(int(line[0]))
    if not os.path.isfile(os.path.join(dir_path, cam_1_file)):
        non_existing_images_cam_1.append(int(line[0]))

differences = sorted(list(set(non_existing_images_cam_1)
                          - set(non_existing_images_cam_0)))
non_existing_images = non_existing_images_cam_0 + differences

non_existing_images = sorted(non_existing_images)

print(non_existing_images)
new_csv_data = [header]

# # Delete unnecessary csv entries
# for line in csv_data:
#     if int(line[0]) not in non_existing_images:
#         new_csv_data.append(line)
#
# # Write csv file
# csv_file = open(os.path.join(dir_path, "control_input.csv"), mode='w')
# writer = csv.writer(csv_file)
# writer.writerows(new_csv_data)
# csv_file.close()
#
# # Delete unnecessary images with just one camera value
# for index in differences:
#     # Make the file names first
#     cam_0_file = "image_{:0>5d}-cam_0.png".format(index)
#     cam_0_file = os.path.join(dir_path, cam_0_file)
#     cam_1_file = "image_{:0>5d}-cam_1.png".format(index)
#     cam_1_file = os.path.join(dir_path, cam_1_file)
#
#     if os.path.isfile(cam_0_file):
#         os.remove(cam_0_file)
#     if os.path.isfile(cam_1_file):
#         os.remove(cam_1_file)
