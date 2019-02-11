# AISProject
AIS Project for a self-driving car

## Usage
### Command-Line
To use the our network, run the following command in command line:
```bash
python src/main.py [weight_path] -e
```
Alternatively, to train the network, the following can be used:
```bash
python src/main.py [weight_path] -t [training_data_directory] -b [batch_size] -p [number of epochs]
```
Our command line tool can unfortunately only run the standard network (i.e. with a single forward facing camera).

### GUI
We have also created a GUI for simpler usage.
Open it with:
```bash
python src/project.py
```
It also allows the selection of different variations of the networks.
The different network variations are:

  | Network           | Description
  |-------------------|-------------
  | standard          | Uses only one forward facing camera to drive
  | segmented         | Uses only the ground truth segmentation of a forward facing camera
  | seg and normal    | Uses both the image and ground truth segmentation of a forward facing camera
  | last image too    | Uses both the current and previous image from a forward facing camera
  | two cams          | Uses a forward facing and a camera angled towards the right
  | self segmentation | Uses the forward facing camera and segments the image with a network first
