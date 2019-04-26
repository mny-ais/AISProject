# AISProject
This is our bachelor project at the Chair for Autonomous Intelligent Systems at the University of Freiburg for a self-driving car based on the paper "End-to-End Driving via Conditional Imitation Learning" by Codevilla et al. Here we implement the architecture given in the paper and attempt to create variations and improvements, including a version which first attempts semantically segment the image first.

This architecture is designed to run a simulated vehicle within Unreal Engine with the AirSim plugin.

[Check out our short demo here](https://drive.google.com/file/d/111I3EffTU_k3ina_KeWqAyrWH6vVl88L/view?usp=sharing)

## Network variations
The different network variations are:

  | Network           | Description
  |-------------------|-------------
  | standard          | Uses only one forward facing camera to drive
  | segmented         | Uses only the ground truth segmentation of a forward facing camera
  | seg and normal    | Uses both the image and ground truth segmentation of a forward facing camera
  | last image too    | Uses both the current and previous image from a forward facing camera
  | two cams          | Uses a forward facing and a camera angled towards the right
  | self segmentation | Uses the forward facing camera and segments the image with a network first

## Results
Our architecture performs well and generalizes to new unseen environments quite well. The best performing variation is `segmented`, although we believe with more training, the `self segmentation` variation can perform equally well. We were unable to train the `self segmentation` network fully due to time constraints.

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
