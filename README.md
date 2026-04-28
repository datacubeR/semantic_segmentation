# Semantic Segmentation Experiments

These are all the experiments conducted for the Chapter 2 of my Phd Thesis.

## Splitters

This corresponds to a CLI that helps split the datasets into train and validation sets. It moves original images and masks to a new folder structure that is compatible with the Lightning DataModules.

To run the splitter, run the following command:

```bash
uv run -m splitters.deadtrees_splitter --dataset-folder DeadTrees --image-folder dataset_rgb --mask-folder dataset_binary
```

This Example corresponds to the `DeadTrees` Dataset, but it can be easily adapted to the other datasets by changing the parameters. `DeadTrees` is the folder of the original dataset, `dataset_rgb` is the subfolder that contains the original RGB images and `dataset_binary` is the subfolder that contains the binary masks.

An additional `--test-size` parameter can be used to specify the proportion of the dataset to include in the validation set. Default is 0.2 (20%).

The splitter will create new folders called `train` and `val`, each containing `images` and `masks` subfolders. The original images and masks will be moved to the corresponding folders.

# WIP

## So far

* Notebooks to explore the different datasets.
* DataClasses to the different datasets.
* Lightning DataModules to apply dataloading parameters and transformations using Kornia.
* Experimental Lightning Module with Checkpointning and Tensorboard logging.

* So far this only runs on CPU due to memory constraints. Kernel Exploding a lot on my Laptop.

## To Do

* Find a reliable way to run experiments on GPU. 
* Modularize the Code to be able to easily test different architectures and hyperparameters.
* Start experimentation. 

> This is a highly experimental Repo yet. Do not expect anything to work out of the box. I will be updating it as I go along with the experiments.

A lot of honest work here... A long way to go yet...
