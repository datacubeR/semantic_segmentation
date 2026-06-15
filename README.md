# Semantic Segmentation Experiments

These are all the experiments conducted for the Chapter 2 of my Phd Thesis.

## Splitters Folder

Contains the code to split the datasets into train, validation and test sets. To understand how it works refer to the [README](splitters/README.md) file in the Splitters Folder.

## HuggingFace Datasets WIP

To deal with Memory Limitations there are files called `dataset_hf.py` that contain the code to convert datasets into HF Datasets using Arrow. This allows to load the datasets in a more efficient way and apply transformations on the fly using Kornia.

To do this you can use make commands such as: 

```bash
make vaihingen-hf
make potsdam-hf
make loveda-hf
make deadtrees-hf
```

This will create new folders called `{dataset_name}_HF`, containing the Arrow files for the images and masks. The original images and masks will not be moved, so you can still use them for other purposes if needed. This process is sharded to deal with memory limitations, so it might take some time to complete.


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
