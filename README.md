# Semantic Segmentation Experiments

These are all the experiments conducted for the Chapter 2 of my Phd Thesis.

In order to reproduce this you need `uv`. You need can install it by running: 

bash
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If using Windows, refer to the [uv Installation Guide](https://docs.astral.sh/uv/getting-started/installation/).

## Splitters Folder

Contains the code to split the datasets into train, validation and test sets. To understand how it works refer to the [README](splitters/README.md) file in the `splitters` Folder.

## Config Files Folder

Contains the YAML files used to configure each experiment. The files are organized into v* files, with each file corresponding to a different experiment version. For details on the configuration structure and how to use these files, refer to the [README](config_files/README.md) in the `config_files` folder.

## src Folder 

This contains the code to Create Hugging Face Datasets and for model Training. More details of this code can be found in the [README](src/README.md) file in the `src` Folder.

> This is a highly experimental Repo yet. Do not expect anything to work out of the box. I will be updating it as I go along with the experiments.

## A lot of honest work here... A long way to go yet...
