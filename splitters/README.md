# Splitter

This folder contains code to split the dataset into training, validation and test sets. 

To split the dataset, use one of the following commands:

```bash
make vaihingen-split
make potsdam-split
make loveda-split
```

Each one will run the corresponding command. Under the hood, it will run the `dataset_splitter.py` script with the appropriate arguments for each dataset.

For example the command `make vaihingen-split` will run the following command:

```bash
uv run -m splitters.dataset_splitter --dataset-folder "Vaihingen_dataset" --image-folder "top" --mask-folder "labels" --train-size 0.8 --test-size 0.2
```

The arguments for the `dataset_splitter.py` script are as follows:
- `--dataset-folder` is the name of the folder containing the dataset.
- `--image-folder` is the name of the folder containing the image files.
- `--mask-folder` is the name of the folder containing the mask images.
- `--train-size` is the proportion of the ***entire dataset*** to include as Train set .
- `--test-size` is the proportion of the ***validation dataset*** to include as Test set.

> EXAMPLE: If you have 100 images and set `--train-size` to 0.8 and `--test-size` to 0.2, then 80 images will be allocated to the training set. Of the remaining 20 images, 20% (4 images) will be assigned to the test set, while the other 16 images will be used for validation purposes.