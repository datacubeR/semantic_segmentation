## HF Datasets

To deal with Memory Limitations we created a `create_hf_dataset.py` that converts data into an HF dataset using Arrow. This allows to load the datasets into memory in a more efficient way and prevent OOM errors.

Use the following command: 

```bash
make hf NAME=<dataset_name> PATCH_SIZE=<patch_size> VT=1
```
The arguments for this process are as follows: 

`NAME` is the name of the dataset to convert. Normally, previous processes will create folders with the format `{NAME}_dataset.`
`PATCH_SIZE` is the size of the patches to create. This is normally needed since Ortophotos are too large to fit in memory.
`VT` is an optional flag. If set to 1, it will create validation and test sets. Sometimes this not desired if creating several patches sizes, since the Validation and Test won't use patches but the original ortophotos. Default is 0.
`TR` is an optional flag. If set to 1, it will create the training set. This is useful if you want to create several patch sizes, since the training set is the one that uses patches. Default is 1.

## TL;DR:

The Hugging Face Arrow dataset format improves memory efficiency at the cost of increased disk usage. To mitigate memory constraints during dataset generation, the conversion process is performed in shards. Depending on the available system memory, you may need to experiment with different shard sizes, particularly for the training split.

Some values that worked well for me are the following:

| Dataset   | Train Shard Size | Val/Test Shard Size |
|-----------|------------------|---------------------|
| Vaihingen | 5000             | 10                |
| Potsdam   | 3000             | 10                |
 This process is sharded, you can find more details in [here](src/README.md).
