# Configuration Files

TL;DR: This folder contains the configuration files used for the experiments in this repository. Each file is a YAML file that defines the parameters and settings for a specific experiment.

For Example file `deeplab_potsdam_v1.yaml` contains the configuration for the first version of the DeepLab model trained on the Potsdam dataset.

The parameters to modify are the following:

* `checkpoint_path`: Null as default, in other cases it serves to resume training for the model from a previous checkpoint.
* `debug`: false by default. When set to true, the code runs in debug mode, loading the data and displaying the model architecture without performing any training. This option is intended solely for debugging purposes.
* model_name: The name of the model to be used for training. It can be one of the following: `unet`, `unetpp`, `deeplab`, `segnet`, `segformer`, `swin`, `upernet`, `dpt`. This is used by the trainer to instantiate model class. 
* dataset_name: The name of the dataset to be used for training. It can be one of the following: `Potsdam`, `Vaihingen`, `LoveDA`, `DeadTrees`. This is used by the trainer to instantiate dataset class.
* max_epochs: Number of epochs to train the model.
* `save_top_k`: Number of Checkpoints to save. 1 means only the best model will be saved. 
* `batch_size`: Batch size for training and validation. In case of Grid Training, corresponds to the number of patches used per training.
* `grad_accumulation_batches`: Number of batches to accumulate.
* `lr`: Learning Rate
* `weight_decay`: Weight Decay for the optimizer.
precision: Float number precision. This depends on the GPU used. In my case i used bf16-mixed since I have an RTX 3090. 

* `in_channels`: Number of input channels. For RGB images, this is 3. 
* `n_classes`: This is the number of classes to predict and depends directly in the dataset used. For Potsdam and Vaihingen, this is 6. For LoveDA, this is 8. For DeadTrees, this is 2.

###############################
# Modifiable Parameters:
###############################
* `patch_size`: Size in Píxels of the patches if Grid Trainer is used (Only for Vaihingen and Potsdam datasets).
* `overlap`: Size in Píxels of the overlap between patches if Grid Trainer is used (Only for Vaihingen and Potsdam datasets).
* `image_size`: Size of the input images in pixels when using the whole image trainer (Only for LoveDA and DeadTrees datasets).

* `use_scheduler`: Flag to use a learning rate scheduler. If set to True ReduceLROnPlateau scheduler will be used. If set to False, no scheduler will be used.
* `scheduler_monitor`: Metric to monitor for the scheduler. 
* `lr_scheduler_kwargs`: Kwargs for the learning rate scheduler. Refer to Pytorch Docs for what options are available. In our case we used the following: 
  * `mode`: Set to max if the metric to monitor is a metric that should be maximized (e.g. accuracy, f1 score, etc.). Set to min if the metric to monitor is a metric that should be minimized (e.g. loss).
  * `factor`: Factor by which the learning rate will be reduced. new_lr = lr * factor.
  * `patience`: Number of epochs with no improvement after which learning rate will be reduced.
  * `threshold`: Threshold for measuring the new optimum, to only focus on significant changes.
  * `threshold_mode`: Use `rel` if the threshold is relative to the best value, `abs` if it is an absolute value.
  * `min_lr`: Minimum learning rate. Learning rate will not be reduced below this value.

*  `loss_function`: Loss Function to use. The codebase only supports the following loss functions: `cross_entropy`, `dice`, `focal`.
* `loss_kwargs`: Kwargs for the loss function. Refer to Pytorch Docs for what options are available. You can check available config options to check some examples.

* `model`: This model name inherits from the `model_name` parameter. It is used to instantiate the model class. The model class will be instantiated with the following parameters:
* `model_kwargs`: Used for the model class instantiation. Depending of the model refer to Segmentation Model Pytorch or HuggingFace Docs to check what parameters are available.

> I have made the best of the efforts to validate this parameters using Pydantic. Hopefully, this will help you to avoid mistakes when modifying the configuration files. If you find any error, please report it in the issues section of this repository.