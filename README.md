# WIP: base_project
## Goal
The goal of this package is to write a package that makes it easy to train machine learning
projects. The aim is to manage as much of the settings of the project in the config.yaml file, to
make the project reproducible and easily adaptable. The training process can be logged using Weights
and Biases, which can be found [here](https://wandb.ai/).
## Configuration
The config.yaml file provides an overview of which settings can be utilized.
### train_dataset_cfg/val_dataset_cfg
Requires a name that corresponds with a name in the dictionary found in dataset_builder.py This
determines which dataset builder function is called. Additionally you can pass the kwargs that the
dataset builder function needs to build the dataset.

### train_dataloader_cfg/val_dataloader_cfg
Here you can pass all the parameters that are necessary to set up the dataloaders, such as
batch_size, shuffle, num_workers, drop_last, etc.

### trainer_cfg
Requires a name that corresponds with a name in the dictionary found in trainer_builder.py This
determines which trainer builder function is called.

### model_cfg
Requires a name that corresponds with a name in the dictionary found in model_builder.py This
determines which model builder function is called. Additionally you can pass the kwargs that the
model builder function needs to build the model.

### loss_cfg
Requires a name that corresponds with a name in the dictionary found in loss_builder.py This
determines which loss builder function is called. Additionally you can pass the kwargs that the
loss builder function needs to build the loss function.

### optimizer_cfg
Requires a name that corresponds with a name in the dictionary found in optimizer_builder.py This
determines which optimizer builder function is called. Additionally you can pass the kwargs that the
optimizer builder function needs to build the optimizer.

### session_cfg
Takes epochs to determine the number of epochs, and metrics which is a list of metrics that should
be logged to Weights and Biases. The available metrics are found in the run_files/metrics.py file.
They are registered to the dictionary via decorator functions, so if you import the metrics.py file
you can check the dictionary called METRIC_FUNCTIONS to see which ones are available. Training and
validation loss are logged on every epoch by default. You can also select a metric to select the
best model on with 'selection_metric' (default: 'loss_val_epoch'), and specifiy the 'goal' as
'minimize' or 'maximize'.

### log_cfg
Takes a 'wandb_init' key with as value the kwargs that you want to pass to the wandb.init function.

## Additional programming
You will probably have to create your own pytorch dataset and add it to the dataset_builder, as well
as your own model architecture which you should add to the model_builder. If you want more options
for loss or optimizers you can add those easily as well to their respective builder files.

## Running
Once everything is configured in the configuration file, cd to your project folder and install the
anaconda environment in the environment.yaml file. Then you just have to run 'main.py' from the
command line to start the training process.
# Contributing
Right now there are not so many model architectures available, nor are there many metrics, loss
functions or optimizers. It is however very easy to implement new ones, just add a function in one
of the builder classes and register it to the dictionary with a name. Then you can call it from the
configuration file. If you want to contribute, make a pull request with the feature you would like
to add.