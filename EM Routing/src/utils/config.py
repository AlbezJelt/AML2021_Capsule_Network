"""
License: Apache 2.0
Author: Ashley Gritzman
E-mail: ashley.gritzman@za.ibm.com
"""

import tensorflow as tf
from datetime import datetime   # date stamp the log directory
import json  # for saving and loading hyperparameters
import os
import sys
import re
import time
import pathlib

import daiquiri
import logging
logger = daiquiri.getLogger(__name__)

sys.argv = sys.argv[:1]

flags = tf.app.flags

# Need this line for flags to work with Jupyter
# https://github.com/tensorflow/tensorflow/issues/17702
flags.DEFINE_string('f', '', 'kernel')

# ------------------------------------------------------------------------------
# HYPERPARAMETERS
# ------------------------------------------------------------------------------
# set to 64 according to authors (https://openreview.net/forum?id=HJWLfGWRb)
flags.DEFINE_integer('batch_size', 32, 'batch size in total across all gpus')
flags.DEFINE_integer('epoch', 2000, 'epoch')
flags.DEFINE_integer('iter_routing', 2, 'number of iterations')
flags.DEFINE_integer('num_gpus', 1, 'number of GPUs')
flags.DEFINE_float('epsilon', 1e-9, 'epsilon')
flags.DEFINE_float('lrn_rate', 3e-3, 'learning rate to use in Adam optimiser')
flags.DEFINE_boolean('weight_reg', False,
                     'train with regularization of weights')
flags.DEFINE_string('norm', 'norm2', 'norm type')
flags.DEFINE_float('final_lambda', 0.01, 'final lambda in EM routing')


# ------------------------------------------------------------------------------
# ARCHITECTURE PARAMETERS
# ------------------------------------------------------------------------------
flags.DEFINE_integer('A', 64, 'number of channels in output from ReLU Conv1')
flags.DEFINE_integer('B', 8, 'number of capsules in output from PrimaryCaps')
flags.DEFINE_integer('C', 16, 'number of channels in output from ConvCaps1')
flags.DEFINE_integer('D', 16, 'number of channels in output from ConvCaps2')


# ------------------------------------------------------------------------------
# ENVIRONMENT SETTINGS
# ------------------------------------------------------------------------------
flags.DEFINE_string('mode', 'train', 'train, validate, or test')
flags.DEFINE_string('name', '', 'name of experiment in log directory')
flags.DEFINE_boolean('reset', False, 'clear the train or test log directory')
flags.DEFINE_string('debugger', None,
                    '''set to host of TensorBoard debugger e.g. "dccxc180:8886 
										or dccxl015:8770"''')
flags.DEFINE_boolean('profile', False,
                     '''get runtime statistics to display inTensorboard e.g. 
										 compute time''')
flags.DEFINE_string('load_dir', None,
                    '''directory containing train or test checkpoints to 
										continue from''')
flags.DEFINE_string('ckpt_name', None,
                    '''None to load the latest ckpt; all to load all ckpts in 
											dir; name to load specific ckpt''')
flags.DEFINE_string('params_path', None, 'path to JSON containing parameters')

this_file_path = pathlib.Path(__file__).parent.resolve()
LOCAL_STORAGE = this_file_path.parent.parent
flags.DEFINE_string('storage', str(LOCAL_STORAGE),
                    'directory where logs and data are stored')
                    
flags.DEFINE_string('db_name', 'capsules_ex1',
                    'Name of the DB for mongo for sacred')

# Parse flags
FLAGS = flags.FLAGS


# ------------------------------------------------------------------------------
# DIRECTORIES
# ------------------------------------------------------------------------------
def setup_train_directories(model_name):

    # Set log directory
    date_stamp = datetime.now().strftime('%Y%m%d')
    save_dir = os.path.join(tf.app.flags.FLAGS.storage, 'logs/',
                            model_name)
    train_dir = '{}/{}_{}/train'.format(save_dir, date_stamp, FLAGS.name)

    # Clear the train log directory
    if FLAGS.reset is True and tf.io.gfile.exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)

    # Create train directory
    if not tf.io.gfile.exists(train_dir):
        tf.gfile.MakeDirs(train_dir)

    # Set summary directory
    train_summary_dir = os.path.join(train_dir, 'summary')

    # Create summary directory
    if not tf.io.gfile.exists(train_summary_dir):
        tf.gfile.MakeDirs(train_summary_dir)

    return train_dir, train_summary_dir


# ------------------------------------------------------------------------------
# SETUP LOGGER
# ------------------------------------------------------------------------------
def setup_logger(logger_dir, name="logger"):
    os.environ['TZ'] = 'Europe/Rome'
    time.tzset()
    daiquiri_formatter = daiquiri.formatter.ColorFormatter(
        fmt="%(asctime)s %(color)s%(levelname)s: %(message)s%(color_stop)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    logger_path = os.path.join(logger_dir, name)
    daiquiri.setup(level=logging.INFO, outputs=(
        daiquiri.output.Stream(formatter=daiquiri_formatter),
        daiquiri.output.File(logger_path, formatter=daiquiri_formatter),
    ))
    # To access the logger from other files, just put this line at the top:
    # logger = daiquiri.getLogger(__name__)


# ------------------------------------------------------------------------------
# LOAD OR SAVE HYPERPARAMETERS
# ------------------------------------------------------------------------------
def load_or_save_hyperparams(train_dir, load_dir=None):

    # Load parameters from file
    # params_path is given in the case that run a new training using existing
    # parameters
    # load_dir is given in the case of testing or continuing training
    if load_dir:

        params_path = os.path.join(load_dir, "train",
                                    "params", "params.json")
        params_path = os.path.abspath(params_path)

        with open(params_path, 'r') as params_file:
            params = json.load(params_file)
            for name, value in params.items():
                FLAGS.__flags[name].value = value

        logger.info("Loaded parameters from file: {}".format(params_path))

    # Save parameters to file
    else:
        params_dir_path = os.path.join(train_dir, "params")
        os.makedirs(params_dir_path, exist_ok=True)
        params_file_path = os.path.join(params_dir_path, "params.json")
        params = FLAGS.flag_values_dict()
        params_json = json.dumps(params, indent=4, separators=(',', ':'))
        with open(params_file_path, 'w') as params_file:
            params_file.write(params_json)
        logger.info("Parameters saved to file: {}".format(params_file_path))

