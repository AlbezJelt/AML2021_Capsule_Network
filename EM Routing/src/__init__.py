import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

# Disable warnings
tf.get_logger().setLevel('INFO')
# Disable tf.contrib warnings
if type(tf.contrib) != type(tf): tf.contrib._warning = None

from src.EMNet import EMNet