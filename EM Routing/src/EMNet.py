import os
import re
import shutil  # for regular expressions
import sys
import time

import daiquiri
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import debug as tf_debug  # for debugging
from stat import S_ISREG, ST_CTIME, ST_MODE # to sort files in directory by date

import src.datasets.dataset_factories as dsFactories
import src.models.metrics as met
import src.models.model as mod
import src.utils.config as conf
import src.utils.utils as utl
from src.utils.config import FLAGS

# Get logger that has already been created in config.py
logger = daiquiri.getLogger(__name__)


class EMNet(object):

    def __init__(self, model_name, load_dir=None) -> None:
        """CapsNet with EM routing

        Encapsulates all the operation needed to build and train a CapsNet using EM as routing mechanism.

        Args:
            model_name: name of the model to initialize. All the information to load the model are stored in EM routing/src/datasets/dataset_factories.py
            load_dir=None: path of the directory containing the model to continue training. Ex: 'EM routing/logs/PatchCamelyon/20210901'
        
        Model parameters
            All model parameters are found in EM routing/src/utils/configs.py
            - 
        """
        self.load_dir = load_dir
        self.model_name = model_name
        # Set reproduciable random seed
        tf.compat.v1.set_random_seed(1234)

        # Directories
        self.train_dir, self.train_summary_dir = conf.setup_train_directories()

        # Logger
        conf.setup_logger(logger_dir=self.train_dir, name="logger_train.txt")

        # Hyperparameters
        conf.load_or_save_hyperparams(self.train_dir)

        # Get dataset hyperparameters
        logger.info('Using dataset: {}'.format(self.model_name))
        self.dataset_size_train = dsFactories.get_dataset_size_train(
            self.model_name)
        self.dataset_size_val = dsFactories.get_dataset_size_validate(
            self.model_name)
        self.build_arch = dsFactories.get_dataset_architecture(self.model_name)
        self.num_classes = dsFactories.get_num_classes(self.model_name)
        self.create_inputs_train = dsFactories.get_create_inputs(
            self.model_name, mode="train")
        self.create_inputs_val = dsFactories.get_create_inputs(
            self.model_name, mode="validate")

    def build_model_graph(self):
        """Build the model graph (train and validation)
        """

        # ----------------------------------------------------------------------------
        # GRAPH - TRAIN
        # ----------------------------------------------------------------------------
        logger.info('BUILD TRAIN GRAPH')
        self.g_train = tf.Graph()
        with self.g_train.as_default(), tf.device('/cpu:0'):

            # Get global_step
            global_step = tf.compat.v1.train.get_or_create_global_step()

            # Get batches per epoch
            self.num_batches_per_epoch = int(
                self.dataset_size_train / FLAGS.batch_size)

            # In response to a question on OpenReview, Hinton et al. wrote the
            # following:
            # "We use an exponential decay with learning rate: 3e-3, decay_steps: 20000,     # decay rate: 0.96."
            # https://openreview.net/forum?id=HJWLfGWRb&noteId=ryxTPFDe2X
            lrn_rate = tf.compat.v1.train.exponential_decay(learning_rate=FLAGS.lrn_rate,
                                                            global_step=global_step,
                                                            decay_steps=20000,
                                                            decay_rate=0.96)
            tf.compat.v1.summary.scalar('learning_rate', lrn_rate)
            opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lrn_rate)

            # Get batch from data queue. Batch size is FLAGS.batch_size, which is then
            # divided across multiple GPUs
            input_dict = self.create_inputs_train()
            batch_x = input_dict['image']
            batch_labels = input_dict['label']

            # AG 03/10/2018: Split batch for multi gpu implementation
            # Each split is of size FLAGS.batch_size / FLAGS.num_gpus
            # See: https://github.com/naturomics/CapsNet-Tensorflow/blob/master/
            # dist_version/distributed_train.py
            splits_x = tf.split(
                axis=0,
                num_or_size_splits=FLAGS.num_gpus,
                value=batch_x)
            splits_labels = tf.split(
                axis=0,
                num_or_size_splits=FLAGS.num_gpus,
                value=batch_labels)

            # --------------------------------------------------------------------------
            # MULTI GPU - TRAIN
            # --------------------------------------------------------------------------
            # Calculate the gradients for each model tower
            tower_grads = []
            tower_losses = []
            tower_logits = []
            reuse_variables = None
            for i in range(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % i) as scope:
                        logger.info('TOWER %d' % i)
                        # with slim.arg_scope([slim.model_variable, slim.variable],
                        # device='/cpu:0'):
                        with slim.arg_scope([slim.variable], device='/cpu:0'):
                            loss, logits = tower_fn(
                                self.build_arch,
                                splits_x[i],
                                splits_labels[i],
                                scope,
                                self.num_classes,
                                reuse_variables=reuse_variables,
                                is_train=True)

                        # Don't reuse variable for first GPU, but do reuse for others
                        reuse_variables = True

                        # Compute gradients for one GPU
                        grads = opt.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

                        # Keep track of losses and logits across for each tower
                        tower_logits.append(logits)
                        tower_losses.append(loss)

                        # Loss for each tower
                        tf.compat.v1.summary.scalar("loss", loss)

            # We must calculate the mean of each gradient. Note that this is the
            # synchronization point across all towers.
            grad = average_gradients(tower_grads)

            # See: https://stackoverflow.com/questions/40701712/how-to-check-nan-in-
            # gradients-in-tensorflow-when-updating
            grad_check = ([tf.debugging.check_numerics(g, message='Gradient NaN Found!')
                           for g, _ in grad if g is not None]
                          + [tf.debugging.check_numerics(loss, message='Loss NaN Found')])

            # Apply the gradients to adjust the shared variables
            with tf.control_dependencies(grad_check):
                update_ops = tf.compat.v1.get_collection(
                    tf.compat.v1.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.train_op = opt.apply_gradients(
                        grad, global_step=global_step)

            # Calculate mean loss
            loss = tf.reduce_mean(tower_losses)

            # Calculate accuracy
            logits = tf.concat(tower_logits, axis=0)
            acc = met.accuracy(logits, batch_labels)

            # Prepare predictions and one-hot labels
            probs = tf.nn.softmax(logits=logits)
            labels_oh = tf.one_hot(batch_labels, self.num_classes)

            # Group metrics together
            # See: https://cs230-stanford.github.io/tensorflow-model.html
            self.trn_metrics = {'loss': loss,
                                'labels': batch_labels,
                                'labels_oh': labels_oh,
                                'logits': logits,
                                'probs': probs,
                                'acc': acc,
                                }

            # Reset and read operations for streaming metrics go here
            self.trn_reset = {}
            self.trn_read = {}

            # Logging
            tf.compat.v1.summary.scalar('trn_loss', loss)
            tf.compat.v1.summary.scalar('trn_acc', acc)

            # Set Saver
            # AG 26/09/2018: Save all variables including Adam so that we can continue
            # training from where we left off
            # max_to_keep=None should keep all checkpoints
            self.saver = tf.compat.v1.train.Saver(
                tf.compat.v1.global_variables(), max_to_keep=None)

            # Display number of parameters
            train_params = np.sum([np.prod(v.get_shape().as_list())
                                   for v in tf.compat.v1.trainable_variables()]).astype(np.int32)
            logger.info('Trainable Parameters: {}'.format(train_params))

            # Set summary op
            self.trn_summary = tf.compat.v1.summary.merge_all()

        # ----------------------------------------------------------------------------
        # GRAPH - VALIDATION
        # ----------------------------------------------------------------------------
        logger.info('BUILD VALIDATION GRAPH')
        self.g_val = tf.Graph()
        with self.g_val.as_default():
            # Get global_step
            global_step = tf.compat.v1.train.get_or_create_global_step()

            self.num_batches_val = int(
                self.dataset_size_val / FLAGS.batch_size)

            # Get data
            input_dict = self.create_inputs_val()
            batch_x = input_dict['image']
            batch_labels = input_dict['label']

            # AG 10/12/2018: Split batch for multi gpu implementation
            # Each split is of size FLAGS.batch_size / FLAGS.num_gpus
            # See: https://github.com/naturomics/CapsNet-
            # Tensorflow/blob/master/dist_version/distributed_train.py
            splits_x = tf.split(
                axis=0,
                num_or_size_splits=FLAGS.num_gpus,
                value=batch_x)
            splits_labels = tf.split(
                axis=0,
                num_or_size_splits=FLAGS.num_gpus,
                value=batch_labels)

            # --------------------------------------------------------------------------
            # MULTI GPU - VALIDATE
            # --------------------------------------------------------------------------
            # Calculate the logits for each model tower
            tower_logits = []
            reuse_variables = None
            for i in range(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % i) as scope:
                        with slim.arg_scope([slim.variable], device='/cpu:0'):
                            loss, logits = tower_fn(
                                self.build_arch,
                                splits_x[i],
                                splits_labels[i],
                                scope,
                                self.num_classes,
                                reuse_variables=reuse_variables,
                                is_train=False)

                        # Don't reuse variable for first GPU, but do reuse for others
                        reuse_variables = True

                        # Keep track of losses and logits across for each tower
                        tower_logits.append(logits)

                        # Loss for each tower
                        tf.compat.v1.summary.histogram("val_logits", logits)

            # Combine logits from all towers
            logits = tf.concat(tower_logits, axis=0)

            # Calculate metrics
            val_loss = mod.spread_loss(logits, batch_labels)
            val_acc = met.accuracy(logits, batch_labels)

            # Prepare predictions and one-hot labels
            val_probs = tf.nn.softmax(logits=logits)
            val_labels_oh = tf.one_hot(batch_labels, self.num_classes)

            # Group metrics together
            # See: https://cs230-stanford.github.io/tensorflow-model.html
            self.val_metrics = {'loss': val_loss,
                                'labels': batch_labels,
                                'labels_oh': val_labels_oh,
                                'logits': logits,
                                'probs': val_probs,
                                'acc': val_acc,
                                }

            # Reset and read operations for streaming metrics go here
            self.val_reset = {}
            self.val_read = {}

            tf.compat.v1.summary.scalar("val_loss", val_loss)
            tf.compat.v1.summary.scalar("val_acc", val_acc)

            # Saver
            self.saver = tf.compat.v1.train.Saver(max_to_keep=None)

            # Set summary op
            self.val_summary = tf.compat.v1.summary.merge_all()

    def train(self):
        """Start the training and validation session, than execute the train loop.
        """
        # ****************************************************************************
        # SESSIONS
        # ****************************************************************************

        #----- SESSION TRAIN -----#
        # Session settings
        config_train = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                                log_device_placement=False)
        config_train.gpu_options.allow_growth = True
        sess_train = tf.compat.v1.Session(
            config=config_train, graph=self.g_train)

        # Debugger
        # AG 05/06/2018: Debugging using either command line or TensorBoard
        if FLAGS.debugger is not None:
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess_train = tf_debug.TensorBoardDebugWrapperSession(sess_train,
                                                                 FLAGS.debugger)

        with self.g_train.as_default():
            sess_train.run([tf.compat.v1.global_variables_initializer(),
                            tf.compat.v1.local_variables_initializer()])

            # Restore previous checkpoint
            # AG 26/09/2018: where should this go???
            if self.load_dir is not None:
                load_dir_checkpoint = os.path.join(
                    self.load_dir, "train", "checkpoint")
                prev_step = load_training(
                    self.saver, sess_train, load_dir_checkpoint)
            else:
                prev_step = 0

        # Create summary writer, and write the train graph
        summary_writer = tf.compat.v1.summary.FileWriter(self.train_summary_dir,
                                                         graph=sess_train.graph)

        #----- SESSION VALIDATION -----#
        config_val = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                              log_device_placement=False)
        config_val.gpu_options.allow_growth = True
        #config_train.gpu_options.per_process_gpu_memory_fraction = 0.8
        sess_val = tf.compat.v1.Session(config=config_val, graph=self.g_val)
        with self.g_val.as_default():
            sess_val.run([tf.compat.v1.local_variables_initializer(),
                          tf.compat.v1.global_variables_initializer()])

        # ****************************************************************************
        # MAIN LOOP
        # ****************************************************************************
        SUMMARY_FREQ = 100
        SAVE_MODEL_FREQ = self.num_batches_per_epoch  # 500
        VAL_FREQ = self.num_batches_per_epoch  # 500
        PROFILE_FREQ = 5

        for step in range(prev_step, FLAGS.epoch * self.num_batches_per_epoch + 1):
            # for step in range(0,3):
            # AG 23/05/2018: limit number of iterations for testing
            # for step in range(100):
            epoch_decimal = step/self.num_batches_per_epoch
            epoch = int(np.floor(epoch_decimal))

            # TF queue would pop batch until no file
            try:
                # TRAIN
                with self.g_train.as_default():

                    # With profiling
                    if (FLAGS.profile is True) and ((step % PROFILE_FREQ) == 0):
                        logger.info("Train with Profiling")
                        run_options = tf.RunOptions(
                            trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                    # Without profiling
                    else:
                        run_options = None
                        run_metadata = None

                    # Reset streaming metrics
                    if step % (self.num_batches_per_epoch/4) == 1:
                        logger.info("Reset streaming metrics")
                        sess_train.run([self.trn_reset])

                    # MAIN RUN
                    tic = time.time()
                    train_op_v, trn_metrics_v, trn_summary_v = sess_train.run(
                        [self.train_op, self.trn_metrics, self.trn_summary],
                        options=run_options,
                        run_metadata=run_metadata)
                    toc = time.time()

                    # Read streaming metrics
                    trn_read_v = sess_train.run(self.trn_read)

                    # Write summary for profiling
                    if run_options is not None:
                        summary_writer.add_run_metadata(
                            run_metadata, 'step{:d}'.format(step))

                    # Logging
                    logger.info('TRN'
                                + ' e-{:d}'.format(epoch)
                                + ' stp-{:d}'.format(step)
                                + ' {:.2f}s'.format(toc - tic)
                                + ' loss: {:.4f}'.format(trn_metrics_v['loss'])
                                + ' acc: {:.2f}%'.format(trn_metrics_v['acc']*100)
                                )

            except KeyboardInterrupt:
                sess_train.close()
                sess_val.close()
                sys.exit()

            except tf.errors.InvalidArgumentError as e:
                logger.warning(
                    '%d iteration contains NaN gradients. Discard.' % step)
                logger.error(str(e))
                continue

            else:
                # WRITE SUMMARY
                if (step % SUMMARY_FREQ) == 0:
                    logger.info("Write Train Summary")
                    with self.g_train.as_default():
                        # Summaries from graph
                        summary_writer.add_summary(trn_summary_v, step)

                # SAVE MODEL
                if (step % SAVE_MODEL_FREQ) == 100:
                    logger.info("Save Model")
                    with self.g_train.as_default():
                        train_checkpoint_dir = self.train_dir + '/checkpoint'
                        if not os.path.exists(train_checkpoint_dir):
                            os.makedirs(train_checkpoint_dir)

                        # Save ckpt from train session
                        ckpt_path = os.path.join(
                            train_checkpoint_dir, 'model.ckpt')
                        self.saver.save(sess_train, ckpt_path,
                                        global_step=step)

                # VALIDATE MODEL
                if (step % VAL_FREQ) == 100:
                    #----- Validation -----#
                    with self.g_val.as_default():
                        logger.info("Start Validation")

                        # Restore ckpt to val session
                        latest_ckpt = tf.train.latest_checkpoint(
                            train_checkpoint_dir)
                        self.saver.restore(sess_val, latest_ckpt)

                        # Reset accumulators
                        accuracy_sum = 0
                        loss_sum = 0
                        sess_val.run(self.val_reset)

                        for i in range(self.num_batches_val):
                            val_metrics_v, val_summary_str_v = sess_val.run(
                                [self.val_metrics, self.val_summary])

                            # Update
                            accuracy_sum += val_metrics_v['acc']
                            loss_sum += val_metrics_v['loss']

                            # Read
                            val_read_v = sess_val.run(self.val_read)

                            # Get checkpoint number
                            ckpt_num = re.split('-', latest_ckpt)[-1]

                            # Logging
                            logger.info('VAL ckpt-{}'.format(ckpt_num)
                                        + ' bch-{:d}'.format(i)
                                        + ' cum_acc: {:.2f}%'.format(accuracy_sum/(i+1)*100)
                                        + ' cum_loss: {:.4f}'.format(loss_sum/(i+1))
                                        )

                        # Average across batches
                        ave_acc = accuracy_sum / self.num_batches_val
                        ave_loss = loss_sum / self.num_batches_val

                        logger.info('VAL ckpt-{}'.format(ckpt_num)
                                    + ' avg_acc: {:.2f}%'.format(ave_acc*100)
                                    + ' avg_loss: {:.4f}'.format(ave_loss)
                                    )

                        logger.info("Write Val Summary")
                        summary_val = tf.compat.v1.Summary()
                        summary_val.value.add(
                            tag="val_acc", simple_value=ave_acc)
                        summary_val.value.add(
                            tag="val_loss", simple_value=ave_loss)
                        summary_writer.add_summary(summary_val, step)

        # Close (main loop)
        sess_train.close()
        sess_val.close()
        sys.exit()

    def test(self):
        
        # Directories
        # Get name
        split = self.load_dir.split('/')
        if split[-1]:
            name = split[-1]
        else:
            name = split[-2]

        # Get parent directory
        split = self.load_dir.split("/" + name)
        parent_dir = split[0]

        test_dir = '{}/{}/test'.format(parent_dir, name)
        test_summary_dir = test_dir + '/summary'

        # Clear the test log directory
        if (FLAGS.reset is True) and os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        if not os.path.exists(test_summary_dir):
            os.makedirs(test_summary_dir)

        # Logger
        conf.setup_logger(logger_dir=test_dir, name="logger_test.txt")
        logger.info("name: " + name)
        logger.info("parent_dir: " + parent_dir)
        logger.info("test_dir: " + test_dir)

        # Load hyperparameters from train run
        conf.load_or_save_hyperparams()

        # Get dataset hyperparameters
        logger.info('Using dataset: {}'.format(FLAGS.dataset))

        # Dataset
        dataset_size_test = conf.get_dataset_size_test(FLAGS.dataset)
        num_classes = conf.get_num_classes(FLAGS.dataset)
        create_inputs_test = conf.get_create_inputs(FLAGS.dataset, mode="test")

        # ----------------------------------------------------------------------------
        # GRAPH - TEST
        # ----------------------------------------------------------------------------
        logger.info('BUILD TEST GRAPH')
        g_test = tf.Graph()
        with g_test.as_default():
            # Get global_step
            global_step = tf.train.get_or_create_global_step()

            num_batches_test = int(dataset_size_test / FLAGS.batch_size)

            # Get data
            input_dict = create_inputs_test()
            batch_x = input_dict['image']
            batch_labels = input_dict['label']

            # AG 10/12/2018: Split batch for multi gpu implementation
            # Each split is of size FLAGS.batch_size / FLAGS.num_gpus
            # See: https://github.com/naturomics/CapsNet-
            # Tensorflow/blob/master/dist_version/distributed_train.py
            splits_x = tf.split(
                axis=0,
                num_or_size_splits=FLAGS.num_gpus,
                value=batch_x)
            splits_labels = tf.split(
                axis=0,
                num_or_size_splits=FLAGS.num_gpus,
                value=batch_labels)

            # Build architecture
            build_arch = conf.get_dataset_architecture(FLAGS.dataset)

            # --------------------------------------------------------------------------
            # MULTI GPU - TEST
            # --------------------------------------------------------------------------
            # Calculate the logits for each model tower
            tower_logits = []
            reuse_variables = None
            for i in range(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % i) as scope:
                        with slim.arg_scope([slim.variable], device='/cpu:0'):
                            loss, logits = tower_fn(
                                build_arch,
                                splits_x[i],
                                splits_labels[i],
                                scope,
                                num_classes,
                                reuse_variables=reuse_variables,
                                is_train=False)

                        # Don't reuse variable for first GPU, but do reuse for others
                        reuse_variables = True

                        # Keep track of losses and logits across for each tower
                        tower_logits.append(logits)

                        # Loss for each tower
                        tf.compat.v1.summary.histogram("test_logits", logits)

            # Combine logits from all towers
            logits = tf.concat(tower_logits, axis=0)

            # Calculate metrics
            test_loss = mod.spread_loss(logits, batch_labels)
            test_acc = met.accuracy(logits, batch_labels)

            # Prepare predictions and one-hot labels
            test_probs = tf.nn.softmax(logits=logits)
            test_labels_oh = tf.one_hot(batch_labels, num_classes)

            # Group metrics together
            # See: https://cs230-stanford.github.io/tensorflow-model.html
            test_metrics = {'loss': test_loss,
                            'labels': batch_labels,
                            'labels_oh': test_labels_oh,
                            'logits': logits,
                            'probs': test_probs,
                            'acc': test_acc,
                            }

            # Reset and read operations for streaming metrics go here
            test_reset = {}
            test_read = {}

            tf.compat.v1.summary.scalar("test_loss", test_loss)
            tf.compat.v1.summary.scalar("test_acc", test_acc)

            # Saver
            saver = tf.train.Saver(max_to_keep=None)

            # Set summary op
            test_summary = tf.compat.v1.Summary.merge_all()

            # --------------------------------------------------------------------------
            # SESSION - TEST
            # --------------------------------------------------------------------------
            sess_test = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False),
                graph=g_test)

            # sess_test.run(tf.local_variables_initializer())
            # sess_test.run(tf.global_variables_initializer())

            summary_writer = tf.compat.v1.Summary.FileWriter(
                test_summary_dir,
                graph=sess_test.graph)

            ckpts_to_test = []
            load_dir_checkpoint = os.path.join(
                FLAGS.load_dir, "train", "checkpoint")

            # Evaluate the latest ckpt in dir
            if FLAGS.ckpt_name is None:
                latest_ckpt = tf.train.latest_checkpoint(load_dir_checkpoint)
                ckpts_to_test.append(latest_ckpt)

            # Evaluate all ckpts in dir
            elif FLAGS.ckpt_name == "all":
                # Get list of files in firectory and sort by date created
                filenames = os.listdir(load_dir_checkpoint)
                regex = re.compile(r'.*.index')
                filenames = filter(regex.search, filenames)
                data_ckpts = (os.path.join(load_dir_checkpoint, fn)
                            for fn in filenames)
                data_ckpts = ((os.stat(path), path) for path in data_ckpts)

                # regular files, insert creation date
                data_ckpts = ((stat[ST_CTIME], path) for stat, path in data_ckpts
                            if S_ISREG(stat[ST_MODE]))
                data_ckpts = sorted(data_ckpts)
                # remove ".index"
                ckpts_to_test = [path[:-6] for ctime, path in data_ckpts]

            # Evaluate ckpt specified by name
            else:
                ckpt_name = os.path.join(load_dir_checkpoint, FLAGS.ckpt_name)
                ckpts_to_test.append(ckpt_name)

            # --------------------------------------------------------------------------
            # MAIN LOOP
            # --------------------------------------------------------------------------
            # Run testing on checkpoints
            for ckpt in ckpts_to_test:
                saver.restore(sess_test, ckpt)

                # Reset accumulators
                accuracy_sum = 0
                loss_sum = 0
                sess_test.run(test_reset)

                for i in range(num_batches_test):

                    test_metrics_v, test_summary_str_v = sess_test.run(
                        [test_metrics, test_summary])

                    # Update
                    accuracy_sum += test_metrics_v['acc']
                    loss_sum += test_metrics_v['loss']

                    ckpt_num = re.split('-', ckpt)[-1]
                    logger.info('TEST ckpt-{}'.format(ckpt_num)
                                + ' bch-{:d}'.format(i)
                                + ' cum_acc: {:.2f}%'.format(accuracy_sum/(i+1)*100)
                                + ' cum_loss: {:.4f}'.format(loss_sum/(i+1))
                                )

                ave_acc = accuracy_sum / num_batches_test
                ave_loss = loss_sum / num_batches_test

                logger.info('TEST ckpt-{}'.format(ckpt_num)
                            + ' avg_acc: {:.2f}%'.format(ave_acc*100)
                            + ' avg_loss: {:.4f}'.format(ave_loss))

                logger.info("Write Test Summary")
                summary_test = tf.compat.v1.Summary()
                summary_test.value.add(tag="test_acc", simple_value=ave_acc)
                summary_test.value.add(tag="test_loss", simple_value=ave_loss)
                summary_writer.add_summary(summary_test, ckpt_num)

def tower_fn(build_arch,
             x,
             y,
             scope,
             num_classes,
             is_train=True,
             reuse_variables=None):
    """Model tower to be run on each GPU.

    Author:
            Ashley Gritzman 27/11/2018

    Args: 
            build_arch:
            x: split of batch_x allocated to particular GPU
            y: split of batch_y allocated to particular GPU
            scope:
            num_classes:
            is_train:
            reuse_variables: False for the first GPU, and True for subsequent GPUs

    Returns:
            loss: mean loss across samples for one tower (scalar)
            scores: 
                    If the architecture is a capsule network, then the scores are the output 
                    activations of the class caps.
                    If the architecture is the CNN baseline, then the scores are the logits of 
                    the final layer.
                    (samples_per_tower, n_classes)
                    (64/4=16, 5)
    """

    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=reuse_variables):
        output = build_arch(x, is_train, num_classes=num_classes)
        scores = output['scores']

    loss = mod.total_loss(scores, y)

    return loss, scores


def average_gradients(tower_grads):
    """Compute average gradients across all towers.

    Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.

    Credit:
            https://github.com/naturomics/CapsNet-
            Tensorflow/blob/master/dist_version/distributed_train.py
    Args:
            tower_grads: 
                    List of lists of (gradient, variable) tuples. The outer list is over 
                    individual gradients. The inner list is over the gradient calculation for       each tower.
    Returns:
            average_grads:
                    List of pairs of (gradient, variable) where the gradient has been 
                    averaged across all towers.
    """

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def extract_step(path):
    """Returns the step from the file format name of Tensorflow checkpoints.

    Credit:
            Sara Sabour
            https://github.com/Sarasra/models/blob/master/research/capsules/
            experiment.py
    Args:
            path: The checkpoint path returned by tf.train.get_checkpoint_state.
            The format is: {ckpnt_name}-{step}
    Returns:
            The last training step number of the checkpoint.
    """
    file_name = os.path.basename(path)
    return int(file_name.split('-')[-1])


def load_training(saver, session, load_dir):
    """Loads a saved model into current session or initializes the directory.

    If there is no functioning saved model or FLAGS.restart is set, cleans the
    load_dir directory. Otherwise, loads the latest saved checkpoint in load_dir
    to session.

    Author:
            Ashley Gritzman 26/09/2018
    Credit:
            Adapted from Sara Sabour
            https://github.com/Sarasra/models/blob/master/research/capsules/
            experiment.py
    Args:
            saver: An instance of tf.compat.v1.train.Saver to load the model in to the session.
            session: An instance of tf.compat.v1.Session with the built-in model graph.
            load_dir: The directory which is used to load the latest checkpoint.

    Returns:
            The latest saved step.
    """

    if tf.io.gfile.exists(load_dir):
        ckpt = tf.train.get_checkpoint_state(load_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            prev_step = extract_step(ckpt.model_checkpoint_path)
            logger.info("Restored checkpoint")
        else:
            raise IOError("""AG: load_ckpt directory exists but cannot find a valid 
										checkpoint to resore, consider using the reset flag""")
    else:
        raise IOError("AG: load_ckpt directory does not exist")

    return prev_step


def find_checkpoint(load_dir, seen_step):
    """Finds the global step for the latest written checkpoint to the load_dir.

    Credit:
            Sara Sabour
            https://github.com/Sarasra/models/blob/master/research/capsules/
            experiment.py
    Args:
            load_dir: The directory address to look for the training checkpoints.
            seen_step: Latest step which evaluation has been done on it.
    Returns:
            The latest new step in the load_dir and the file path of the latest model
            in load_dir. If no new file is found returns -1 and None.
    """
    ckpt = tf.train.get_checkpoint_state(load_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = extract_step(ckpt.model_checkpoint_path)
        if int(global_step) != seen_step:
            return int(global_step), ckpt.model_checkpoint_path
    return -1, None
