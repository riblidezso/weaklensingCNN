#!/usr/bin/env python
# coding: utf-8
# *Author: Dezso Ribli*


"""
Multi functional CNN training script.
"""


################################################################################
# modules
import numpy as np
import pickle
import argparse
from functools import partial
import os

################################################################################
# args
parser = argparse.ArgumentParser()

# physical params
parser.add_argument("--map-size", type=int, choices=[512], default=512,
                    help="Map size in pixels")
parser.add_argument("--shape-noise", type=float,
                    help="Shape noise, parametrized by the galaxy denisty \
                    number of galaxies per square arcmin.")
parser.add_argument("--smoothing", type=float,
                    help="Smoothing with a Gaussian kernel, width should be \
                    defined in arcmins.")
# cnn params
parser.add_argument("--n-filters", type=int, choices=[16,32,64], default = 32,
                    help="Number of filters in the first convolutional layer")
parser.add_argument("--l2reg", type=float, default = 5e-5,
                    help="L2 regularization coefficient")
# training params
parser.add_argument("--gpu", type=str, choices=['0', '1', '2'],
                    help="GPU id")
parser.add_argument("--reserve-vram", type=float, default=0.9,
                    help="Ratio of memory reserved on gpu.")
parser.add_argument("--batch-size", type=int, default = 32,
                    help="Mini-batch size")
parser.add_argument("--base-lr", type=float, default = 0.001,
                    help="Base learning rate")
parser.add_argument("--n-epochs", type=int, default = 30,
                    help="Number of epochs to train.")
parser.add_argument("--loss", type=str, choices=['mean_absolute_error',
                                                 'mean_squared_error'],
                    default = 'mean_absolute_error',
                    help="Training loss function [mean_absolute_error]")
parser.add_argument("--augment-test", help="Augment test images", 
                    action="store_true")
parser.add_argument("--augment-train", help="Augment train images", 
                    action="store_true")
parser.add_argument("--only-predict", 
                    help="Skip training, only predict if model was saved.",
                    action="store_true")
parser.add_argument("--from-files", 
                    help="Read maps from files when generator datat instead \
                    of reading from memory", action="store_true")
parser.add_argument("--dir", type=str, default = '../../data/sparsegrid',\
                    help="Base directory for data if reading from files")

# boolean switches for different experiments
parser.add_argument("--exclude-fiducial", 
                    help="Exclude maps with fiducial paramteters from the \
                    training set",
                    action="store_true")
parser.add_argument("--grfized", 
                    help="Use equivalent Gaussian random field maps instead of \
                    physical maps",
                    action="store_true")
parser.add_argument("--dense-grid", 
                    help="Use grid with additional extra grid points around \
                    parameter point: (0.26,0.8)",
                    action="store_true")
parser.add_argument("--random-split", 
                    help="Random train-test split instead of based on views",
                    action="store_true")
args = parser.parse_args()
print '\n\n',args,'\n\n'  # report args collected

################################################################################
# set gpu id, and only 90% to keep the N-body simulation alive
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = args.reserve_vram
config.gpu_options.visible_device_list = args.gpu
set_session(tf.Session(config=config))

#keras only after gpu ID and memory usage is set
from keras.callbacks import LearningRateScheduler
from keras import optimizers
# custom stuff 
from utils import load_training_data, step_decay, DataGenerator, predict_on_generator
from nn import mycnn

################################################################################
# Some basic settings whcih can not be changed as of now
# learning schedule drops this is not specified in the argparse due to
# complexity of list assignment
EPOCHS_DROP = [10, 15, 20, 25]  # 10x drop in learning rate after these epochs

################################################################################
# assign a simpler run-name to find data product file names
# more detailed args namespace is saved with preidctions too
RUN_NAME = 'pix'+str(args.map_size)+'_noise'+str(args.shape_noise)
if args.exclude_fiducial:
    RUN_NAME += '_fiducialexcluded'
if args.grfized:
    RUN_NAME += '_grfized'
print 'Run name', RUN_NAME

################################################################################
# create CNN model
model = mycnn(imsize=args.map_size, reg=args.l2reg, nf=args.n_filters)
# simple SGD (later there will be step decay added to the train function)
sgd = optimizers.SGD(lr=args.base_lr, decay=0, momentum=0.9, nesterov=True)
model.compile(loss=args.loss, optimizer=sgd, metrics=[args.loss])  # compile
print model.summary()  # print a summary

################################################################################
# load / preparre data
X_train, X_test, X_newidf, y_train, y_test, y_newidf = load_training_data(
    args.map_size, args.grfized, args.exclude_fiducial, args.dense_grid,
    args.random_split, args.from_files) 

# scale targets to std == 1
scales = y_train.std(axis=0)
y_train /= scales
y_test /= scales

# create data generators for on the fly augmentations and shape noise generation
dg_train = DataGenerator(X_train, y_train, ng=args.shape_noise,  
                         smoothing=args.smoothing, batch_size=args.batch_size,
                         augment=args.augment_train, from_files = args.from_files, 
                         map_size=args.map_size, d=os.path.join(args.dir,'train'))
dg_test = DataGenerator(X_test, y_test, ng=args.shape_noise, shuffle=False,
                        smoothing=args.smoothing, from_files = args.from_files,
                        d=os.path.join(args.dir,'test'), map_size=args.map_size)

################################################################################
# train and save model
sdecay = partial(step_decay, base_lr=args.base_lr, epochs_drop=EPOCHS_DROP)
if not args.only_predict:
    model.fit_generator(dg_train,  nb_epoch=args.n_epochs,
                        steps_per_epoch=dg_train.n_steps, 
                        validation_data=dg_test, validation_steps=dg_test.n_steps,
                        callbacks=[LearningRateScheduler(sdecay)], verbose=2)
    model.save('results/cnn_model_'+RUN_NAME+'.p')  # save the model
else:
    model.load_weights('results/cnn_model_'+RUN_NAME+'.p')  # load the model

################################################################################
# make predictions on test set
y_true_test, p_test = predict_on_generator(model, dg_test, 
                                           args.augment_test)
# scale back targets to original scale
y_true_test *= scales
p_test *= scales
# save predicitons
with open('results/cnn_predictions_' + RUN_NAME + '.pkl', 'wb') as fh:
    pickle.dump((y_true_test, p_test, args),fh)
    
################################################################################ 
# make predictions on unseen new idf
dg_new_idf = DataGenerator(X_newidf, y_newidf, ng=args.shape_noise,
                           shuffle=False, smoothing = args.smoothing,
                           map_size=args.map_size)
y_true_newidf, p_newidf = predict_on_generator(model, dg_new_idf,
                                               args.augment_test)
# scale back targets to original scale
p_newidf *= scales
# save predicitons
with open('results/cnn_predictions_newidf_' + RUN_NAME + '.pkl', 'wb') as fh:
    pickle.dump((y_true_newidf, p_newidf, args),fh)
    
################################################################################
print
print '-----------------------------------'
print 'Finished training and evaluation...'
print 'MAE: %.5f' % np.abs(y_true_test-p_test).mean() 
print 'RMSE: %.5f' % ((y_true_test-p_test)**2).mean()**0.5
print '-----------------------------------'
print