"""
This is the helper functions for various functions
"""
import os
import shutil
from copy import deepcopy
import sys
import pickle
import numpy as np


# 5
def put_param_into_folder(ckpt_dir):
    """
    Put the parameter.txt into the folder and the flags.obj as well
    :return: None
    """
    """
    Old version of finding the latest changing file, deprecated
    # list_of_files = glob.glob('models/*')                           # Use glob to list the dirs in models/
    # latest_file = max(list_of_files, key=os.path.getctime)          # Find the latest file (just trained)
    # print("The parameter.txt is put into folder " + latest_file)    # Print to confirm the filename
    """
    # Move the parameters.txt
    destination = os.path.join(ckpt_dir, "parameters.txt");
    shutil.move("parameters.txt", destination)
    # Move the flags.obj
    destination = os.path.join(ckpt_dir, "flags.obj");
    shutil.move("flags.obj", destination)


# 6
def save_flags(flags, save_dir, save_file="flags.obj"):
    """
    This function serialize the flag object and save it for further retrieval during inference time
    :param flags: The flags object to save
    :param save_file: The place to save the file
    :return: None
    """
    with open(os.path.join(save_dir, save_file),'wb') as f:          # Open the file
        pickle.dump(flags, f)               # Use Pickle to serialize the object


# 7
def load_flags(save_dir, save_file="flags.obj"):
    """
    This function inflate the pickled object to flags object for reuse, typically during evaluation (after training)
    :param save_dir: The place where the obj is located
    :param save_file: The file name of the file, usually flags.obj
    :return: flags
    """
    with open(os.path.join(save_dir, save_file), 'rb') as f:     # Open the file
        flags = pickle.load(f)                                  # Use pickle to inflate the obj back to RAM
    return flags


# 8
def write_flags_and_BVE(flags, ntwk, forward_best_loss=None):
    """
    The function that is usually executed at the end of the training where the flags and the best validation loss are recorded
    They are put in the folder that called this function and save as "parameters.txt"
    This parameter.txt is also attached to the generated email
    :param flags: The flags struct containing all the parameters
    :param best_validation_loss: The best_validation_loss recorded in a training
    :param forard_best_loss: The forward best loss only applicable for Tandem model
    :return: None
    """
    flags.best_validation_loss = ntwk.best_validation_loss  # Change the y range to be acceptable long string
    flags.best_training_loss = ntwk.best_training_loss  # Change the y range to be acceptable long string
    if forward_best_loss is not None:
        flags.best_forward_validation_loss = forward_best_loss
    copy_flags = deepcopy(flags)
    flags_dict = vars(copy_flags)
    # Convert the dictionary into pandas data frame which is easier to handle with and write read
    with open(os.path.join(ntwk.ckpt_dir, 'parameters.txt'), 'w') as f:
        print(flags_dict, file=f)
    # Pickle the obj
    save_flags(flags, save_dir=ntwk.ckpt_dir)

