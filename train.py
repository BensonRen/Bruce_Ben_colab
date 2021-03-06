"""
This file serves as a training interface for training the network
"""
# Built in
import glob
import os
import pandas as pd
import numpy as np
import sys
# Torch

# Own
import flag_reader
import data_reader
from class_wrapper import Network
from model_maker import MLP, CNN
from helper_functions import put_param_into_folder, write_flags_and_BVE


def training_from_flag(flags):
    """
    Training interface. 1. Read data 2. initialize network 3. train network 4. record flags
    :param flag: The training flags read from command line or parameter.py
    :return: None
    """
    
    print(flags)

    # Get the data
    train_loader, test_loader = data_reader.read_data(flags)

    print("Making network now")

    # Make Network
    if flags.use_conv:
        ntwk = Network(CNN, flags, train_loader, test_loader)
    else:
        ntwk = Network(MLP, flags, train_loader, test_loader)
    total_param = sum(p.numel() for p in ntwk.model.parameters() if p.requires_grad)
    print("Total learning parameter is: %d"%total_param)
    
    # Training process
    print("Start training now...")
    ntwk.train()

    # Do the house keeping, write the parameters and put into folder, also use pickle to save the flags obejct
    write_flags_and_BVE(flags, ntwk)
    # put_param_into_folder(ntwk.ckpt_dir)


def get_list_comp_ind(gpu):
    """
    The function to get the list of comp index by gpu number from 0-3
    if GPU==-1, return the list of all the models
    """
    if gpu not in [0, 1, 2, 3, -1]:
        print('Your gpu index is not correct, check again')
        quit()
    data_dir = '/home/sr365/Bruce/cvdata'
    ind_list = []
    for file in os.listdir(data_dir):
        #print(file)
        # Check if this is a comp file
        if not file.endswith('.npy') or (not file[:-4].isdigit()):
            print('This file is {}, does not satisfy requirement, continue'.format(file))
            continue
        ind = int(file[:-4])
        #print('current comp ind is {}'.format(ind))
        ind_list.append(ind)
    #print(ind_list)
    length = len(ind_list)
    print(length)
    # If GPU == -1, return all list values
    if gpu == -1:
        return ind_list
    gpu_specific_list = ind_list[gpu*int(length / 4):(gpu+1)*int(length / 4)]
    print(len(gpu_specific_list))
    return gpu_specific_list
  


def train_all_models(gpu):
    """
    The aggregate training
    """
    comp_ind_list = get_list_comp_ind(gpu)
    for hidden_layer_num in [6]:
    #for hidden_layer_num in [1, 3, 5, 7]:
        for reg_scale in [0.1]:#, 1, 16]:
            for neurons in [20, 50]:#, 100, 500]:
                for comp_ind in comp_ind_list:
                    flags = flag_reader.read_flag()
                    flags.comp_ind = comp_ind
                    flags.reg_scale = reg_scale
                    flags.linear = [flags.dim_x] + [neurons for i in range(hidden_layer_num)] + [flags.dim_y]
                    flags.model_name = flags.data_set + '_every_' + str(flags.average) + '_squared_' + str(flags.square) + '_ind_' + str(flags.comp_ind) + '_complexity_{}x{}_lr_{}_decay_{}_reg_{}_bs_{}'.format(flags.linear[1], len(flags.linear) - 2, flags.lr, flags.lr_decay_rate, flags.reg_scale, flags.batch_size)
                    print(flags.model_name)
                    training_from_flag(flags)


def sweep_image_model():
    """
    The function for doing image sweeping
    """
    for c1 in [4, 8, 16]:
        for c2 in [2, 4]:
            for c3 in [2, 4]:
                for c4 in [1, 2]:
                    flags = flag_reader.read_flag()
                    print(c1)
                    flags.channel_list = c1 * np.array([1, c2, c2*c3, c2*c3*c4])
                    print('channel list = ', flags.channel_list)
                    flags.last_dim = flags.channel_list[-1]
                    flags.model_name = flags.data_set + '_channel_' + str(flags.channel_list).replace('[','').replace(']','').replace(' ','_') + \
                                        '_dim_last_' + str(flags.last_dim) + '_ind_' + str(flags.comp_ind) + \
                                        '_lr_{}_decay_{}_reg_{}_bs_{}'.format(flags.lr, flags.lr_decay_rate, flags.reg_scale, flags.batch_size)
                    print(flags.model_name)
                    training_from_flag(flags)



if __name__ == '__main__':
    # Read the parameters to be set
    flags = flag_reader.read_flag()
    # Call the train from flag function
    #training_from_flag(flags)
    sweep_image_model()
    #train_all_models(-1)
    #for i in range(4):
    #    get_list_comp_ind(i)
