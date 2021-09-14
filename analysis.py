import numpy as np
import pandas as pd
import os
from flag_reader import load_flags
import glob
import shutil

def analze_runs(mother_dir='models/'):
    """
    This function puts the same run into the same folder
    """

    # First get the list of run params
    run_param_list = []
    for folder in os.listdir(mother_dir):
        cur_folder = os.path.join(mother_dir, folder)
        # print(cur_folder)
        # If this is not a run folder, skip
        if not os.path.isdir(cur_folder) or not folder.startswith('bruce'):
            continue
        run_name = folder.split('complexity_')[-1]
        print(run_name)
        # Check whether this fun is already included and put that there
        if not run_name in run_param_list:
            run_param_list.append(run_name)
    print('The run param list is ', run_param_list)
    
    # Now, with a list of run params, get those id and 
    for run_param in run_param_list:
        result_name = os.path.join(mother_dir, run_param + '.csv')
        folder_list = [ a for a in glob.glob(os.path.join(mother_dir, '*' + run_param)) if os.path.isdir(a) ]
        print('len of folder_list = ', len(folder_list))
        result_mat = []
        # Loop over the folders
        for folder in folder_list:
            # Skip if this folder does not have a flags.obj
            if not os.path.isfile(os.path.join(folder, 'flags.obj')):
                # Remove this entry since we are going to delete the whole list afterwards
                folder_list.remove(folder)
                continue
            flag = load_flags(folder)
            comp_ind, insample, outsample = flag.comp_ind, flag.best_training_loss, flag.best_validation_loss
            result_mat.append([comp_ind, insample, outsample])
        print('shape of result mat is ', np.shape(result_mat))
        np.savetxt(result_name, result_mat, delimiter=',')

        ## Deleting those folders
        #for folder in folder_list:
        #    if 'model' in folder:
        #        #print('removing file')
        #        shutil.rmtree(folder)

        #quit()
        #with open(result_name, 'a') as fout:
        #    fout.write('comp_ind, insample_mse, outsample_mse')

if __name__ == '__main__':
    analze_runs()   
