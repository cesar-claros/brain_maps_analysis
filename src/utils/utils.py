"""
Utility functions.
"""

# import pathlib
import os
import random
import nibabel as nib
import tensorflow as tf
# from tensorflow.keras import layers
# from scipy.interpolate import RegularGridInterpolator
# from tensorflow.keras.callbacks import Callback
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import OneHotEncoder
# from keras.utils.vis_utils import plot_model
# from sklearn.metrics import mean_absolute_error as mae
# from numpy.random import seed
import pandas as pd
# import sklearn
# import sklearn.model_selection
import numpy as np
# import matplotlib.pyplot as plt
# from kerashypetune import KerasRandomSearchCV


def common_path(arr, pos='prefix'):
    """This function finds the longest common prefix and suffix given a list of files."""
    # The longest common prefix of an empty array is "".
    if not arr:
        print("Longest common", pos, ":", "")
    # The longest common prefix of an array containing
    # only one element is that element itself.
    elif len(arr) == 1:
        print("Longest common", pos, ":", str(arr[0]))
    else:
        path_string = range(len(arr[0])) if pos=="prefix" else range(-1,-len(arr[0])+1,-1)
        # Sort the array
        arr.sort()
        result = ""
        # Compare the first and the last string character
        # by character.
        for i in path_string:
            #  If the characters match, append the character to
            #  the result.
            if arr[0][i] == arr[-1][i]:
                result += arr[0][i]
            # Else, stop the comparison
            else:
                break
    if pos=="suffix":
        result = result[::-1]
    print("Longest common", pos, ":", result)
    return result

def read_files(data_folder_path, label_folder_path, set_id, only_map=False):
    """This function reads all the .nib files contained in a given folder."""
    labels = pd.read_csv(label_folder_path+'labels_final.csv')
    labels_list = []
    map_list = []
    sex_list = []
    study_list = []
    meta_list = []
    for _, _, files in os.walk(data_folder_path):
        common_prefix = common_path(files, pos="prefix")
        common_suffix = common_path(files, pos="suffix")
        for id_number in set_id:
            age =  labels.loc[labels["ID"] == id_number,'Age'].to_numpy()[0]
            sex =  labels.loc[labels["ID"] == id_number,'Antipodal_Sex'].to_numpy()[0]
            study = labels.loc[labels["ID"] == id_number,'Study_ID'].to_numpy()[0]
            filename = common_prefix + str(id_number) + common_suffix
            try:
                nib_raw = nib.load(data_folder_path + filename)
            except FileNotFoundError:
                # filename = common_prefix + '{:0>3d}'.format(id_number) + common_suffix
                filename = common_prefix + f'{id_number:0>3d}' + common_suffix
                try:
                    nib_raw = nib.load(data_folder_path + filename)
                except FileNotFoundError:
                    print(id_number)
                    continue
            meta = nib_raw.header
            brain_map = nib_raw.get_fdata()[:,:,:]
            labels_list.append(age)
            sex_list.append(sex)
            map_list.append(brain_map)
            study_list.append(study)
            meta_list.append(meta)
    X_map = np.array(map_list).astype(np.float32)
    X_sex = np.array(sex_list)
    X_study = np.array(study_list)
    y = np.array(labels_list).astype(np.float32)
    m = np.array(meta_list)
    if only_map:
        output = X_map
    else:
        output = (X_map, X_sex, X_study, y, m)
    return output

def seed_everything(seed=1234):
    """This function sets the seeds to get consistent results."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
