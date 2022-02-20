import os
from copy import deepcopy

import pickle

import numpy as np
from scipy import stats

from sklearn.preprocessing import QuantileTransformer, KBinsDiscretizer
from skimage.exposure import equalize_hist

from tensorflow.keras.models import load_model
import tensorflow as tf


def generate_dir(path):
    '''
    Function checking the existence of a directory and generating it
    if not present.
    '''
    if not os.path.exists(path):
        os.makedirs(path)


def top_k_variance(df, columns, k=50, no_variance_filter=True):
    """Utility function for retaining top k features with the highest variance.
    """
    X = df[columns].values
    columns = np.array(columns)
    var = np.var(X, axis=0)
    if no_variance_filter:
        mask = var != 0
        columns = columns[mask]
    sorted_var = np.sort(var)[::-1]
    threshold = sorted_var[k - 1]

    mask = var <= threshold
    columns_to_drop = columns[mask]
    # df = df.drop(columns_to_drop, axis=1)
    return df, columns_to_drop


def group_wise_binning(array, n_bins, grouper=None, method=None, **kwargs):
    '''
    '''
    def binning(array):
        '''
        '''
        ranked = stats.rankdata(array)
        data_percentile = ranked/len(array)*100
        binned = np.digitize(
            data_percentile,
            [i for i in range(1, n_bins+1)],
            right=True
        )
        return binned

    array = deepcopy(array)
    grouper = deepcopy(grouper)
    listed_input = False

    if isinstance(array, list):
        lengths = [len(a.flatten()) for a in array]
        array = np.hstack([a.flatten() for a in array])
        grouper = np.hstack([g.flatten() for g in grouper])
        listed_input = True

    array = array.flatten()
    if grouper is None:
        grouper = np.zeros(shape=(len(array)))
    else:
        grouper = np.array(grouper)
    for unique_group in np.unique(grouper):

        indices = np.argwhere(grouper == unique_group).flatten()
        if method == 'eq_hist':
            array[indices] = equalize_hist(
                array[indices],
                nbins=n_bins
            )
        elif method == 'quant_uni':
            array[indices] = QuantileTransformer(
                **kwargs
            ).fit_transform(array[indices].reshape(-1, 1)).flatten()
        elif method == 'quant_norm':
            array[indices] = QuantileTransformer(
                output_distribution='normal',
                **kwargs
            ).fit_transform(array[indices].reshape(-1, 1)).flatten()
        elif method == 'discret':
            array[indices] = KBinsDiscretizer(
                n_bins=n_bins,
                encode='ordinal',
                **kwargs
            ).fit_transform(array[indices].reshape(-1, 1)).flatten()
        else:
            array[indices] = binning(array[indices])

    if listed_input:
        new_array = []
        current_index = 0
        for length in lengths:

            new_array.append(array[current_index:current_index + length])
            current_index += length

        return new_array

    return array


def generate_3d_pad(list_arrays, shape, pad=np.nan):
    '''
    '''
    padded_array = np.empty(shape=shape)
    padded_array[:] = pad
    index = 0
    for array in list_arrays:

        size = array.shape[0]
        length = array.shape[1]
        padded_array[index:size + index, 0:length, :] = array
        index += size

    return padded_array


def generate_exp_decay_weights(length, gamma=0.1):
    """
    """
    weights = []
    for t in range(length):

        weights.append(gamma ** t)

    weights = np.array(weights)
    return weights


def save_arrays(arrays, dir_name):
    '''
    Function for loading saved arrays, this assume the existence of a local
    directory named data

    Arguments:
        - arrays: a list of strings, specifying the name of the arrays
                  to be loaded
        - dir_name: a string, specifying the directory where the
                    arrays are locatdd

    Returs:
        - loaded_arrays: a dictionry where keys are the identifiers of the
                         arrays and value are the loaded arrays
    '''
    save_dir = 'data\\{}'.format(dir_name)
    generate_dir(save_dir)
    for name, array in arrays.items():

        path = '{}\\{}.npy'.format(save_dir, name)
        np.save(array, path, allow_pickle=True)


def load_arrays(arrays, dir_name):
    '''
    Function for loading saved arrays, this assume the existence of a local
    directory named data

    Arguments:
        - arrays: a list of strings, specifying the name of the arrays
                  to be loaded
        - dir_name: a string, specifying the directory where the
                    arrays are locatdd

    Returs:
        - loaded_arrays: a dictionry where keys are the identifiers of the
                         arrays and value are the loaded arrays
    '''
    load_dir = 'data\\{}'.format(dir_name)
    loaded_arrays = {}
    for array in arrays:

        path = '{}\\{}.npy'.format(load_dir, array)
        loaded_array = np.load(path, allow_pickle=True)
        loaded_arrays[array] = loaded_array

    return loaded_arrays


def save_objects(objects, dir_name):
    '''
    Function for saving obejcts, this assume the existence of a local
    directory named results

    Arguments:
        - objects: a dictionary, specifying the name of the objets and the
                   relative objects to be saved
        - dir_name: a string, specifying the directory where the
                    objects will be locatdd

    Returs:
        - loaded_arrays: a dictionry where keys are the identifiers of the
                         arrays and value are the loaded arrays
    '''
    save_dir = 'results\\{}'.format(dir_name)
    generate_dir(save_dir)
    for name, obj in objects.items():

        path = '{}\\{}.pkl'.format(save_dir, name)

        with open(path, 'wb') as out:
            pickle.dump(obj, out, pickle.HIGHEST_PROTOCOL)


def load_objects(objects, dir_name):
    '''
    Function for saving obejcts, this assume the existence of a local
    directory named results

    Arguments:
        - objects: a list, specifying the names of the objets that need to be
                   loaded
        - dir_name: a string, specifying the directory where the
                    objects are locatdd

    Returs:
        - loaded_arrays: a dictionry where keys are the identifiers of the
                         objects and value are the loaded objects
    '''
    load_dir = 'results\\{}'.format(dir_name)
    loaded_objects = {}
    for obj in objects:

        path = '{}\\{}.pkl'.format(load_dir, obj)
        with open(path, 'rb') as inp:
            loaded_obj = pickle.load(inp)
            loaded_objects[obj] = loaded_obj

    return loaded_objects


def save_full_model(model, path='results\\saved_models\\{}'):
    '''
    '''
    name = model.get_model_tag()
    path = path.format(name)
    generate_dir(path)

    keras_model = model.get_model()
    tf.saved_model.save(
        keras_model,
        '{}\\engine\\'.format(path)
    )

    with open('{}\\scaffolding.pkl'.format(path), 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)


def load_full_model(name, custom_objects=None,
                    path='results\\saved_models\\{}', **compile_schema):
    '''
    '''
    path = path.format(name)

    keras_model = load_model(
        '{}\\engine\\'.format(path),
        custom_objects=custom_objects,
        compile=False
    )
    keras_model.compile(
        **compile_schema
    )
    with open('{}\\scaffolding.pkl'.format(path), 'rb') as input:
        model = pickle.load(input)

    try:
        model.set_model(keras_model)
    except Exception:
        import types

        def set_model(self, model):
            setattr(self, '_model', model)
            setattr(self, 'n_parameters', model.count_params())

        model.set_model = types.MethodType(set_model, model)
        model.set_model(keras_model)

    return model
