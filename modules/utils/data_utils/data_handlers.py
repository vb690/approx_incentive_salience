import gc

import numpy as np
import pandas as pd

from tensorflow.keras.utils import Sequence

from ..general_utils.utilities import generate_dir, save_objects
from ..data_utils.data_preprocessers import preprocessing_df


class DataGenerator(Sequence):
    """
    Class implementing a data generator
    """
    def __init__(self, list_batches, inputs, targets,
                 train=True, shuffle=True):
        """
        """
        self.list_batches = list_batches
        self.inputs = inputs
        self.targets = targets
        self.shuffle = shuffle
        self.root_dir = 'data\\train' if train else 'data\\test'
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch'
        """
        return int(len(self.list_batches))

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Pick a batch
        batch = self.list_batches[index]
        # Generate X and y
        X, y = self.__data_generation(batch)
        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        if self.shuffle is True:
            np.random.shuffle(self.list_batches)

    def __data_generation(self, batch):
        """Generates data containing batch_size samples"""
        X = []
        y = []
        for subdir in self.inputs:

            X.append(
                np.load('{}\\inputs\\{}\\{}.npy'.format(
                    self.root_dir, subdir, batch
                    )
                )
            )

        for subdir in self.targets:

            y.append(
                np.load('{}\\targets\\{}\\{}.npy'.format(
                    self.root_dir, subdir, batch
                    )
                )
            )

        return X, y


def create_features_batches(df, features_keys, train=True, id_key='user_id',
                            sorting_keys=['user_id', 'session_order'],
                            grouping_key='max_sess_cut', batch_size=256):
    """
    """
    root_dir = 'data\\train' if train else 'data\\test'
    df = df.sort_values(sorting_keys)

    batch_index_filename = 0
    for key, group in df.groupby(grouping_key):

        unique_ids = len(group[id_key].unique())
        # generate array of behavioural features
        generate_dir(f'{root_dir}\\inputs\\continuous_features')
        array = np.array(group[features_keys])
        try:
            array = array.reshape((unique_ids, key, len(features_keys)))
        except:
            group.to_csv('fucked_up.csv')
        array = array[:, :-1, :]
        num_batches = (array.shape[0] + batch_size - 1) // batch_size
        print('Dumping group {}'.format(key))
        print('With {} unique ids'.format(unique_ids))

        for batch_index in range(num_batches):

            minimum = min(array.shape[0], (batch_index + 1) * batch_size)
            batch = array[batch_index * batch_size: minimum]
            batch = np.float32(batch)
            np.save(
                '{}\\inputs\\continuous_features\\{}.npy'.format(
                    root_dir,
                    batch_index_filename
                ),
                arr=batch
            )
            batch_index_filename += 1
            gc.collect()


def create_embedding_batches(df, embeddings_keys, train=True, id_key='user_id',
                             sorting_keys=['user_id', 'session_order'],
                             grouping_key='max_sess_cut', batch_size=256):
    """
    """
    root_dir = 'data\\train' if train else 'data\\test'
    df = df.sort_values(sorting_keys)

    batch_index_filename = 0
    for key, group in df.groupby(grouping_key):

        unique_ids = len(group[id_key].unique())
        arrays = {}

        for embedding in embeddings_keys:

            generate_dir('{}\\inputs\\{}'.format(root_dir, embedding))
            array = np.array(group[embedding])
            array = array.reshape((unique_ids, key))
            array = array[:, :-1]
            num_batches = (array.shape[0] + batch_size - 1) // batch_size
            arrays[embedding] = array
            gc.collect()

        print('Dumping group {}'.format(key))
        print('With {} unique ids'.format(unique_ids))
        for batch_index in range(num_batches):

            for embedding, array in arrays.items():

                minimum = min(array.shape[0], (batch_index + 1) * batch_size)
                batch = array[batch_index * batch_size: minimum]
                batch = np.float32(batch)
                np.save(
                    '{}\\inputs\\{}\\{}.npy'.format(
                        root_dir,
                        embedding,
                        batch_index_filename
                    ),
                    arr=batch
                )
                gc.collect()

            batch_index_filename += 1


def create_targets_batches(df, targets_keys, train=True, id_key='user_id',
                           sorting_keys=['user_id', 'session_order'],
                           grouping_key='max_sess_cut', batch_size=256):
    """
    """
    root_dir = 'data\\train' if train else 'data\\test'
    df = df.sort_values(sorting_keys)

    batch_index_filename = 0
    for key, group in df.groupby(grouping_key):

        unique_ids = len(group[id_key].unique())
        arrays = {}

        for target in targets_keys:

            generate_dir('{}\\targets\\{}'.format(root_dir, target))
            array = np.array(group[target])
            array = array.reshape((unique_ids, key, 1))
            array = array[:, :-1, :]
            if target == 'user_id':
                array = array[:, 0, :]
            num_batches = (array.shape[0] + batch_size - 1) // batch_size
            arrays[target] = array
            gc.collect()

        print('Dumping group {}'.format(key))
        print('With {} unique ids'.format(unique_ids))
        for batch_index in range(num_batches):

            for target, array in arrays.items():

                minimum = min(array.shape[0], (batch_index + 1) * batch_size)
                batch = array[batch_index * batch_size: minimum]
                if target == 'user_id':
                    batch = batch.astype(str)
                else:
                    batch = np.float32(batch)
                np.save(
                    '{}\\targets\\{}\\{}.npy'.format(
                        root_dir,
                        target,
                        batch_index_filename
                    ),
                    arr=batch
                )
                gc.collect()

            batch_index_filename += 1


def data_handling_pipeline(games_list, features_keys, targets_keys,
                           embeddings_keys, scaler, global_scaling=True,
                           id_key='user_id',
                           grouping_key='max_sess_cut',
                           sorting_keys=['user_id', 'session_order'],
                           train_size=0.8, batch_size=256):
    """
    """
    global_df = []
    df_tr = []
    df_ts = []
    scalers = {}
    for game in games_list:

        print('Handling game {}'.format(game))
        df = pd.read_csv('data\\csv\\cleaned\\{}.csv'.format(game))
        df = df.sort_values(sorting_keys)

        if global_scaling:
            global_df.append(df)
        else:
            tr, ts, fit_scaler = preprocessing_df(
                df=df,
                features_keys=features_keys,
                scaler=scaler(),
                train_size=train_size
            )
            tr = tr.sort_values(sorting_keys)
            ts = ts.sort_values(sorting_keys)
            df_tr.append(tr)
            df_ts.append(ts)
            scalers[game] = fit_scaler

    if global_scaling:
        global_df = pd.concat(global_df, ignore_index=True)
        global_df = global_df.sort_values(sorting_keys)
        df_tr, df_ts, fit_scaler = preprocessing_df(
            df=global_df,
            features_keys=features_keys,
            scaler=scaler(),
            train_size=train_size
        )
        df_tr = df_tr.sort_values(sorting_keys)
        df_tr.to_csv('data\\train\\df_train.csv', index=False)
        df_ts = df_ts.sort_values(sorting_keys)
        df_ts.to_csv('data\\test\\df_test.csv', index=False)
        scalers['global'] = fit_scaler
    else:
        df_tr = pd.concat(df_tr, ignore_index=True)
        df_tr = df_tr.sort_values(sorting_keys)
        df_tr.to_csv('data\\train\\df_train.csv', index=False)

        df_ts = pd.concat(df_ts, ignore_index=True)
        df_ts = df_ts.sort_values(sorting_keys)
        df_ts.to_csv('data\\test\\df_test.csv', index=False)

    save_objects(
        objects=scalers,
        dir_name='saved_objects\\scalers'
    )

    mappers = {}
    for embedding in embeddings_keys:

        unique_values = df_tr[embedding].unique()
        # zero needs to be sved for the missing values
        mapper = {value: code for code, value in enumerate(unique_values, 1)}
        df_tr[embedding] = df_tr[embedding].map(mapper)
        df_ts[embedding] = df_ts[embedding].map(mapper)
        df_ts = df_ts.fillna(0)
        mappers[embedding] = mapper

    save_objects(
        objects=mappers,
        dir_name='saved_objects\\mappers'
    )

    create_features_batches(
        df=df_tr,
        features_keys=features_keys,
        train=True,
        id_key=id_key,
        sorting_keys=sorting_keys,
        grouping_key=grouping_key,
        batch_size=256
    )

    create_embedding_batches(
        df=df_tr,
        embeddings_keys=embeddings_keys,
        train=True,
        id_key=id_key,
        sorting_keys=sorting_keys,
        grouping_key=grouping_key,
        batch_size=256
    )

    create_targets_batches(
        df=df_tr,
        targets_keys=targets_keys,
        train=True,
        id_key=id_key,
        sorting_keys=sorting_keys,
        grouping_key=grouping_key,
        batch_size=256
    )

    create_features_batches(
        df=df_ts,
        features_keys=features_keys,
        train=False,
        id_key=id_key,
        sorting_keys=sorting_keys,
        grouping_key=grouping_key,
        batch_size=256
    )

    create_embedding_batches(
        df=df_ts,
        embeddings_keys=embeddings_keys,
        train=False,
        id_key=id_key,
        sorting_keys=sorting_keys,
        grouping_key=grouping_key,
        batch_size=256
    )

    create_targets_batches(
        df=df_ts,
        targets_keys=targets_keys,
        train=False,
        id_key=id_key,
        sorting_keys=sorting_keys,
        grouping_key=grouping_key,
        batch_size=256
    )
