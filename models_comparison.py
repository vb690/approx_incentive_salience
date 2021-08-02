import os

import pickle

from tqdm import tqdm

import numpy as np

import pandas as pd

from sklearn.model_selection import KFold, train_test_split

from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

from modules.models.supervised.baselines import MedianModel, Lag1Model
from modules.utils.data_utils.data_handlers import DataGenerator
from modules.utils.model_utils.metrics_losses import smape_k, smape_np
from modules.utils.general_utils.utilities import load_full_model
from modules.utils.general_utils.utilities import generate_3d_pad

###############################################################################

MODELS_RMP = {
    'td_enet': 'TD E-Net',
    'td_mlp': 'TD MLP',
    'rnn': 'RNN',
    'median_model': 'Median',
    'lag_1': 'Lag 1'
}
TARGETS_RMP = {
    'tar_delta_sessions': 'Future Absence',
    'tar_active_time': 'Future Active Time',
    'tar_session_time': 'Future Session Time',
    'tar_activity': 'Future Session Activity',
    'tar_sessions': 'Future N° Sessions'
}
with open('results\\saved_objects\\mappers\\context.pkl', 'rb') as pickle_file:
    CONTEXT_RMP = pickle.load(pickle_file)
CONTEXT_RMP = {value: key for key, value in CONTEXT_RMP.items()}

with open('results\\saved_objects\\scalers\\global.pkl', 'rb') as pickle_file:
    SCALER = pickle.load(pickle_file)

###############################################################################

INPUTS_PATH = 'data\\train\\inputs\\'
TARGETS_PATH = 'data\\train\\targets\\'

INPUTS = [
    'continuous_features',
    'context'
]
TARGETS = [
    'tar_delta_sessions',
    'tar_active_time',
    'tar_session_time',
    'tar_activity',
    'tar_sessions',
]

BTCH = np.array(
    [i for i in range(len(os.listdir(f'{INPUTS_PATH}context')))]
)

###############################################################################

SPLITTER = KFold(n_splits=10, shuffle=True, random_state=0)
MODELS = {
    'lag_1': None,
    'median_model': None,
    'td_enet':  'adam',
    'td_mlp': 'adam',
    'rnn': 'adam'
}
MAX_TRAIN_EPOCHS = 200

###############################################################################

GLOBAL_DF_TR = pd.read_csv(
    'data\\train\\df_train.csv',
    usecols=[
        'user_id',
        'tar_delta_sessions',
        'tar_active_time',
        'tar_session_time',
        'tar_activity',
        'tar_sessions',
        'session_order',
        'max_sess_cut',
        'context'
    ]
)
performance_dfs = []
checkpoints = []
for table_name in os.listdir('results\\tables\\models_performance'):

    checkpoints.append(int(table_name[:-4].split('_')[-1]))

if len(checkpoints) > 0:
    print(f'Checkpoints found: {checkpoints}')

fold_index = 0
for tr_vl_i, test_i in SPLITTER.split(BTCH):

    if fold_index in checkpoints:
        print(f'Skipping fold {fold_index}, checkpoint found.')
        fold_index += 1
        continue

    TR_VL_BTCH = BTCH[tr_vl_i]
    TS_BTCH = BTCH[test_i]

    TR_BTCH, VL_BTCH = train_test_split(TR_VL_BTCH, random_state=0)

    TR_GEN = DataGenerator(
        list_batches=TR_BTCH,
        inputs=INPUTS,
        targets=TARGETS,
        train=True,
        shuffle=True
    )
    VL_GEN = DataGenerator(
        list_batches=VL_BTCH,
        inputs=INPUTS,
        targets=TARGETS,
        train=True,
        shuffle=True
    )

    user_id_tr = []
    for btch in TR_VL_BTCH:

        user_id_tr.append(
            np.load(f'{TARGETS_PATH}\\user_id\\{btch}.npy')
        )

    user_id_tr = np.vstack(user_id_tr).flatten()
    DF_TR = GLOBAL_DF_TR[GLOBAL_DF_TR['user_id'].isin(user_id_tr)]

###############################################################################

    ground_truths = {
        'context': [],
        'tar_delta_sessions': [],
        'tar_active_time': [],
        'tar_session_time': [],
        'tar_activity': [],
        'tar_sessions': []
    }

    for btch in tqdm(TS_BTCH):

        for target_name in TARGETS+['context']:

            if target_name == 'context':
                array = np.load(f'{INPUTS_PATH}\\{target_name}\\{btch}.npy')
                ground_truths[target_name].append(array[:, -1])
            else:
                array = np.load(f'{TARGETS_PATH}\\{target_name}\\{btch}.npy')
                ground_truths[target_name].append(array)

    max_size = 0
    for arrays in ground_truths['tar_delta_sessions']:
        max_size += arrays.shape[0]

###############################################################################

    models_performance_df = []
    for model_name, optimizer in MODELS.items():

        K.clear_session()

        print('')
        print(f'Fitting model {model_name}')
        print('')

        if model_name == 'lag_1':
            model = Lag1Model(
                n_features=4
            )
            model.fit()
        elif model_name == 'median_model':
            model = MedianModel(
                targets_columns=[
                    'tar_delta_sessions',
                    'tar_active_time',
                    'tar_session_time',
                    'tar_activity',
                    'tar_sessions'
                ],
                time_column='session_order',
                contexts_column='context'
            )
            model.fit(
                DF_TR,
                gamma=0.5
            )
        else:
            model = load_full_model(
                name=model_name,
                optimizer=optimizer,
                custom_objects={'smape_k': smape_k},
                loss={
                    'output_absence_act': smape_k,
                    'output_active_act': smape_k,
                    'output_sess_time_act': smape_k,
                    'output_activity_act': smape_k,
                    'output_sess_act': smape_k
                },
                metrics={
                    'output_absence_act': smape_k,
                    'output_active_act': smape_k,
                    'output_sess_time_act': smape_k,
                    'output_activity_act': smape_k,
                    'output_sess_act': smape_k
                }
            )
            ES = EarlyStopping(
                monitor='val_loss',
                min_delta=0.0001,
                patience=5,
                verbose=1,
                mode='auto',
                restore_best_weights=True
            )

            print(model._model.summary())

            model.fit(
                x=TR_GEN,
                validation_data=VL_GEN,
                epochs=MAX_TRAIN_EPOCHS,
                verbose=2,
                callbacks=[ES],
                workers=8,
                max_queue_size=20
            )

###############################################################################

        print('')
        print(f'Inference model {model_name}')
        print('')

        predictions = {
            'tar_delta_sessions': [],
            'tar_active_time': [],
            'tar_session_time': [],
            'tar_activity': [],
            'tar_sessions': []
            }

        for btch in tqdm(TS_BTCH):

            input_features = []

            for inp in INPUTS:

                array = np.load(f'{INPUTS_PATH}{inp}\\{btch}.npy')
                input_features.append(array)

            btch_context = np.load(
                f'data\\train\\inputs\\context\\{btch}.npy')

            if model_name == 'lag_1':
                original_shape = input_features[0].shape
                features = input_features[0].reshape((-1, 4))
                features = SCALER.inverse_transform(features)
                features = features.reshape(original_shape)

                list_pred = model._model.predict(
                    features,
                    batch_size=array.shape[0]
                )
            elif model_name == 'median_model':
                list_pred = model.predict(
                    X=input_features[0],
                    X_contexts=input_features[1][:, 0].flatten()
                )
            else:
                list_pred = model._model.predict(
                    input_features,
                    batch_size=array.shape[0]
                )

            predictions['tar_delta_sessions'].append(list_pred[0])
            predictions['tar_active_time'].append(list_pred[1])
            predictions['tar_session_time'].append(list_pred[2])
            predictions['tar_activity'].append(list_pred[3])
            predictions['tar_sessions'].append(list_pred[4])


###############################################################################

        print('')
        print(f'Comparison model {model_name}')
        print('')

        array_cont = np.hstack(ground_truths['context'])
        unique_context = np.unique(array_cont)
        performance_df = []
        for context in unique_context:

            for target_name in TARGETS:

                if target_name == 'context':
                    continue

                ind = np.argwhere(array_cont == context).flatten()
                partial_results = pd.DataFrame(
                    columns=['context', 'metric', 'target', 'value']
                )

                padded_tar = generate_3d_pad(
                    ground_truths[target_name],
                    shape=(max_size, 20, 1),
                    pad=np.nan
                )
                padded_pred = generate_3d_pad(
                    predictions[target_name],
                    shape=(max_size, 20, 1),
                    pad=np.nan
                )

                error = smape_np(
                    padded_tar[ind, :, :],
                    padded_pred[ind, :, :],
                    axis=-1
                )

                std = np.nanstd(error, axis=0)
                mean = np.nanmean(error, axis=0)

                partial_results['value'] = mean
                partial_results['metric'] = 'SMAPE'
                partial_results['Model'] = MODELS_RMP[model_name]
                partial_results['context'] = CONTEXT_RMP[context]
                partial_results['target'] = TARGETS_RMP[target_name]
                partial_results['Session'] = [
                    i for i in range(2, len(mean) + 2)
                ]
                partial_results['fitting_time'] = model.get_fitting_time()
                partial_results['parameters'] = model.get_para_count()
                partial_results['epochs'] = model.get_n_epochs()

                performance_df.append(partial_results)

        performance_df = pd.concat(performance_df, ignore_index=True)
        performance_df = performance_df.dropna()
        performance_df['fold_n'] = fold_index
        models_performance_df.append(performance_df)

        K.clear_session()

    models_performance_df = pd.concat(models_performance_df, ignore_index=True)
    models_performance_df.to_csv(
        f'results\\tables\\models_performance\\models_performance_fold_{fold_index}.csv',
        index=False
    )

    fold_index += 1
