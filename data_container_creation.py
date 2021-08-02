import os

import pickle

from tqdm import tqdm

import numpy as np

from tensorflow.keras.optimizers import Adam

from modules.utils.model_utils.metrics_losses import smape_k, smape_np
from modules.utils.general_utils.utilities import load_full_model
from modules.utils.general_utils.utilities import generate_exp_decay_weights

###############################################################################

INPUTS_PATH = 'data\\train\\inputs\\{}'
TARGETS_PATH = 'data\\train\\targets\\{}'

BTCH = [
    i for i in range(
        len(os.listdir(INPUTS_PATH.format('continuous_features')))
    )
]
BTCH = BTCH[0::5]

SNAPSHOTS = 10

USER_PATH = TARGETS_PATH.format('user_id\\{}.npy')
CONTEXT_PATH = INPUTS_PATH.format('context\\{}.npy')

INPUTS = ['continuous_features', 'context']
TARGETS = [
    'tar_delta_sessions',
    'tar_active_time',
    'tar_session_time',
    'tar_activity',
    'tar_sessions'
]
INPUTS_METRIC = [
    'delta_sessions',
    'active_time',
    'session_time',
    'activity'
]

MODEL = load_full_model(
    name='rnn',
    optimizer=Adam(),
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
    },
    path='results\\saved_trained_models\\{}'
)

with open('results\\saved_objects\\scalers\\global.pkl', 'rb') as pickle_file:
    SCALER = pickle.load(pickle_file)

DATA_CONTAINER = {}

###############################################################################
inputs_temporal = {
    input_metric: {
        snapshot: [] for snapshot in range(SNAPSHOTS)
    } for input_metric in INPUTS_METRIC
}

# not discounted
predictions_temporal = {
    target_name: {
        snapshot: [] for snapshot in range(SNAPSHOTS)
    } for target_name in TARGETS
}
ground_truths_temporal = {
    target_name: {
        snapshot: [] for snapshot in range(SNAPSHOTS)
    } for target_name in TARGETS
}

# discounted
predictions_temporal_ds = {
    target_name: {
        snapshot: [] for snapshot in range(SNAPSHOTS)
    } for target_name in TARGETS
}
ground_truths_temporal_ds = {
    target_name: {
        snapshot: [] for snapshot in range(SNAPSHOTS)
    } for target_name in TARGETS
}

errors_temporal = {
    target_name: {
        snapshot: [] for snapshot in range(SNAPSHOTS)
    } for target_name in TARGETS
}

users_temporal = {snapshot: [] for snapshot in range(SNAPSHOTS)}
contexts_temporal = {snapshot: [] for snapshot in range(SNAPSHOTS)}


for btch in tqdm(BTCH):

    input_features = []

    for inp in INPUTS:

        array = np.load(f'{INPUTS_PATH.format(inp)}\\{btch}.npy')
        input_features.append(array)

    list_pred = MODEL._model.predict(
        input_features,
        batch_size=array.shape[0]
    )
    total_snapshots = array.shape[1]
    if total_snapshots > 10:
        total_snapshots = 10

    context_array = np.load(f'data\\train\\inputs\\context\\{btch}.npy')
    user_id_array = np.load(f'data\\train\\targets\\user_id\\{btch}.npy')

    for target_index, target_name in enumerate(TARGETS):

        prediction_array = list_pred[target_index]
        ground_truth_array = np.load(
            f'{TARGETS_PATH.format(target_name)}\\{btch}.npy'
        )
        error_array = smape_np(ground_truth_array, prediction_array, axis=2)

        for snapshot in range(total_snapshots):

            # input metrics
            if target_name != 'tar_sessions':
                input_name = INPUTS_METRIC[target_index]
                input_metrics = input_features[0]
                input_metrics_shape = input_metrics.shape
                input_metrics = input_metrics.reshape(
                    (-1, input_metrics_shape[2])
                )
                input_metrics = SCALER.inverse_transform(
                    input_metrics
                )
                input_metrics = input_metrics.reshape(input_metrics_shape)
                inputs_temporal[input_name][snapshot].append(
                    input_metrics[:, :snapshot+1, target_index]
                )

            weights = generate_exp_decay_weights(
                prediction_array[:, snapshot:, :].shape[1],
            )

            # predictions not discounted sum
            predictions_temporal[target_name][snapshot].append(
                prediction_array[:, snapshot, :]
            )

            # predictions discounted sum
            discounted_sum_predictions = \
                prediction_array[:, snapshot:, :] * weights[:, np.newaxis]
            discounted_sum_predictions = \
                discounted_sum_predictions.sum(axis=1)
            predictions_temporal_ds[target_name][snapshot].append(
                discounted_sum_predictions
            )

            # ground_truth not discounted
            ground_truths_temporal[target_name][snapshot].append(
                ground_truth_array[:, snapshot, :]
            )
            # ground truths discounted sum
            discounted_sum_ground_truth = \
                ground_truth_array[:, snapshot:, :] * weights[:, np.newaxis]
            discounted_sum_ground_truth = \
                discounted_sum_ground_truth.sum(axis=1)
            ground_truths_temporal_ds[target_name][snapshot].append(
                discounted_sum_ground_truth
            )

            # errors
            errors_temporal[target_name][snapshot].append(
                error_array[:, snapshot]
            )

    for snapshot in range(total_snapshots):

        contexts_temporal[snapshot].append(context_array[:, snapshot])
        users_temporal[snapshot].append(user_id_array[:, 0])

###############################################################################

for snapshot in range(SNAPSHOTS):

    contexts_temporal[snapshot] = np.hstack(contexts_temporal[snapshot])
    users_temporal[snapshot] = np.hstack(users_temporal[snapshot])

    for target_index, target_name in enumerate(TARGETS):

        if target_name != 'tar_sessions':
            input_name = INPUTS_METRIC[target_index]
            inputs_temporal[input_name][snapshot] = np.vstack(
                inputs_temporal[input_name][snapshot]
            )

        predictions_temporal[target_name][snapshot] = np.vstack(
            predictions_temporal[target_name][snapshot]
        )
        ground_truths_temporal[target_name][snapshot] = np.vstack(
            ground_truths_temporal[target_name][snapshot]
        )
        predictions_temporal_ds[target_name][snapshot] = np.vstack(
            predictions_temporal_ds[target_name][snapshot]
        )
        ground_truths_temporal_ds[target_name][snapshot] = np.vstack(
            ground_truths_temporal_ds[target_name][snapshot]
        )
        errors_temporal[target_name][snapshot] = np.hstack(
            errors_temporal[target_name][snapshot]
        )

###############################################################################

DATA_CONTAINER['input_metrics'] = inputs_temporal
DATA_CONTAINER['prediction_ds'] = predictions_temporal_ds
DATA_CONTAINER['ground_truth_ds'] = ground_truths_temporal_ds
DATA_CONTAINER['prediction'] = predictions_temporal
DATA_CONTAINER['ground_truth'] = ground_truths_temporal
DATA_CONTAINER['error'] = errors_temporal
DATA_CONTAINER['context'] = contexts_temporal
DATA_CONTAINER['user_id'] = users_temporal

with open('results\\saved_data_containers\\rnn.pkl', 'wb') as container:
    pickle.dump(DATA_CONTAINER, container, pickle.HIGHEST_PROTOCOL)
