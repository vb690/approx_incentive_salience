from tqdm import tqdm

import time

import numpy as np

import pandas as pd

from tensorflow.keras.layers import Input, Dense, Embedding, Lambda, Reshape
from tensorflow.keras.layers import TimeDistributed, Activation, Concatenate
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.models import Model
from tensorflow.keras.backend import ones_like

from tensorflow.keras.optimizers import Adam

from ...utils.model_utils.metrics_losses import smape_k
from ...utils.model_utils.abstract_models import _AbstractHyperEstimator


class Lag1Model(_AbstractHyperEstimator):
    """
    """
    def __init__(self, n_features, model_tag=None):
        """
        """
        self.n_features = n_features
        if model_tag is None:
            self.model_tag = 'lag_1'
        else:
            self.model_tag = model_tag
        self.prob = False
        self.n_parameters = 1
        self._model = self.__build()

    def __build(self):
        """
        """
        feat_input = Input(
            shape=(None, self.n_features),
            name='features_input'
        )

        absence = Lambda(lambda x: x[:, :, 0])(feat_input)
        absence = Reshape(
            (-1, 1),
            name='output_absence_act'
        )(absence)
        active = Lambda(lambda x: x[:, :, 1])(feat_input)
        active = Reshape(
            (-1, 1),
            name='output_active_act'
        )(active)
        sess_time = Lambda(lambda x: x[:, :, 2])(feat_input)
        sess_time = Reshape(
            (-1, 1),
            name='output_sess_time_act'
        )(sess_time)
        activity = Lambda(lambda x: x[:, :, 3])(feat_input)
        activity = Reshape(
            (-1, 1),
            name='output_activity_act'
        )(activity)
        sess = Lambda(lambda x: x[:, :, 3])(feat_input)
        sess = Lambda(lambda x: ones_like(x))(sess)
        sess = Reshape(
            (-1, 1),
            name='output_sess_act'
        )(sess)

        model = Model(
            inputs=feat_input,
            outputs=[
                absence,
                active,
                sess_time,
                activity,
                sess
            ]
        )
        model.compile(
            optimizer=Adam(),
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

        return model

    def fit(self, **kwargs):
        """
        """
        start = time.time()
        end = time.time()
        setattr(self, 'fitting_time', end - start)
        setattr(self, 'n_epochs', 1)
        return None


class MedianModel(_AbstractHyperEstimator):
    """
    """
    def __init__(self, targets_columns, time_column, contexts_column,
                 id_column='user_id',
                 cont_mapper='results\\saved_objects\\mappers\\context.pkl'):
        """
        """
        self.targets_columns = targets_columns
        self.time_column = time_column
        self.contexts_column = contexts_column
        self.cont_mapper = pd.read_pickle(cont_mapper)
        self.id_column = id_column
        self.n_parameters = 1
        self.model_tag = 'median_model'

    @staticmethod
    def generate_decay_weights(length, gamma=0.5):
        """
        """
        weights = [1]
        for t in range(length - 1):

            weights.append(weights[t] * gamma)

        return weights

    def fit(self, train_df, gamma=None):
        """
        """
        start = time.time()

        for unique_context in tqdm(train_df[self.contexts_column].unique()):

            unique_context = str(unique_context)
            setattr(
                self,
                str(self.cont_mapper[unique_context]),
                {}
            )
            con_df = train_df[train_df[self.contexts_column] == unique_context]

            for time_step in con_df[self.time_column].unique():
                if gamma is not None:
                    time_slice = con_df[
                        con_df['max_sess_cut'] >= time_step
                    ]
                    time_slice = time_slice[
                        time_slice[self.time_column] <= time_step
                    ]
                    time_slice = time_slice.sort_values(
                        [self.id_column, self.time_column]
                    )
                else:
                    time_slice = con_df[con_df[self.time_column] == time_step]

                targets_averages = []

                for target in self.targets_columns:

                    if gamma is not None:
                        unique_ids = len(time_slice[self.id_column].unique())
                        target_slice = time_slice[target]
                        target_slice = target_slice.values
                        target_slice = target_slice.reshape(
                            (unique_ids, time_step)
                        )
                        weights = self.generate_decay_weights(
                            length=target_slice.shape[1],
                            gamma=gamma
                        )
                        time_average = np.average(
                                target_slice,
                                axis=1,
                                weights=weights
                            )
                        targets_averages.append(np.median(time_average))
                    else:
                        targets_averages.append(np.median(time_slice[target]))

                get_context = getattr(
                    self,
                    str(self.cont_mapper[unique_context])
                )
                get_context[time_step] = targets_averages

        end = time.time()
        setattr(self, 'fitting_time', end - start)
        setattr(self, 'n_epochs', 1)

    def predict(self, X, X_contexts):
        """
        """
        predictions = [
            np.empty(shape=(X.shape[0], X.shape[1], 1))
            for target in range(len(self.targets_columns))
        ]
        time_steps = [i for i in range(1, X.shape[1] + 1)]

        for index, context in enumerate(X_contexts):

            context_pred = getattr(self, str(int(context)))
            row = [context_pred[time_step] for time_step in time_steps]
            row = np.array(row)
            row = row.reshape(row.shape[0], row.shape[1], 1)
            for target in range(row.shape[1]):

                predictions[target][index, :, :] = row[:, target, :]

        return predictions


class TimeDistributedENet(_AbstractHyperEstimator):
    """
    """
    def __init__(self, n_features, model_tag=None):
        """
        """
        self.n_features = n_features
        if model_tag is None:
            self.model_tag = 'td_enet'
        else:
            self.model_tag = model_tag
        self.prob = False

    def build(self, hp):
        """
        """
        chosen_optimizer = hp.Choice(
            name='{}_optimizer'.format(self.model_tag),
            values=['rmsprop', 'adam']
        )

        feat_input = Input(
            shape=(None, self.n_features),
            name='features_input'
        )
        cont_input = Input(
            shape=(None, ),
            name='context_input'
        )

        model_input_tensors = [feat_input, cont_input]

        cont_embedding = Embedding(
            input_dim=10,
            output_dim=1,
            input_length=None,
            name='embedding_layer_{}'.format('context')
        )(cont_input)

        features = Concatenate(
            name='features_concatenation'
        )([feat_input, cont_embedding])

        # ABSENCE
        absence = TimeDistributed(
            Dense(
                units=1,
                kernel_regularizer=l1_l2(
                    hp.Float(
                        min_value=1e-5,
                        max_value=0.1,
                        sampling='log',
                        name='absence_l1_l2'
                    )
                )
            ),
            name='output_absence_td'
        )(features)
        absence = Activation(
            'relu',
            name='output_absence_act'
        )(absence)

        # ACTIVE TIME ESTIMATION
        active = TimeDistributed(
            Dense(
                units=1,
                kernel_regularizer=l1_l2(
                    hp.Float(
                        min_value=1e-5,
                        max_value=0.1,
                        sampling='log',
                        name='active_l1_l2'
                    )
                )
            ),
            name='output_active_td'
        )(features)
        active = Activation(
            'relu',
            name='output_active_act'
        )(active)

        # INACTIVE TIME ESTIMATION
        sess_time = TimeDistributed(
            Dense(
                units=1,
                kernel_regularizer=l1_l2(
                    hp.Float(
                        min_value=1e-5,
                        max_value=0.1,
                        sampling='log',
                        name='sess_time_l1_l2'
                    )
                )
            ),
            name='output_sess_time_td'
        )(features)
        sess_time = Activation(
            'relu',
            name='output_sess_time_act'
        )(sess_time)

        # ACTIVITY ESTIMATION
        activity = TimeDistributed(
            Dense(
                units=1,
                kernel_regularizer=l1_l2(
                    hp.Float(
                        min_value=1e-5,
                        max_value=0.1,
                        sampling='log',
                        name='activity_l1_l2'
                    )
                )
            ),
            name='output_activity_td'
        )(features)
        activity = Activation(
            'relu',
            name='output_activity_act'
        )(activity)

        # SESSION ESTIMATION
        sess = TimeDistributed(
            Dense(
                units=1,
                kernel_regularizer=l1_l2(
                    hp.Float(
                        min_value=1e-5,
                        max_value=0.1,
                        sampling='log',
                        name='sess_l1_l2'
                    )
                )
            ),
            name='output_sess_td'
        )(features)
        sess = Activation(
            'relu',
            name='output_sess_act'
        )(sess)

        model = Model(
            inputs=model_input_tensors,
            outputs=[
                absence,
                active,
                sess_time,
                activity,
                sess
            ]
        )
        model.compile(
            optimizer=chosen_optimizer,
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
        return model


class TimeDistributedMLP(_AbstractHyperEstimator):
    """
    """
    def __init__(self, n_features, prob=False, model_tag=None):
        """
        """
        self.n_features = n_features
        if model_tag is None:
            self.model_tag = 'td_mlp'
        else:
            self.model_tag = model_tag
        self.prob = prob

    def build(self, hp):
        """
        """
        chosen_optimizer = hp.Choice(
            name='{}_optimizer'.format(self.model_tag),
            values=['rmsprop', 'adam']
        )
        self.dropout_spatial = hp.Boolean(
            name='{}_dropout_spatial'.format(self.model_tag)
        )
        self.dropout_rate = hp.Float(
            min_value=0.0,
            max_value=0.4,
            step=0.1,
            name='{}_dropout_rate'.format(self.model_tag)
        )

        feat_input = Input(
            shape=(None, self.n_features),
            name='features_input'
        )
        cont_input = Input(
            shape=(None, ),
            name='context_input'
        )
        cont_embedding = self._generate_embedding_block(
            hp=hp,
            input_tensor=cont_input,
            input_dim=10,
            tag='context'
        )

        model_input_tensors = [feat_input, cont_input]

        features = Concatenate(
            name='features_concatenation'
        )([feat_input, cont_embedding])

        # DENSE BLOCK
        dense = self._generate_fully_connected_block(
            hp=hp,
            input_tensor=features,
            tag='global_features',
            prob=self.prob,
            max_layers=15
        )

        # ABSENCE ESTIMATION
        absence = self._generate_fully_connected_block(
            hp=hp,
            input_tensor=dense,
            tag='absence',
            prob=self.prob,
            max_layers=15
        )
        absence = TimeDistributed(
            Dense(
                units=1
            ),
            name='output_absence_td'
        )(absence)
        absence = Activation(
            'relu',
            name='output_absence_act'
        )(absence)

        # ACTIVE TIME ESTIMATION
        active = self._generate_fully_connected_block(
            hp=hp,
            input_tensor=dense,
            tag='active',
            prob=self.prob,
            max_layers=15
        )
        active = TimeDistributed(
            Dense(
                units=1
            ),
            name='output_active_td'
        )(active)
        active = Activation(
            'relu',
            name='output_active_act'
        )(active)

        # SESSION TIME ESTIMATION
        sess_time = self._generate_fully_connected_block(
            hp=hp,
            input_tensor=dense,
            tag='sess_time',
            prob=self.prob,
            max_layers=15
        )
        sess_time = TimeDistributed(
            Dense(
                units=1
            ),
            name='output_sess_time_td'
        )(sess_time)
        sess_time = Activation(
            'relu',
            name='output_sess_time_act'
        )(sess_time)

        # ACTIVITY ESTIMATION
        activity = self._generate_fully_connected_block(
            hp=hp,
            input_tensor=dense,
            tag='activity',
            prob=self.prob,
            max_layers=15
        )
        activity = TimeDistributed(
            Dense(
                units=1
            ),
            name='output_activity_td'
        )(activity)
        activity = Activation(
            'relu',
            name='output_activity_act'
        )(activity)

        # SESSIONS ESTIMATION
        sess = self._generate_fully_connected_block(
            hp=hp,
            input_tensor=dense,
            tag='sess',
            prob=self.prob,
            max_layers=15
        )
        sess = TimeDistributed(
            Dense(
                units=1
            ),
            name='output_sess_td'
        )(sess)
        sess = Activation(
            'relu',
            name='output_sess_act'
        )(sess)

        model = Model(
            inputs=model_input_tensors,
            outputs=[
                absence,
                active,
                sess_time,
                activity,
                sess
            ]
        )
        model.compile(
            optimizer=chosen_optimizer,
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
        return model
