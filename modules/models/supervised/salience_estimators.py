from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import Dense

from ...utils.model_utils.supervised import _AbstractHyperEstimator
from ...utils.model_utils.metrics_losses import smape_k


class RNN(_AbstractHyperEstimator):
    """
    """
    def __init__(self, n_features, prob=False, model_tag=None):
        """
        Args:
            -
            -
            -

        Returns:
            -None
        """
        self.n_features = n_features
        if model_tag is None:
            self.model_tag = 'rnn'
        else:
            self.model_tag = model_tag
        self.prob = prob

    def build(self, hp):
        """
        Method for building a tunable TensorFlow graph.

        Args:
         - hp:

        Returns:
         - model: a compile keras model
        """
        chosen_optimizer = hp.Choice(
            name='{}_optimizer'.format(self.model_tag),
            values=['rmsprop', 'adam']
        )
        self.dropout_rate = hp.Float(
            min_value=0.0,
            max_value=0.4,
            step=0.05,
            name='{}_dropout_rate'.format(self.model_tag)
        )
        self.dropout_spatial = True

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
            input_dim=7,
            tag='context'
        )
        model_input_tensors = [feat_input, cont_input]

        feat_cont_concat = Concatenate(
            name='concat_feat_cont'
        )([feat_input, cont_embedding])

        shared_recurrent = self._generate_recurrent_block(
            hp=hp,
            input_tensor=feat_cont_concat,
            tag='features',
            max_layers=1
        )

        # ABSENCE ESTIMATION
        absence = self._generate_fully_connected_block(
            hp=hp,
            input_tensor=shared_recurrent,
            tag='absence',
            prob=self.prob,
            max_dim=224,
            max_layers=3
        )
        absence = Dense(
            units=1,
            name='output_absence_time_td'
        )(absence)
        absence = Activation(
            'relu',
            name='output_absence_act'
        )(absence)

        # ACTIVE TIME ESTIMATION
        active = self._generate_fully_connected_block(
            hp=hp,
            input_tensor=shared_recurrent,
            tag='active',
            prob=self.prob,
            max_dim=224,
            max_layers=3
        )
        active = Dense(
            units=1,
            name='output_active_td'
        )(active)
        active = Activation(
            'relu',
            name='output_active_act'
        )(active)

        # SESSION TIME ESTIMATION
        sess_time = self._generate_fully_connected_block(
            hp=hp,
            input_tensor=shared_recurrent,
            tag='sess_time',
            prob=self.prob,
            max_dim=224,
            max_layers=3
        )
        sess_time = Dense(
            units=1,
            name='output_sess_time_td'
        )(sess_time)
        sess_time = Activation(
            'relu',
            name='output_sess_time_act'
        )(sess_time)

        # ACTIVITY ESTIMATION
        activity = self._generate_fully_connected_block(
            hp=hp,
            input_tensor=shared_recurrent,
            tag='activity',
            prob=self.prob,
            max_dim=224,
            max_layers=3
        )
        activity = Dense(
            units=1,
            name='output_activity_td'
        )(activity)
        activity = Activation(
            'relu',
            name='output_activity_act'
        )(activity)

        # SESSIONS ESTIMATION
        sess = self._generate_fully_connected_block(
            hp=hp,
            input_tensor=shared_recurrent,
            tag='sess',
            prob=self.prob,
            max_dim=224,
            max_layers=3
        )
        sess = Dense(
            units=1,
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
