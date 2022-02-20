from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import Dense

from ...utils.model_utils.abstract_models import _AbstractHyperEstimator
from ...utils.model_utils.metrics_losses import smape_k


class MelchiorModel(_AbstractHyperEstimator):
    '''
    '''
    def __init__(self, n_features, prob=False, model_tag=None,
                 adjust_for_env=False):
        '''
        Method called when instatiating a MultilayerPerceptron object

        Args:
            -
            -
            -

        Returns:
            -None
        '''
        self.n_features = n_features
        if model_tag is None:
            self.model_tag = 'melchior'
        else:
            self.model_tag = model_tag
        self.prob = prob
        self.adjust_for_env = adjust_for_env

    def build(self, hp):
        '''
        Method for building a tunable TensorFlow graph.

        Args:
         - hp:

        Returns:
         - model: a compile keras model
        '''
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

        # I LEVEL INPUTS
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

        # OPTIONAL ENVIRONMENT ADJUSTING
        if self.adjust_for_env:
            area_input = Input(
                shape=(None,),
                name='area_input'
            )
            hour_input = Input(
                shape=(None, ),
                name='hours_input'
            )
            day_week_input = Input(
                shape=(None, ),
                name='days_week_input'
            )
            day_year_input = Input(
                shape=(None, ),
                name='days_year_input'
            )

            model_input_tensors.extend(
                [
                    area_input,
                    hour_input,
                    day_week_input,
                    day_year_input
                ]
            )

            area_embedding = self._generate_embedding_block(
                hp=hp,
                input_tensor=area_input,
                input_dim=900,
                tag='area'
            )
            hour_embedding = self._generate_embedding_block(
                hp=hp,
                input_tensor=hour_input,
                input_dim=25,
                tag='hours'
            )
            day_week_embedding = self._generate_embedding_block(
                hp=hp,
                input_tensor=day_week_input,
                input_dim=8,
                tag='days_week'
            )
            day_year_embedding = self._generate_embedding_block(
                hp=hp,
                input_tensor=day_year_input,
                input_dim=367,
                tag='days_year'
            )

        # II LEVEL FIRST LSTM
        # explicitly model the contribution of the behavioural features
        # weighted by the context
        feat_cont_concat = Concatenate(
            name='concat_feat_cont'
        )([feat_input, cont_embedding])

        feat_recurrent = self._generate_recurrent_block(
            hp=hp,
            input_tensor=feat_cont_concat,
            tag='features',
            max_layers=1
        )
        if self.adjust_for_env:
            # explicitly model the contribution of hour of the day
            # weighted by the context
            hour_recurrent = self._generate_recurrent_block(
                hp=hp,
                input_tensor=hour_embedding,
                tag='hours'
            )
            hour_cont_concat = Concatenate(
                name='concat_hour_cont'
            )([hour_recurrent, cont_embedding, area_embedding])
            hour_cont = self._generate_fully_connected_block(
                hp=hp,
                input_tensor=hour_cont_concat,
                tag='hour_cont',
                prob=self.prob
            )

            # explicitly model the contribution of day of the week
            # weighted by the context
            day_week_recurrent = self._generate_recurrent_block(
                hp=hp,
                input_tensor=day_week_embedding,
                tag='days_week'
            )
            day_week_cont_concat = Concatenate(
                name='concat_days_week_cont'
            )([day_week_recurrent, cont_embedding, area_embedding])
            day_week_cont = self._generate_fully_connected_block(
                hp=hp,
                input_tensor=day_week_cont_concat,
                tag='days_week_cont',
                prob=self.prob
            )

            # explicitly model the contribution of day of the year
            # weighted by the context
            day_year_recurrent = self._generate_recurrent_block(
                hp=hp,
                input_tensor=day_year_embedding,
                tag='days_year'
            )
            day_year_cont_concat = Concatenate(
                name='concat_days_year_cont'
            )([day_year_recurrent, cont_embedding, area_embedding])
            day_year_cont = self._generate_fully_connected_block(
                hp=hp,
                input_tensor=day_year_cont_concat,
                tag='days_year_cont',
                prob=self.prob
            )

            # merge all the variables used for modelling the environment state
            # and model them temporally
            env_recurrent = Concatenate(
                name='concat_env'
            )([
                hour_cont,
                day_week_cont,
                day_year_cont
                ]
              )
            env_recurrent = self._generate_recurrent_block(
                hp=hp,
                input_tensor=env_recurrent,
                tag='env'
            )

            # V LEVEL SECOND LSTM
            # merge behavioural features with the representation of the
            # environment and model this temporally
            shared_concat = Concatenate(
                name='concat_feat_env'
            )([feat_recurrent, env_recurrent])
            shared_time_distributed = self._generate_fully_connected_block(
                hp=hp,
                input_tensor=shared_concat,
                tag='shared',
                prob=self.prob
            )
            shared_recurrent = self._generate_recurrent_block(
                hp=hp,
                input_tensor=shared_time_distributed,
                tag='shared'
            )
        else:
            shared_recurrent = feat_recurrent

        # VI LEVEL ESTIMATORS
        # create the heds for ostimating the target metrics

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
