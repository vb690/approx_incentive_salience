import os

import numpy as np

from tensorflow.keras.callbacks import EarlyStopping

from kerastuner.tuners import Hyperband

from modules.models.supervised.engagement_estimators import MelchiorModel
from modules.models.supervised.baselines import TimeDistributedENet
from modules.models.supervised.baselines import TimeDistributedMLP
from modules.utils.data_utils.data_handlers import DataGenerator

from modules.utils.general_utils.utilities import save_full_model

os.environ['PATH'] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'

##############################################################################

TUN_PATH = 'data\\test\\inputs\\context'

VAL_FRAC = 0.2

MAX_EPOCHS = 40
HB_ITERATIONS = 1

BTCH = [i for i in range(len(os.listdir(TUN_PATH)))]
BTCH = np.random.choice(BTCH, len(BTCH), replace=False)

VAL_CUT = int(VAL_FRAC * len(BTCH))

TU_BTCH = BTCH[:-VAL_CUT]
VAL_TU_BTCH = BTCH[-VAL_CUT:]

INPUTS = [
    'continuous_features',
    'context'
]

TARGETS = [
    'tar_delta_sessions',
    'tar_active_time',
    'tar_session_time',
    'tar_activity',
    'tar_sessions'
]

MODELS = {
    'enet_td': TimeDistributedENet(
        n_features=4,
        adjust_for_env=False
    ),
    'mlp_td': TimeDistributedMLP(
        n_features=4,
        adjust_for_env=False
    ),
    'melchior': MelchiorModel(
        n_features=4,
        adjust_for_env=False
    )
}
TU_GEN = DataGenerator(
    list_batches=TU_BTCH,
    inputs=INPUTS,
    targets=TARGETS,
    train=True,
    shuffle=True
)
VAL_TU_GEN = DataGenerator(
    list_batches=VAL_TU_BTCH,
    inputs=INPUTS,
    targets=TARGETS,
    train=True,
    shuffle=True
)

##############################################################################

for name, model in MODELS.items():

    ES = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=5,
        verbose=1,
        mode='auto',
        restore_best_weights=True
    )

    model.tune(
        tuner=Hyperband,
        generator=TU_GEN,
        verbose=2,
        validation_data=VAL_TU_GEN,
        epochs=MAX_EPOCHS,
        max_epochs=MAX_EPOCHS,
        hyperband_iterations=HB_ITERATIONS,
        objective='val_loss',
        callbacks=[ES],
        directory='o',
        project_name='{}_hb'.format(name[:3])
    )

    save_full_model(model=model)
