import numpy as np

import pandas as pd

from sklearn.preprocessing import MinMaxScaler as mms

from modules.utils.data_utils.data_handlers import data_handling_pipeline


features = [
    'delta_sessions',
    'session_order',
    'active_time',
    'session_time',
    'activity'
]
targets = [
    'user_id',
    'tar_delta_sessions',
    'tar_active_time',
    'tar_session_time',
    'tar_activity',
    'tar_sessions'
]
embeddings = [
    'context'
]
games = [
    'jc3',
    'lis',
    'lisbf',
    'jc4',
    'hmg',
    'hms'
]
"""
###############################################################################

for game in games:

    print(f'Preprocessing {game}')

    df = pd.read_csv(f'data\\csv\\{game}.csv')
    df = df.sort_values(['user_id', 'session_order'])

    df['user_id'] = df['user_id'] + df['context']
    df = df.drop_duplicates(subset=['user_id', 'session_order'])

    df = df.rename(columns={'session_played_time': 'active_time'})
    df = df.rename(columns={'activity_index': 'activity'})
    df['delta_sessions'] = df['delta_sessions'] // 60

###############################################################################

    # OUTLIERS REMOVAL
    df, outliers_report = outliers_removal(
        df=df,
        contamination=0.025,
        n_estimators=200,
        max_samples=5000,
        features=[
            'delta_sessions',
            'active_time',
            'session_time',
            'activity'
        ],
        n_jobs=-1
    )
    outliers_report.to_csv(f'results\\tables\\eda\\{game}.csv')

###############################################################################

    # ACTIVE TIME RAW
    null_filler = df['active_time'].mean()
    df['active_time'] = df['active_time'].apply(
        lambda x: x if x > 0 else null_filler
    )

###############################################################################

    # SESSION TIME
    null_filler = df['session_time'].mean()
    df['session_time'] = df['session_time'].apply(
        lambda x: x if x > 0 else null_filler
    )
    df['session_time'] = np.where(
        df['session_time'] - df['active_time'] < 0,
        df['active_time'],
        df['session_time']
    )
    # create target
    df['tar_session_time'] = df.groupby('user_id')['session_time'].shift(-1)

###############################################################################

    # ABSENCE
    null_filler = df['delta_sessions'].mean()
    df['delta_sessions'] = df['delta_sessions'].apply(
        lambda x: x if x > 0 else null_filler
    )
    # create target
    df['tar_delta_sessions'] = df.groupby(
        'user_id')['delta_sessions'].shift(-1)

###############################################################################

    # ACTIVE TIME PERCENTAGE
    df['active_time'] = df['active_time'] / df['session_time'] * 100
    df['active_time'] = round(df['active_time'], 2)
    # create target
    df['tar_active_time'] = df.groupby('user_id')['active_time'].shift(-1)


###############################################################################

    # ACTIVITY
    null_filler = int(df['activity'].mean())
    df['activity'] = df['activity'].apply(
        lambda x: x if x >= 0 else null_filler
    )
    # df['activity'] = df['activity'] / df['session_time']
    # create target
    df['tar_activity'] = df.groupby('user_id')['activity'].shift(-1)

###############################################################################

    # SESSION
    # create target
    df['tar_sessions'] = df['maximum_sessions'] - df['session_order']

    df['max_sess_cut'] = df.groupby('user_id')['session_order'].transform(
        np.max
    )

###############################################################################

    df = df.fillna(0)
    df = df[
        [
            'user_id',
            'context',
            'session_order',

            'delta_sessions',
            'active_time',
            'session_time',
            'activity',

            'tar_delta_sessions',
            'tar_active_time',
            'tar_session_time',
            'tar_activity',
            'tar_sessions',

            'max_sess_cut'
        ]
    ]
    df = df.sort_values(['user_id', 'session_order'])
    df.to_csv(f'data\\csv\\cleaned\\{game}.csv', index=False)
"""
###############################################################################

# start the data extraction
data_handling_pipeline(
    games_list=games,
    targets_keys=targets,
    embeddings_keys=embeddings,
    features_keys=features,
    scaler=mms,
    global_scaling=True,
    grouping_key='max_sess_cut',
    sorting_keys=['user_id', 'session_order'],
    train_size=0.90,
    batch_size=512
)
