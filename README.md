## Approximating Attributed Incentive Salience in Large Scale Scenarios. <br> A Representation Learning Approach Based on Artificial Neural Networks

## Motivation

## Data

Due to commerical sensitivity and  [data protection regulations](https://en.wikipedia.org/wiki/General_Data_Protection_Regulation) we are not allowed to pubblicly release the data employed in the present work. However, we will try to provide illustrative examples on the expected data format so to facilitate replication and extension to different contexts.

### Behavioural Features
The behavioural features employed for this project comes from a specific area of application, namely: predicting the intensity of future interacions between individuals and videogames. They aim to be behavioural descriptor of how strong the interactions (i.e. game sessions) between an individual (i.e. an user) and an object (i.e. a videogames) are.

* `sess_order`: order of the interaction in the sequence of considered interactions.
* `absence`: time elapsed since the previous interaction.
* `sess_played_time`: total duration of the interaction.
* `sess_time`: ammount of time spent actively interacting with the game (it is always a fraction of `session_played_time`).
* `activity`: total number of different actions perfromed during the interaction.
* `max_sess`: maximum number of interactons recorded for a specific individual.
* `context`: context from which the interaction comes from.

### Examples Datasets
The interaction data obtained from each object, should be stored in separate `.csv` files located in the `data\csv` directory inside the project root directory. We report here some synthetic examples of the dataset we employed.  
  
**Dataset 1**

| user_id | sess_order | absence | sess_played_time | sess_time | activity | max_sess | context |
|:-------:|:-------------:|:-------:|:-------------------:|:------------:|:--------:|:----------------:|:-------:|
|   XXX   |       1       |    10   |          30         |      35      |    14    |         6        |   lis   |
|   XXX   |       2       |    20   |          5          |      15      |     1    |         6        |   lis   |
|   XXX   |       3       |    10   |          6          |      12      |     3    |         6        |   lis   |

**Dataset 2**

| user_id | sess_order | absence | sess_played_time | sess_time | activity | max_sess | context |
|:-------:|:-------------:|:-------:|:-------------------:|:------------:|:--------:|:----------------:|:-------:|
|   YYY   |       1       |    40   |          4          |       7      |    43    |         2        |   hms   |
|   YYY   |       2       |    50   |          35         |      38      |    12    |         2        |   hms   |

**Dataset N**

| user_id | sess_order | absence | sess_played_time | sess_time | activity | max_sess | context |
|:-------:|:-------------:|:-------:|:-------------------:|:------------:|:--------:|:----------------:|:-------:|
|   ZZZ   |       1       |    12   |          21         |      21      |     3    |         2        |   N   |
|   ZZZ   |       2       |    13   |          8          |       9      |    26    |         2        |   N   |

In order to generalize our approach to other fields of application, the required data should entail the concept of "interaction" meaning a fixed ammount of time during which an individual interact with an object. The intensity of this interaction must be quantifiable, the behavioural features we proposed can be a starting point but other ad-hoc measures can be employed. Each considered individual should at least have had 2 interactions with an object. Knowing the eaxact nature of the considered objects is not mandatory but each object must be distinguisheable from the others. 

### Estimating Attributed Saliency for Predicting the Intensity of Future Interaction


### Architecture

<p align="center">
  <img width="800" height="400" src="https://github.com/vb690/approx_incentive_salience/blob/main/readme_pic/rnn_architecture.png">
</p>

### Hyper-parameters Tuning

Each trainable model in this project can have its hyper-parameters optimized through a range of tuning algorithms. Each model is a subclass of an `AbstractHyperEstimator` (that in turn is a subclass of a KerasTuner [`HyperModel`](https://keras.io/guides/keras_tuner/getting_started/#you-can-use-a-hypermodel-subclass-instead-of-a-modelbuilding-function)) providing functions that allow to define blocks of tunable Keras layers. The available layers are `Embedding`, `Dense` and `LSTM`.  
  
Defining a block only requires to specify the maximum number of layers and hidden units we want to explore (the lower bounds, step size and range of possible activation functions are pre-defined). If we want for example define a tunable block of fully connected layers we would call the relative method:

```python
fc_block = _generate_fully_connected_block(
        hp=hp, # hp object provided by KerasTuner during optimization
        input_tensor=input_tensor, # input to the block of densely connected layers
        tag='dense', # identifier for all the layers in the block
        max_layers=10
        max_dim=512,
    )
```
Series of tunable blocks can then be combined for defining entire computational graphs (i.e. models):
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import Dense

from modules.utils.model_utils.supervised import _AbstractHyperEstimator

class MyModel(_AbstractHyperEstimator):

  def __init__(self, model_tag, prob=False):
    self.model_tag = model_tag,
    # this determines if the model will use monte carlo dropout
    # for approximating bayesian inference (i.e. providing probabilistic predictions)
    self.prob = prob 
    
  def build(self, hp):
    self.dropout_rate = hp.Float(
        min_value=0.0,
        max_value=0.4,
        step=0.05,
        name=f'{self.model_tag}_dropout_rate'
    )
    input_tensor = Input(
        shape=(None, ),
        name='my_input'
    )
    embedding = self._generate_embedding_block(
        hp=hp,
        input_tensor=input_tensor,
        input_dim=100, # this is the size of the vocabulary
        max_dim=512,
        tag='embedding'
    )

    recurrent = self._generate_recurrent_block(
        hp=hp,
        input_tensor=embedding,
        tag='recurrent',
        max_layers=3,
        max_dim=512
    )

    dense = self._generate_fully_connected_block(
        hp=hp,
        input_tensor=recurrent,
        tag='dense',
        prob=self.prob,
        max_layers=10
        max_dim=512,
    )
    
    out = Dense(
        units=1,
        name='out_dense'
    )(dense)
    out = Activation(
      'sigmoid',
      name='out_act'
    )(out)
    
    model = Model(
      inputs=input,
      outputs=out,
    )
    
    model.compile(
      optimizer='adam',
      loss='binary_crossentropy
    )
  return model
```

### Embedding Extraction

## How to Use

### Scripts
1. **Data Preparation**   
Given a number of different datasets coming from various contexts (here different videogames) and a set of behavioural features the `data_preparation` script will first pre-process each dataset seprately, then create targets as the lead-1 version of each feature, concatenate all the the datasets in a single one, shuffle and splitting it in a tuning and validation set (making sure to not disrupt the sequential nature of the data) and finally storing the datasets as numpy arrays of size `(batch_size, sequence_len, n_features)`. Each element in a batch correspond to a single user while `sequence_len` is the number of available sessions for that specific user. Here `sequence_len` has to be consistent within a batch but can vary freely between batches. This process of batch creation was repeated for all the model's inputs (i.e. behavioural and context features).
3. **Models Optimization**
Given a number of different trainable models (here ElasticNet, MultilayerPerceptron and our architecture) the `model_optimization` script will tune their hyperparametrs using the procedure highlighted in the **Hyper-parameters Tuning** section. The optimization can be initiated calling the `tune` method of a model

```python
from tensorflow.keras.callbacks import EarlyStopping

from kerastuner.tuners import Hyperband

model = MyModel(model_tag='my_model')

# we instatiate a callback for halting training
stopper = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=5,
    verbose=1,
    mode='auto',
    restore_best_weights=True
)

model.tune(
    tuner=Hyperband, # the chosen tuner
    generator=my_generator, # this is either a Keras generator or a list of numpy arrays
    verbose=2,
    validation_data=my_validation_data_generator,
    # the following are tuner specific kwargs
    epochs=100,
    max_epochs=100,
    hyperband_iterations=2,
    objective='val_loss',
    callbacks=[stopper],
    directory='optim',
    project_name='my_model_hb'
)
```
5. **Models Comparison**
6. **Embedding Extraction**
    * Dimensionality Reduction
    * Alligned Dimesnionality Reduction
    * Data Container

## Results

We link here to frozen HTML versions of Jupyter Notebook containing a large part of analyses developed for this project. Here the code used for running the analyses as well as some ancillary results (not reported in the pre-print due to space constrains) can be found.

### EDA and Performance Analysis

1. [Exploratory Data Analysis](https://htmlpreview.github.io/?https://github.com/vb690/approx_incentive_salience/blob/main/notebooks_html/EDA_analysis.html)
2. [Models Performance Analysis](https://htmlpreview.github.io/?https://github.com/vb690/approx_incentive_salience/blob/main/notebooks_html/model_performance_analysis.html)

### Representation Analysis

1. [UMAP Analyses](https://htmlpreview.github.io/?https://github.com/vb690/approx_incentive_salience/blob/main/notebooks_html/embedding_analysis.html)
2. [Activations Analyses](https://htmlpreview.github.io/?https://github.com/vb690/approx_incentive_salience/blob/main/notebooks_html/activation_analysis.html) 
3. [Temporal UMAP Analyses](https://htmlpreview.github.io/?https://github.com/vb690/approx_incentive_salience/blob/main/notebooks_html/umap_traces_visualization.html)

### Partition Analysis

1. [Partitioning and Behavioural Profiles Extraction](https://htmlpreview.github.io/?https://github.com/vb690/approx_incentive_salience/blob/main/notebooks_html/cluster_analysis.html)
