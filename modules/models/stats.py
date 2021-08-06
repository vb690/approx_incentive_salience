from IPython.display import display

from itertools import combinations

import numpy as np

import pymc3 as pm

import matplotlib.pyplot as plt
import seaborn as sns


class LMMPerformance:
    """Class for assessing perfromance difference using
    a Linear Mixed Effect Model
    """
    def __init__(self, df, models_column, contexts_column, time_column,
                 fold_column, outcomes_column, targets_column=None,
                 robust=False):
        """
        """
        self.df = df
        self.models_column = models_column
        self.contexts_column = contexts_column
        self.time_column = time_column
        self.fold_column = fold_column
        self.targets_column = targets_column
        self.outcomes_column = outcomes_column
        self.robust = robust

    def __define_data(self, target):
        """
        """
        if self.targets_column is not None:
            data = self.df[self.df[self.targets_column] == target]
        else:
            data = self.df

        unique_contexts = data[self.contexts_column].unique()
        unique_models = data[self.models_column].unique()
        unique_times = data[self.time_column].unique()
        unique_folds = data[self.fold_column].unique()

        contexts = data[self.contexts_column].map(
            {value: code for code, value in enumerate(unique_contexts)}
        ).values
        models = data[self.models_column].map(
            {value: code for code, value in enumerate(unique_models)}
        ).values
        times = data[self.time_column].map(
            {value: code for code, value in enumerate(unique_times)}
        ).values
        folds = data[self.fold_column].map(
            {value: code for code, value in enumerate(unique_folds)}
        ).values
        outcomes = data[self.outcomes_column].values

        return unique_models, unique_contexts, unique_times, unique_folds, \
            models, contexts, times, folds, outcomes

    def __build(self, target, priors={
                    'Hyper Mu Context': {'mu': 0, 'sigma': 0.1},
                    'Hyper Sigma Context': {'beta': 25},

                    'Hyper Mu Time': {'mu': 0, 'sigma': 0.1},
                    'Hyper Sigma Time': {'beta': 25},

                    'Hyper Mu Folds': {'mu': 0, 'sigma': 0.1},
                    'Hyper Sigma Folds': {'beta': 25},

                    'Mu Slope': {'mu': 0, 'sigma': 0.1},
                    'Sigma': {'beta': 25}
                    }
                ):
        """
        """
        unique_models, unique_contexts, unique_times, unique_folds, \
            models, contexts, times, folds, \
            outcomes = self.__define_data(target=target)
        coords = {
            'Models': unique_models,
            'Contexts': unique_contexts,
            'Times': unique_times,
            'Folds': unique_folds,
            'Outcomes': np.arange(outcomes.size)
        }
        with pm.Model(coords=coords) as model:
            contexts_idx = pm.Data(
                'contexts_idx',
                contexts,
                dims='Outcomes'
            )
            times_idx = pm.Data(
                'times_idx',
                times,
                dims='Outcomes'
            )
            folds_idx = pm.Data(
                'folds_idx',
                folds,
                dims='Outcomes'
            )
            models_idx = pm.Data(
                'models_idx',
                models,
                dims='Outcomes'
            )
            grand_mean = pm.Data(
                'Grand Mean',
                np.mean(outcomes)
            )

            # hyper priors
            hyper_mu_context = pm.Normal(
                name='Hyper Mu Context',
                mu=priors['Hyper Mu Context']['mu'],
                sd=priors['Hyper Mu Context']['sigma']
            )
            hyper_sigma_context = pm.HalfCauchy(
                name='Hyper Sigma Context',
                beta=priors['Hyper Sigma Context']['beta']
            )
            hyper_mu_time = pm.Normal(
                name='Hyper Mu Time',
                mu=priors['Hyper Mu Time']['mu'],
                sd=priors['Hyper Mu Time']['sigma']
            )
            hyper_sigma_time = pm.HalfCauchy(
                name='Hyper Sigma Time',
                beta=priors['Hyper Sigma Time']['beta']
            )
            hyper_mu_folds = pm.Normal(
                name='Hyper Mu Fold',
                mu=priors['Hyper Mu Folds']['mu'],
                sd=priors['Hyper Mu Folds']['sigma'],
            )
            hyper_sigma_folds = pm.HalfCauchy(
                name='Hyper Sigma Fold',
                beta=priors['Hyper Sigma Folds']['beta']
            )

            varying_intercept_context = pm.Normal(
                name='Varying Intercept Context',
                mu=hyper_mu_context,
                sd=hyper_sigma_context,
                dims='Contexts'
            )

            varying_intercept_time = pm.Normal(
                name='Varying Intercept Time',
                mu=hyper_mu_time,
                sd=hyper_sigma_time,
                dims='Times'
            )

            varying_intercept_fold = pm.Normal(
                name='Varying Intercept Fold',
                mu=hyper_mu_folds,
                sd=hyper_sigma_folds,
                dims='Folds'
            )

            intercept = pm.Deterministic(
                'Intercept = Grand Mean + Time + Context + Fold N',
                grand_mean
                + varying_intercept_context[contexts_idx]
                + varying_intercept_time[times_idx]
                + varying_intercept_fold[folds_idx]
            )

            # define the slope
            model_slope = pm.Normal(
                name='Model Slope',
                mu=priors['Mu Slope']['mu'],
                sd=priors['Mu Slope']['sigma'],
                dims='Models'
            )

            # build the beta model
            mu = pm.Deterministic(
                'Mu',
                pm.math.invlogit(
                    intercept + model_slope[models_idx],
                )
            )
            sigma = pm.HalfCauchy(
                name='Sigma',
                beta=priors['Sigma']['beta']
            )

            if self.robust:
                nu = pm.Gamma('Nu', alpha=2, beta=0.1)
                out = pm.StudentT(
                    name=f'Observed SMAPE',
                    mu=mu,
                    sigma=sigma,
                    nu=nu,
                    observed=outcomes
                )
            else:
                out = pm.Normal(
                    name=f'Observed SMAPE',
                    mu=mu,
                    sigma=sigma,
                    observed=outcomes
                )
        plate = pm.model_graph.model_to_graphviz(
            model
        )
        display(plate)
        setattr(self, 'model', model)
        setattr(self, 'plate', plate)

    def comparison(self, target):
        """
        """
        if self.targets_column is not None:
            data = self.df[
                self.df[self.targets_column] == target.replace('_', ' ')
            ]
        else:
            data = self.df

        unique_models = data[self.models_column].unique()
        code_to_model = {
            code: model for code, model in enumerate(unique_models)
        }
        trace = getattr(self, target)
        model_trace = trace['Model Slope']
        comparisons = combinations(
            [model for model in range(model_trace.shape[1])], 2
        )
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))
        axs = axs.flatten()
        for index, comp in enumerate(comparisons):

            delta = \
                model_trace[:, comp[0]] - model_trace[:, comp[1]]

            sns.kdeplot(
                delta,
                ax=axs[index]
            )

            axs[index].set_title(
                f'{code_to_model[comp[0]]} - {code_to_model[comp[1]]}'
            )
            axs[index].set_ylabel('')

        plt.tight_layout()
        fig.text(
            0.5,
            -0.01,
            'Difference in SMAPE',
            ha='center'
        )
        fig.text(
            -0.01,
            0.5,
            'Density',
            va='center',
            rotation='vertical'
        )
        plt.show()

        return None

    def analyze(self, targets, figsize=(5, 5), approx=False, **kwargs):
        """
        """
        for target in targets:

            self.__build(
                target=target
            )

            with self.model:

                if approx:
                    mean_field = pm.fit(**kwargs)
                    traces = mean_field.sample(1000)
                else:
                    traces = pm.sample(
                        **kwargs
                    )
                setattr(self, target.replace(' ', '_'), traces)

                print(target)

                summary_time = pm.summary(
                        traces,
                        var_names=['Varying Intercept Time']
                )
                summary_context = pm.summary(
                        traces,
                        var_names=['Varying Intercept Context']
                )
                summary_folds = pm.summary(
                        traces,
                        var_names=['Varying Intercept Fold']
                )
                summary_model = pm.summary(
                    traces,
                    var_names=['Model Slope']
                )

                ax_1 = pm.traceplot(
                    traces,
                    var_names=[
                        'Model Slope',
                        'Varying Intercept Time',
                        'Varying Intercept Context',
                        'Varying Intercept Fold'
                    ],
                    figsize=figsize
                )
                ax_2 = pm.plot_forest(
                    traces,
                    var_names=[
                        'Varying Intercept Time',
                        'Varying Intercept Context',
                        'Varying Intercept Fold'
                    ],
                    combined=True,
                    ridgeplot_quantiles=[0.05, .25, .5, .75, 0.95],
                    figsize=figsize
                )
                ax_3 = pm.plot_forest(
                    traces,
                    var_names=['Model Slope'],
                    combined=True,
                    ridgeplot_quantiles=[0.05, .25, .5, .75, 0.95],
                    figsize=figsize
                )

            display(summary_context)
            display(summary_time)
            display(summary_folds)
            display(summary_model)
            plt.show()

            self.comparison(target.replace(' ', '_'))

        return None
