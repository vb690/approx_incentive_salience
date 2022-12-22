from IPython.display import display

from itertools import combinations

from patsy import dmatrix

import numpy as np

import pymc3 as pm

import matplotlib.pyplot as plt
import seaborn as sns


class LMMPerformance:
    """Class for assessing perforrmance difference using
    a Linear Mixed Effect Model
    """
    def __init__(self, df, models_column, contexts_column, time_column,
                 outcomes_column, targets_column=None,
                 robust=False):
        """
        """
        self.df = df
        self.models_column = models_column
        self.contexts_column = contexts_column
        self.time_column = time_column
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

        contexts = data[self.contexts_column].map(
            {value: code for code, value in enumerate(unique_contexts)}
        ).values
        models = data[self.models_column].map(
            {value: code for code, value in enumerate(unique_models)}
        ).values
        times = data[self.time_column]
        times_matrix = np.array(
            dmatrix(
                "bs(x, df=6, degree=3, include_intercept=True) - 1",
                {"x": np.arange(data[self.time_column].max() + 1)}
            )
        )

        outcomes = data[self.outcomes_column].values

        return unique_models, unique_contexts, unique_times, \
            models, contexts, times, times_matrix, outcomes

    def __build(self, target, priors={
                    "Hyper Mu Time": {"mu": 0, "sigma": 1},
                    "Hyper Sigma Time": {"beta": 1},

                    "Mu Slope": {"mu": 0, "sigma": 1},
                    "Sigma": {"beta": 1}
                    }
                ):
        """
        """
        unique_models, unique_contexts, unique_times, \
            models, contexts, times, times_matrix, outcomes = self.__define_data(target=target)
        coords = {
            "Models": unique_models,
            "Outcomes": np.arange(outcomes.size)
        }
        with pm.Model(coords=coords) as model:
            models_idx = pm.Data(
                "models_idx",
                models,
                dims="Outcomes"
            )
            times_matrix_data = pm.Data(
                "b_spline_matrix_6_dof",
                times_matrix
            )

            # hyper priors
            hyper_mu_time_context = pm.Normal(
                name="Hyper Mu Time Context",
                mu=priors["Hyper Mu Time"]["mu"],
                sd=priors["Hyper Mu Time"]["sigma"]
            )
            hyper_sigma_time_context = pm.HalfCauchy(
                name="Hyper Sigma Time Context",
                beta=priors["Hyper Sigma Time"]["beta"]
            )
            coef_time_context = pm.Normal(
                name="Coef Time Context",
                mu=hyper_mu_time_context,
                sd=hyper_sigma_time_context,
                shape=(6, 6)
            )

            for idx, context in enumerate(unique_contexts):

                unique_times_context = np.unique(
                    self.df[self.df[self.contexts_column] == context][self.time_column].values
                )
                pm.Deterministic(
                    f"Varying Intercept Context {context}",
                    pm.math.dot(
                        times_matrix_data[unique_times_context, :],
                        coef_time_context[:, idx]
                    )
                )

            varying_intercept_time_comp = pm.math.dot(
                times_matrix_data,
                coef_time_context
            )
            intercept = pm.Deterministic(
                "Varying Intercept Time Context",
                varying_intercept_time_comp[times, contexts]
            )

            # define the slope
            model_slope = pm.Normal(
                name="Model Slope",
                mu=priors["Mu Slope"]["mu"],
                sd=priors["Mu Slope"]["sigma"],
                dims="Models"
            )

            # build Student T model
            mu = pm.Deterministic(
                "Mu",
                intercept + model_slope[models_idx]
            )
            sigma = pm.HalfCauchy(
                name="Sigma",
                beta=priors["Sigma"]["beta"]
            )

            if self.robust:
                nu = pm.Gamma("Nu", alpha=2, beta=0.1)
                out = pm.StudentT(
                    name=f"Observed SMAPE",
                    mu=mu,
                    sigma=sigma,
                    nu=nu,
                    observed=outcomes
                )
            else:
                out = pm.Normal(
                    name=f"Observed SMAPE",
                    mu=mu,
                    sigma=sigma,
                    observed=outcomes
                )
        plate = pm.model_graph.model_to_graphviz(
            model
        )
        display(plate)
        setattr(self, "model", model)
        setattr(self, "plate", plate)

    def comparison(self, target):
        """
        """
        if self.targets_column is not None:
            data = self.df[
                self.df[self.targets_column] == target.replace("_", " ")
            ]
        else:
            data = self.df

        unique_models = data[self.models_column].unique()
        code_to_model = {
            code: model for code, model in enumerate(unique_models)
        }
        trace = getattr(self, target)
        model_trace = trace["Model Slope"]
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
            axs[index].axvline(0, linestyle=":", color="r")
            axs[index].axvline(-.1, linestyle=":", color="k")
            axs[index].axvline(.1, linestyle=":", color="k")
            axs[index].set_title(
                f"{code_to_model[comp[0]]} - {code_to_model[comp[1]]}"
            )
            axs[index].set_ylabel("")

        plt.tight_layout()
        fig.text(
            0.5,
            -0.01,
            "Difference in SMAPE",
            ha="center"
        )
        fig.text(
            -0.01,
            0.5,
            "Density",
            va="center",
            rotation="vertical"
        )
        plt.savefig(f"results\\figures\\models_performance\\bayes\\{target}_comp_2.png", dpi=300)
        plt.show()

        return None

    def analyze(self, targets, figsize=(5, 5), approx=False, **kwargs):
        """
        """
        var_intercepts = [
            f"Varying Intercept Context {context}" for context in self.df[self.contexts_column].unique()
        ]
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
                setattr(self, target.replace(" ", "_"), traces)

                print(target)

                summary_context_time = pm.summary(
                        traces,
                        var_names=var_intercepts
                )
                summary_model = pm.summary(
                    traces,
                    var_names=["Model Slope", "Nu", "Sigma"]
                )

                ax_1 = pm.traceplot(
                    traces,
                    var_names=["Nu", "Sigma"],
                    figsize=figsize
                )
                plt.savefig(f"results\\figures\\models_performance\\bayes\\{target}_marginals_2.png", dpi=300)
                ax_2 = pm.traceplot(
                    traces,
                    var_names=var_intercepts,
                    figsize=figsize
                )
                ax_3 = pm.plot_forest(
                    traces,
                    var_names=["Model Slope"],
                    combined=True,
                    ridgeplot_quantiles=[0.05, .25, .5, .75, 0.95],
                    figsize=figsize
                )
                plt.savefig(f"results\\figures\\models_performance\\bayes\\{target}_models_2.png", dpi=300)

                fig, axs = plt.subplots(2, 3, figsize=(15, 5))
                for var_intercept, ax in zip(var_intercepts, axs.flatten()):

                    ax.plot(
                        np.arange(2, traces[var_intercept].shape[1] + 2),
                        traces[var_intercept].mean(0)
                    )
                    ax.fill_between(
                        np.arange(2, traces[var_intercept].shape[1] + 2),
                        np.percentile(traces[var_intercept], 2.5, axis=0),
                        np.percentile(traces[var_intercept], 97.5, axis=0),
                        alpha=0.25
                    )
                    ax.set_title(var_intercept)

                plt.tight_layout()
                plt.savefig(f"results\\figures\\models_performance\\bayes\\{target}_interc_2.png", dpi=300)

            display(summary_context_time)
            display(summary_model)
            plt.show()

            self.comparison(target.replace(" ", "_"))

        return None
