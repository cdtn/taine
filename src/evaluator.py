""" Predictions evaluation tools. """
import json
import pandas as pd
import sklearn.metrics

from IPython.display import display

from batchflow.models.metrics import ClassificationMetrics



class MetricsEvaluator:
    """ Calculates classification or clustering metrics for given targets and predictions.

    - To evaluate classification metrics, one should provide either `targets` and `predictions` or `metrics`.
      Then the methods `plot_confusion_matrix`, `evaluate_classwise_metrics`, `evaluate_altogether_metrics`,
      and `evaluate_classification_metrics` can be used.
    - To evaluate clusterting metrics, one should provide `targets` and `predictions` and then use
      `evaluate_clustering_metrics` method.

    Notes
    -----
    While it's technically possible to provide `predictions` obtained via clustering and
    call classification metrics evaluation methods upon them, the results will make no sense.
    And contrary, for `predictions` obtained via classification, clustering metrics calculation
    might be used to estimate clustering predictions goodness vs classification ones.
    TODO: split functionality between two classes, one for clusteriing and one for classfication

    Parameters
    ----------
    targets : sequence of integers or None
        If sequence of integers, represent target labels.
        If None, it's assumed that a `metrics` parameter is provided.
    predictions : sequence of numbers or None
        If sequence of integers, represent predicted labels.
        If sequence of floats, represent either predicted labels probability or their logits
        If None, it's assumed that a `metrics` parameter is provided.
    classes : list or None
        If list, represent target classes names to show next to metrics values when visualising them.
        If None, target classes names are displayed as just their ordinal numbers.
    metrics : batchflow.ClassificationMetrics or None
        If batchflow.ClassificationMetrics, use it to call `evaluate` upon it.
        Used, when one already has an aggregated metrics.
        If None, `targets` and `predictions` must be provided.
    shares : list of floats or None
        If list of floats, specificies and additional column of classes shares in data to show next to metrics values.
    fmt : 'labels', 'proba' or 'logits'
        Specifices `prediction` format.
        If 'labels', `predictions` is treated as already calculated labels.
        If 'proba', `predictions` is treated as predicted labels probabilities (argmax applied).
        If 'logits', `predictions` is treated as predicted labels logits (sigmoid + argmax applied).
    kwargs : misc
        For `batchflow.ClassificationMetrics`.
    """
    def __init__(self, targets=None, predictions=None, classes=None, metrics=None, shares=None, fmt='labels', **kwargs):
        """ Store parameters necessary for further metrics calculation. """
        self.targets = targets
        self.predictions = predictions

        self.shares = shares
        self.classes = classes
        self.num_classes = len(self.classes) # FIXME: fails when classes is None

        self.fmt = fmt
        self.kwargs = kwargs
        self._classification_metrics = metrics

    @property
    def classification_metrics(self):
        """ Lazy confusion matrix calculation. """
        if self.num_classes is None:
            raise ValueError("!!.")

        if self._classification_metrics is None:
            metrics = ClassificationMetrics(self.targets, self.predictions, num_classes=self.num_classes,
                                            fmt=self.fmt, **self.kwargs)
            self._classification_metrics = metrics
        return self._classification_metrics

    def plot_confusion_matrix(self, normalize=True, vmin=0, vmax=1, **kwargs):
        """ Display confusion matrix with convinient defaults.

        Parameters
        ----------
        normalize : bool
            Whether to normalize confusion matrix over target classes.
        vmin, vmax : numbers
            Used for colormap lower and upper limits calculation. Affects matrix colouring.
        kwargs : misc
            For `batchflow.ClassificationMetrics.plot_confusio_matrix`.
        """
        self.classification_metrics.plot_confusion_matrix(normalize=normalize, vmin=vmin, vmax=vmax, **kwargs)

    @staticmethod
    def value_to_percent(value):
        """ Values formatter for metrics dataframe visualiation. """
        return f"{(value * 100).round(1)}"

    def evaluate_classwise_metrics(self, names=('precision', 'recall'), show=True, cmap='RdYlGn', vmin=0, vmax=1):
        """ Calculate classification metrics for every class.

        Parameters
        ----------
        names : str or sequence of str
            List of metrics to evaluate.
            See `batchflow.models.metrics.METRICS_ALIASES` for detailed info about names and aliases.
        show : bool
            Whether display evaluated metrics values packed into a dataframe.
        cmap : valid matplotlib colormap
            Used to color displayed dataframe values.
        vmin, vmax : numbers
            Used for colormap lower and upper limits calculation.
        """
        metrics_dict = self.classification_metrics.evaluate(names, multiclass=None)
        metrics_dict = {key: list(value) for key, value in metrics_dict.items()}

        if show:
            if self.shares is None:
                data = metrics_dict.copy()
            else:
                data = {'shares': self.shares, **metrics_dict}
            index = self.classes or range(self.num_classes)
            metrics_df = pd.DataFrame(data=data, index=index)
            styled_df = metrics_df.style.background_gradient(cmap, vmin=vmin, vmax=vmax, subset=list(names))
            styled_df = styled_df.format(self.value_to_percent)
            display(styled_df)

        return metrics_dict

    def evaluate_altogether_metrics(self, names=('accuracy', 'f1_score'), show=True, cmap='RdYlGn', vmin=0, vmax=1):
        """ Calculate generalized classification metrics.

        Parameters
        ----------
        names : str or sequence of str
            List of metrics to evaluate.
            See `batchflow.models.metrics.METRICS_ALIASES` for detailed info about names and aliases.
        show : bool
            Whether display evaluated metrics values packed into a dataframe.
        cmap : valid matplotlib colormap
            Used to color displayed dataframe values.
        vmin, vmax : numbers
            Used for colormap lower and upper limits calculation.
        """
        metrics_dict = self.classification_metrics.evaluate(names)
        metrics_dict = {key: [value] for key, value in metrics_dict.items()}

        if show:
            index = ['altogether']
            metrics_df = pd.DataFrame(data=metrics_dict, index=index)
            styled_df = metrics_df.style.background_gradient(cmap, vmin=vmin, vmax=vmax, subset=list(names))
            styled_df = styled_df.format(self.value_to_percent)
            display(styled_df)

        return metrics_dict

    def evaluate_classification_metrics(self, classwise=('precision', 'recall'), altogether=('accuracy', 'f1_score'),
                                        show=True, cmap='RdYlGn', vmin=0, vmax=1, savepath=None, **kwargs):
        """ Calculate both classwise and generalized classification metrics and display confusion matrix.

        Parameters
        ----------
        classwise, altogether : str or sequence of str
            Lists of classwise and generalized metrics to evaluate.
            See `batchflow.models.metrics.METRICS_ALIASES` for detailed info about names and aliases.
        show : bool
            Whether display evaluated metrics values packed into a dataframe.
        cmap : valid matplotlib colormap
            Used to color displayed dataframe values.
        vmin, vmax : numbers
            Used for colormap lower and upper limits calculation.
        """
        self.plot_confusion_matrix(**kwargs) # FIXME: matrix is displayed when `show=False`
        classwise_metrics = self.evaluate_classwise_metrics(names=classwise, show=show, cmap=cmap, vmin=vmin, vmax=vmax)
        altogether_metrics = self.evaluate_altogether_metrics(names=altogether, show=show, cmap=cmap, vmin=vmin, vmax=vmax)
        all_metrics = {**classwise_metrics, **altogether_metrics}

        if savepath is not None:
            with open(savepath, 'w') as f:
                json.dump(all_metrics, f)

        return all_metrics

    def evaluate_clustering_metrics(self, names=('adjusted_rand_score', 'adjusted_mutual_info_score'), show=True,
                                    cmap='RdYlGn', vmin=0, vmax=1):
        """ Calculate clustering metrics.

        Parameters
        ----------
        names : sequence of str
            Names of functions from `sklearn.metrics` to call upon given targets and predictions.
        show : bool
            Whether display evaluated metrics values packed into a dataframe.
        cmap : valid matplotlib colormap
            Used to color displayed dataframe values.
        vmin, vmax : numbers
            Used for colormap lower and upper limits calculation.
        """
        metrics_dict = {name: getattr(sklearn.metrics, name)(self.targets, self.predictions) for name in names}

        if show:
            df_data = {key: [value] for key, value in metrics_dict.items()}
            metrics_df = pd.DataFrame(data=df_data, )
            styled_df = metrics_df.style.background_gradient(cmap, vmin=vmin, vmax=vmax)
            styled_df = styled_df.format(self.value_to_percent)
            display(styled_df)

        return metrics_dict
