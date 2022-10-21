""" !!. """
import json
import pandas as pd
from IPython.display import display
import sklearn.metrics


from batchflow.models.metrics import ClassificationMetrics



class MetricsEvaluator:
    def __init__(self, targets=None, predictions=None, classes=None, metrics=None, shares=None, fmt='labels', **kwargs):
        self.targets = targets
        self.predictions = predictions

        self.shares = shares
        self.classes = classes
        self.num_classes = len(self.classes)

        self.fmt = fmt
        self.kwargs = kwargs
        self._classification_metrics = metrics

    @property
    def classification_metrics(self):
        if self.num_classes is None:
            raise ValueError("!!.")

        if self._classification_metrics is None:
            metrics = ClassificationMetrics(self.targets, self.predictions, num_classes=self.num_classes,
                                            fmt=self.fmt, **self.kwargs)
            self._classification_metrics = metrics
        return self._classification_metrics

    def plot_confusion_matrix(self, normalize=True, vmin=0, vmax=1, **kwargs):
        self.classification_metrics.plot_confusion_matrix(normalize=normalize, vmin=vmin, vmax=vmax, **kwargs)

    @staticmethod
    def value_to_percent(value):
        return f"{(value * 100).round(1)}"

    def evaluate_classwise_metrics(self, names=('precision', 'recall'), show=True, cmap='RdYlGn', vmin=0, vmax=1):
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
        self.plot_confusion_matrix(**kwargs)
        classwise_metrics = self.evaluate_classwise_metrics(names=classwise, show=show, cmap=cmap, vmin=vmin, vmax=vmax)
        altogether_metrics = self.evaluate_altogether_metrics(names=altogether, show=show, cmap=cmap, vmin=vmin, vmax=vmax)
        all_metrics = {**classwise_metrics, **altogether_metrics}

        if savepath is not None:
            with open(savepath, 'w') as f:
                json.dump(all_metrics, f)

        return all_metrics

    def evaluate_clustering_metrics(self, names=('adjusted_rand_score', 'adjusted_mutual_info_score'), show=True,
                                    cmap='RdYlGn', vmin=0, vmax=1):
        metrics_dict = {name: getattr(sklearn.metrics, name)(self.targets, self.predictions) for name in names}

        if show:
            df_data = {key: [value] for key, value in metrics_dict.items()}
            metrics_df = pd.DataFrame(data=df_data, )
            styled_df = metrics_df.style.background_gradient(cmap, vmin=vmin, vmax=vmax)
            styled_df = styled_df.format(self.value_to_percent)
            display(styled_df)

        return metrics_dict