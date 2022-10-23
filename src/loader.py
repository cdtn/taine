""" Data loading and preprocessing tools. """
import os
import numpy as np

from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder

from batchflow import Dataset, ImagesBatch, Notifier, plot



class ImagesDataset(Dataset):
    """ A `batchflow.Dataset` with additional loading and preprocessing functinality.

    Parameters
    ----------
    path : str or None
        If str, denotes a data source.
        Images are loaded from every non-hidden subdirectory, which names act as class labels.
    encode_labels : bool
        If False, keep labels unchanged.
        If True, use `LabelEncoder` to map class names into numeric labels.
        Save encoder instance under `label_encoder` attribute name.
    normalize : bool or a sequence of two numbers
        If False, keep images unchanged.
        If True, apply mean-std normalization using statistics,
        that are hard-coded into class attributes under `MEAN` and `STD` names.
        If a sequence of two numbers, apply mean-std normalization using first value as mean and second as std.
        TODO: rename to `normalize_images` for clarity
    resize_shape : tuple of two/three integers or None
        If tuple of integers, defines a shape to resize loaded images to.
        If None, images left intact.
    preload : bool
        If False, only parse images paths and put them in dataset, but do not load images themselfes.
        If True, load images from disk.
        TODO: store `paths` in dataset separately, not under `images` name in case of `preload=False`
    batch_class : batchflow.Batch
        For `batchflow.Dataset`.
        TODO: maybe hard-code it
    args, kwargs : misc
        For `batchflow.Dataset`.
    """
    # pre-calculated stats for ArtStyle dataset
    MEAN = np.array([134.4449684 , 123.43467167, 107.95887036])
    STD = np.array([58.42912666, 55.37764538, 53.68485717])

    def __init__(self, *args, path=None, encode_labels=False, normalize=False, resize_shape=None,
                 preload=True, batch_class=ImagesBatch, **kwargs):
        """ Perform custom data loading if `path` provided, else fall back to default base class initialization. """
        if path is not None:
            self.names, images, labels = self.load(path=path, normalize=normalize,
                                                   resize_shape=resize_shape, preload=preload)
            labels, self.label_encoder = self.encode(labels) if encode_labels else (labels, None)
            self.classes = self.label_encoder.classes_.tolist()
            self.n_classes = len(self.classes)

            kwargs['index'] = range(len(self.names))
            kwargs['preloaded'] = {'images': np.array(images), 'labels': np.array(labels)}

        super().__init__(*args, batch_class=batch_class, **kwargs)

    @classmethod
    def load(cls, path, normalize=False, resize_shape=False, preload=False):
        """ Look through every non-hidden directory of `path` and either save inside files' paths or load them and
        optionally apply preprocessing, such as normalization and resize.

        Parameters
        -----—----
        path : str or None
            If str, denotes a data source.
            Images are loaded from every non-hidden subdirectory, which names act as class labels.
        normalize : bool or a sequence of two numbers
            If False, keep images unchanged.
            If True, apply mean-std normalization using statistics,
            that are hard-coded into class attributes under `MEAN` and `STD` names.
            If a sequence of two numbers, apply mean-std normalization using first value as mean and second as std.
        resize_shape : tuple of two/three integers or None
            If tuple of integers, defines a shape to resize loaded images to.
            If None, images left intact.

        Returns
        -------
        names : list of parsed filenames
        inputs : list of either full paths to files or loaded images, depending on `preload` parameter value
        labels : list of parsed images labels, parsed from directories names containing corresponding files
        """
        names, inputs, labels = [], [], []

        for style_name in Notifier('t')(os.listdir(path)):
            if not style_name.startswith('.'):
                style_path = f"{path}/{style_name}"
                for image_name in Notifier('t', leave=False, position=0, desc=style_name)(os.listdir(style_path)):
                    if not image_name.startswith('.'):
                        names.append(image_name)

                        image_path = f"{style_path}/{image_name}"

                        if preload:
                            image = imread(image_path)

                            if normalize:
                                mean, std = normalize if isinstance(normalize, (tuple, list)) else (None, None)
                                image = cls.normalize_image(image, mean=mean, std=std)

                            if resize_shape is not None:
                                image = resize(image, resize_shape, preserve_range=True)

                            inputs.append(image)
                        else:
                            inputs.append(image_path)

                        labels.append(style_name)

        return names, inputs, labels

    @classmethod
    def normalize_image(cls, image, mean=None, std=None):
        """ Perform mean-std normalization. If normalization parameters are not provided, use default precaculated. """
        mean = cls.MEAN if mean is None else mean
        std = cls.STD if std is None else std
        image = (image - mean) / std
        return image

    @classmethod
    def denormalize_image(cls, image, mean=None, std=None):
        """ Perform mean-std denormalization followed by [0, 1] range normalization.
        If original normalization parameters are not provided, use default ones.
        Used for visualizations, when already mean-std normalized rgb image need to be either from [0, 1] or [0, 255].
        TODO: choose less misleading method naming
        """
        mean = cls.MEAN if mean is None else mean
        std = cls.STD if std is None else std
        image = (image * std + mean) / 255
        return image

    @staticmethod
    def encode(labels):
        """ Encode given labels using ordinal encoding. """
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        labels = label_encoder.transform(labels)
        return labels, label_encoder

    def show_samples(self, rng=None, indices='examples', n_samples=None):
        """ Display sample images from dataset.

        Parameters
        ----------
        rng : None or np.random.Generator
            Used to choose sample images, when `indices='random'` or `indices='examples'`.
        indices : integer, sequence of integers, 'examples' or 'random'
            If integer, display image stored in dataset under that index.
            If sequence of integers, display images stored in dataset under that indices.
            If 'examples', display one random sample for every dataset class.
            If 'random`, display `n_samples` arbitrary images from dataset.
        n_samples : None or integer
            Number of images to display, when `indices='random'`.
        """
        if indices == 'examples':
            indices = [rng.choice(np.nonzero(self.labels == label)[0]) for label in range(self.n_classes)]
        elif indices == 'random':
            indices = rng.integers(0, self.size, n_samples)

        if isinstance(indices, (int, np.integer)):
            indices = [indices]

        images = list(self.images[indices])
        labels = self.label_encoder.inverse_transform(self.labels[indices]).tolist()
        plot(data=images, title=labels, combine='separate')
