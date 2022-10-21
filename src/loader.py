""" !!. """
import os
import numpy as np

from skimage.io import imread
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize

from batchflow import Dataset, ImagesBatch, Notifier, plot



class ImagesDataset(Dataset):
    """ !!. """
    # pre-calculated dataset stats
    MEAN = np.array([134.4449684 , 123.43467167, 107.95887036])
    STD = np.array([58.42912666, 55.37764538, 53.68485717])

    def __init__(self, *args, path=None, encode_labels=False, normalize=False, resize_shape=None,
                 preload=True, batch_class=ImagesBatch, **kwargs):
        """ !!. """
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
        """ !!. """
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
        mean = cls.MEAN if mean is None else mean
        std = cls.STD if std is None else std
        image = (image - mean) / std
        return image

    @classmethod
    def denormalize_image(cls, image, mean=None, std=None):
        mean = cls.MEAN if mean is None else mean
        std = cls.STD if std is None else std
        image = (image * std + mean) / 255
        return image

    @staticmethod
    def encode(labels):
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        labels = label_encoder.transform(labels)
        return labels, label_encoder

    def show_samples(self, rng=None, indices='examples', n_samples=None):
        if indices == 'examples':
            indices = [rng.choice(np.nonzero(self.labels == label)[0]) for label in range(self.n_classes)]
        elif indices == 'random':
            indices = rng.integers(0, self.size, n_samples)

        if isinstance(indices, (int, np.integer)):
            indices = [indices]

        images = list(self.images[indices])
        labels = self.label_encoder.inverse_transform(self.labels[indices]).tolist()
        plot(data=images, title=labels, combine='separate')
