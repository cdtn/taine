import os
import numpy as np

from skimage.io import imread
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize
from batchflow import Dataset, ImagesBatch, Notifier



class ImagesDataset(Dataset):
    """ !!. """
    def __init__(self, *args, path=None, encode_labels=False, normalize=False, resize_shape=None, validate=False,
                 preload=True, batch_class=ImagesBatch, **kwargs):
        """ !!. """
        if path is not None:
            names, images, labels = self.load(path=path, normalize=normalize, resize_shape=resize_shape,
                                              validate=validate, preload=preload)
            self.names = names
            labels, self.label_encoder = self.encode(labels) if encode_labels else (labels, None)

            kwargs['index'] = range(len(names))
            kwargs['preloaded'] = {'images': np.array(images), 'labels': np.array(labels)}

        super().__init__(*args, batch_class=batch_class, **kwargs)

    @staticmethod
    def load(path, normalize=False, resize_shape=False, validate=False, preload=False):
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
                                image = image / 255

                            if resize_shape is not None:
                                image = resize(image, resize_shape)
                            inputs.append(image)

                            if validate:
                                if any(image.min() < 0):
                                    print(f"{image_path} has values < 0")
                                if any(image.max() > 255):
                                    print(f"{image_path} has values > 255")
                        else:
                            inputs.append(image_path)

                        labels.append(style_name)

        return names, inputs, labels

    @staticmethod
    def encode(labels):
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        labels = label_encoder.transform(labels)
        return labels, label_encoder