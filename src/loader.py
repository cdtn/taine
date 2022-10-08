import os
import numpy as np

from tqdm.auto import tqdm
from skimage.io import imread
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize
from batchflow import Dataset, ImagesBatch



class ImagesDataset(Dataset):
    """ !!. """
    def __init__(self, *args, path=None, encode_labels=False, normalize=False,
                 resize_shape=None, validate=False, batch_class=ImagesBatch, **kwargs):
        """ !!. """
        if path is not None:
            names, images, labels = self.load(path=path, normalize=normalize,
                                              resize_shape=resize_shape, validate=validate)
            self.names = names

            if encode_labels:
                label_encoder = LabelEncoder()
                label_encoder.fit(labels)
                labels = label_encoder.transform(labels)
            else:
                label_encoder = None
            self.label_encoder = label_encoder

            kwargs['index'] = range(len(names))
            kwargs['preloaded'] = {'images': np.array(images), 'labels': np.array(labels)}

        super().__init__(*args, batch_class=batch_class, **kwargs)

    @staticmethod
    def load(path, normalize, resize_shape, validate):
        """ !!. """
        names, images, labels = [], [], []

        for style_name in tqdm(os.listdir(path)):
            if not style_name.startswith('.'):
                style_path = f"{path}/{style_name}"
                for image_name in tqdm(os.listdir(style_path), leave=False):
                    if not image_name.startswith('.'):
                        names.append(image_name)

                        image_path = f"{style_path}/{image_name}"
                        image = imread(image_path)
                        if normalize:
                            image = image / 255

                        if resize_shape is not None:
                            image = resize(image, resize_shape)
                        images.append(image)

                        labels.append(style_name)

                        if validate:
                            if any(image.min() < 0):
                                print(f"{image_path} has values < 0")
                            if any(image.max() > 255):
                                print(f"{image_path} has values > 255")

        return names, images, labels
