import json
import numpy as np
import h5py
import tensorflow as tf

class DataNormalization():

    def __init__(self, json_summary=None, weather_fields=['msl','u10','v10','t2m']):
        self.summary = {}
        if json_summary is not None:
            self.load_summary(json_summary)

        self.weather_fields = weather_fields

    def compute_summary(self, weather_data, ssh):
        """ Computes the mean and std from the data. """

        for i, field in enumerate(self.weather_fields):
            weather_field = weather_data[...,i]

            self.summary[field] = (weather_field.mean(), weather_field.std())

        self.summary['ssh'] = (ssh.mean(), ssh.std())

    def save_summary(self, filename):
        json_summary = {key: {'mean': float(mean), 'std': float(std)} for key, (mean, std) in self.summary.items()}

        with open(filename, 'w') as file:
            json.dump(json_summary, file, indent=2)

    def load_summary(self, filename):
        with open(filename, 'r') as file:
            json_summary = json.load(file)

        self.summary = {key: (val['mean'], val['std']) for key,val in json_summary.items()}

    def normalize_weather_data(self, weather_data):
        """ Normalizes weather data using precomputed mean and std values. """

        feat_means = np.array([self.summary[field][0] for field in self.weather_fields])
        feat_stds = np.array([self.summary[field][1] for field in self.weather_fields])

        # Standardize
        weather_standardized = (weather_data - feat_means) / feat_stds

        return weather_standardized

    def normalize_ssh(self, ssh_delta):
        """ Normalize ssh values. """

        mean, std = self.summary['ssh']
        return (ssh_delta - mean) / std

    def denormalize_ssh(self, ssh_delta_norm):
        """ Denormalize ssh values. """

        mean, std = self.summary['ssh']
        return (ssh_delta_norm * std) + mean


def hdf5_dataset(path, fields):
    """ Creates a tf.data.Dataset from a hdf5 file

    path: path to the hdf5 file,
    fields: dataset names to read from the hdf5 file,

    returns: tf.data.Dataset
    """

    # Reads dataset row by row
    def _generator():
        with h5py.File(path, 'r') as file:
            datasets = [file[field] for field in fields]
            for row in zip(*datasets):
                yield row

    # Reads hdf5 metadata (types and shapes)
    with h5py.File(path, 'r') as file:
        datasets = [file[field] for field in fields]

        types = tuple(ds.dtype for ds in datasets)
        shapes = tuple(ds.shape[1:] for ds in datasets)

    # Create dataset
    ds = tf.data.Dataset.from_generator(_generator, types, shapes)
    return ds
