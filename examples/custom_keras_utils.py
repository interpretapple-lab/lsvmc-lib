# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras image dataset loading utilities."""
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import multiprocessing
import os

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def index_directory(directory,
                    labels,
                    formats,
                    class_names=None,
                    shuffle=True,
                    seed=None,
                    follow_links=False):
  """Make list of all files in the subdirs of `directory`, with their labels.

  Args:
    directory: The target directory (string).
    labels: Either "inferred"
        (labels are generated from the directory structure),
        or a list/tuple of integer labels of the same size as the number of
        valid files found in the directory. Labels should be sorted according
        to the alphanumeric order of the image file paths
        (obtained via `os.walk(directory)` in Python).
    formats: Whitelist of file extensions to index (e.g. ".jpg", ".txt").
    class_names: Only valid if "labels" is "inferred". This is the explict
        list of class names (must match names of subdirectories). Used
        to control the order of the classes
        (otherwise alphanumerical order is used).
    shuffle: Whether to shuffle the data. Default: True.
        If set to False, sorts the data in alphanumeric order.
    seed: Optional random seed for shuffling.
    follow_links: Whether to visits subdirectories pointed to by symlinks.

  Returns:
    tuple (file_paths, labels, class_names).
      file_paths: list of file paths (strings).
      labels: list of matching integer labels (same length as file_paths)
      class_names: names of the classes corresponding to these labels, in order.
  """
  inferred_class_names = []
  for subdir in sorted(os.listdir(directory)):
    if os.path.isdir(os.path.join(directory, subdir)):
      inferred_class_names.append(subdir)
  if not class_names:
    class_names = inferred_class_names
  else:
    if set(class_names) != set(inferred_class_names):
      raise ValueError(
          'The `class_names` passed did not match the '
          'names of the subdirectories of the target directory. '
          'Expected: %s, but received: %s' %
          (inferred_class_names, class_names))
  class_indices = dict(zip(class_names, range(len(class_names))))

  # Build an index of the files
  # in the different class subfolders.
  pool = multiprocessing.pool.ThreadPool()
  results = []
  filenames = []
  for dirpath in (os.path.join(directory, subdir) for subdir in class_names):
    results.append(
        pool.apply_async(index_subdirectory,
                         (dirpath, class_indices, follow_links, formats)))
  labels_list = []
  for res in results:
    partial_filenames, partial_labels = res.get()
    labels_list.append(partial_labels)
    filenames += partial_filenames
  if labels != 'inferred':
    if len(labels) != len(filenames):
      raise ValueError('Expected the lengths of `labels` to match the number '
                       'of files in the target directory. len(labels) is %s '
                       'while we found %s files in %s.' % (
                           len(labels), len(filenames), directory))
  else:
    i = 0
    labels = np.zeros((len(filenames),), dtype='int32')
    for partial_labels in labels_list:
      labels[i:i + len(partial_labels)] = partial_labels
      i += len(partial_labels)

  print('Found %d files belonging to %d classes.' %
        (len(filenames), len(class_names)))
  pool.close()
  pool.join()
  file_paths = [os.path.join(directory, fname) for fname in filenames]

  if shuffle:
    # Shuffle globally to erase macro-structure
    if seed is None:
      seed = np.random.randint(1e6)
    rng = np.random.RandomState(seed)
    rng.shuffle(file_paths)
    rng = np.random.RandomState(seed)
    rng.shuffle(labels)
  return file_paths, labels, class_names


def iter_valid_files(directory, follow_links, formats):
  walk = os.walk(directory, followlinks=follow_links)
  for root, _, files in sorted(walk, key=lambda x: x[0]):
    for fname in sorted(files):
      if fname.lower().endswith(formats):
        yield root, fname


def index_subdirectory(directory, class_indices, follow_links, formats):
  """Recursively walks directory and list image paths and their class index.

  Arguments:
    directory: string, target directory.
    class_indices: dict mapping class names to their index.
    follow_links: boolean, whether to recursively follow subdirectories
      (if False, we only list top-level images in `directory`).
    formats: Whitelist of file extensions to index (e.g. ".jpg", ".txt").

  Returns:
    tuple `(filenames, labels)`. `filenames` is a list of relative file
      paths, and `labels` is a list of integer labels corresponding to these
      files.
  """
  dirname = os.path.basename(directory)
  valid_files = iter_valid_files(directory, follow_links, formats)
  labels = []
  filenames = []
  for root, fname in valid_files:
    labels.append(class_indices[dirname])
    absolute_path = os.path.join(root, fname)
    relative_path = os.path.join(
        dirname, os.path.relpath(absolute_path, directory))
    filenames.append(relative_path)
  return filenames, labels


def get_training_or_validation_split(samples, labels, validation_split, subset):
  """Potentially restict samples & labels to a training or validation split.

  Args:
    samples: List of elements.
    labels: List of corresponding labels.
    validation_split: Float, fraction of data to reserve for validation.
    subset: Subset of the data to return.
      Either "training", "validation", or None. If None, we return all of the
      data.

  Returns:
    tuple (samples, labels), potentially restricted to the specified subset.
  """
  if not validation_split:
    return samples, labels

  num_val_samples = int(validation_split * len(samples))
  if subset == 'training':
    print('Using %d files for training.' % (len(samples) - num_val_samples,))
    samples = samples[:-num_val_samples]
    labels = labels[:-num_val_samples]
  elif subset == 'validation':
    print('Using %d files for validation.' % (num_val_samples,))
    samples = samples[-num_val_samples:]
    labels = labels[-num_val_samples:]
  else:
    raise ValueError('`subset` must be either "training" '
                     'or "validation", received: %s' % (subset,))
  return samples, labels


def labels_to_dataset(labels, label_mode, num_classes):
  label_ds = dataset_ops.Dataset.from_tensor_slices(labels)
  if label_mode == 'binary':
    label_ds = label_ds.map(
        lambda x: array_ops.expand_dims(math_ops.cast(x, 'float32'), axis=-1))
  elif label_mode == 'categorical':
    label_ds = label_ds.map(lambda x: array_ops.one_hot(x, num_classes))
  return label_ds


def check_validation_split_arg(validation_split, subset, shuffle, seed):
  """Raise errors in case of invalid argument values."""
  if validation_split and not 0 < validation_split < 1:
    raise ValueError(
        '`validation_split` must be between 0 and 1, received: %s' %
        (validation_split,))
  if (validation_split or subset) and not (validation_split and subset):
    raise ValueError(
        'If `subset` is set, `validation_split` must be set, and inversely.')
  if subset not in ('training', 'validation', None):
    raise ValueError('`subset` must be either "training" '
                     'or "validation", received: %s' % (subset,))
  if validation_split and shuffle and seed is None:
    raise ValueError(
        'If using `validation_split` and shuffling the data, you must provide '
        'a `seed` argument, to make sure that there is no overlap between the '
        'training and validation subset.')
  





# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras image dataset loading utilities."""
# pylint: disable=g-classes-have-attributes
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import numpy as np

from tensorflow.python.data.ops import dataset_ops
# from tensorflow.python.keras.layers.preprocessing import image_preprocessing
# from tensorflow.python.keras.preprocessing import dataset_utils
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.util.tf_export import keras_export


WHITELIST_FORMATS = ('.bmp', '.gif', '.jpeg', '.jpg', '.png')


# @keras_export('keras.preprocessing.image_dataset_from_directory', v1=[])
def image_dataset_from_directory_v2(directory,
                                 labels='inferred',
                                 label_mode='int',
                                 class_names=None,
                                 color_mode='rgb',
                                 batch_size=32,
                                 image_size=(256, 256),
                                 shuffle=True,
                                 seed=None,
                                 validation_split=None,
                                 subset=None,
                                 interpolation='bilinear',
                                 follow_links=False):
  """Generates a `tf.data.Dataset` from image files in a directory.

  If your directory structure is:

  ```
  main_directory/
  ...class_a/
  ......a_image_1.jpg
  ......a_image_2.jpg
  ...class_b/
  ......b_image_1.jpg
  ......b_image_2.jpg
  ```

  Then calling `image_dataset_from_directory(main_directory, labels='inferred')`
  will return a `tf.data.Dataset` that yields batches of images from
  the subdirectories `class_a` and `class_b`, together with labels
  0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).

  Supported image formats: jpeg, png, bmp, gif.
  Animated gifs are truncated to the first frame.

  Arguments:
    directory: Directory where the data is located.
        If `labels` is "inferred", it should contain
        subdirectories, each containing images for a class.
        Otherwise, the directory structure is ignored.
    labels: Either "inferred"
        (labels are generated from the directory structure),
        or a list/tuple of integer labels of the same size as the number of
        image files found in the directory. Labels should be sorted according
        to the alphanumeric order of the image file paths
        (obtained via `os.walk(directory)` in Python).
    label_mode:
        - 'int': means that the labels are encoded as integers
            (e.g. for `sparse_categorical_crossentropy` loss).
        - 'categorical' means that the labels are
            encoded as a categorical vector
            (e.g. for `categorical_crossentropy` loss).
        - 'binary' means that the labels (there can be only 2)
            are encoded as `float32` scalars with values 0 or 1
            (e.g. for `binary_crossentropy`).
        - None (no labels).
    class_names: Only valid if "labels" is "inferred". This is the explict
        list of class names (must match names of subdirectories). Used
        to control the order of the classes
        (otherwise alphanumerical order is used).
    color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
        Whether the images will be converted to
        have 1, 3, or 4 channels.
    batch_size: Size of the batches of data. Default: 32.
    image_size: Size to resize images to after they are read from disk.
        Defaults to `(256, 256)`.
        Since the pipeline processes batches of images that must all have
        the same size, this must be provided.
    shuffle: Whether to shuffle the data. Default: True.
        If set to False, sorts the data in alphanumeric order.
    seed: Optional random seed for shuffling and transformations.
    validation_split: Optional float between 0 and 1,
        fraction of data to reserve for validation.
    subset: One of "training" or "validation".
        Only used if `validation_split` is set.
    interpolation: String, the interpolation method used when resizing images.
      Defaults to `bilinear`. Supports `bilinear`, `nearest`, `bicubic`,
      `area`, `lanczos3`, `lanczos5`, `gaussian`, `mitchellcubic`.
    follow_links: Whether to visits subdirectories pointed to by symlinks.
        Defaults to False.

  Returns:
    A `tf.data.Dataset` object.
      - If `label_mode` is None, it yields `float32` tensors of shape
        `(batch_size, image_size[0], image_size[1], num_channels)`,
        encoding images (see below for rules regarding `num_channels`).
      - Otherwise, it yields a tuple `(images, labels)`, where `images`
        has shape `(batch_size, image_size[0], image_size[1], num_channels)`,
        and `labels` follows the format described below.

  Rules regarding labels format:
    - if `label_mode` is `int`, the labels are an `int32` tensor of shape
      `(batch_size,)`.
    - if `label_mode` is `binary`, the labels are a `float32` tensor of
      1s and 0s of shape `(batch_size, 1)`.
    - if `label_mode` is `categorial`, the labels are a `float32` tensor
      of shape `(batch_size, num_classes)`, representing a one-hot
      encoding of the class index.

  Rules regarding number of channels in the yielded images:
    - if `color_mode` is `grayscale`,
      there's 1 channel in the image tensors.
    - if `color_mode` is `rgb`,
      there are 3 channel in the image tensors.
    - if `color_mode` is `rgba`,
      there are 4 channel in the image tensors.
  """
  if labels != 'inferred':
    if not isinstance(labels, (list, tuple)):
      raise ValueError(
          '`labels` argument should be a list/tuple of integer labels, of '
          'the same size as the number of image files in the target '
          'directory. If you wish to infer the labels from the subdirectory '
          'names in the target directory, pass `labels="inferred"`. '
          'If you wish to get a dataset that only contains images '
          '(no labels), pass `label_mode=None`.')
    if class_names:
      raise ValueError('You can only pass `class_names` if the labels are '
                       'inferred from the subdirectory names in the target '
                       'directory (`labels="inferred"`).')
  if label_mode not in {'int', 'categorical', 'binary', None}:
    raise ValueError(
        '`label_mode` argument must be one of "int", "categorical", "binary", '
        'or None. Received: %s' % (label_mode,))
  if color_mode == 'rgb':
    num_channels = 3
  elif color_mode == 'rgba':
    num_channels = 4
  elif color_mode == 'grayscale':
    num_channels = 1
  else:
    raise ValueError(
        '`color_mode` must be one of {"rbg", "rgba", "grayscale"}. '
        'Received: %s' % (color_mode,))
  # image_preprocessing.
  interpolation = get_interpolation(interpolation)
  #dataset_utils.
  check_validation_split_arg(validation_split, subset, shuffle, seed)

  if seed is None:
    seed = np.random.randint(1e6)
     #dataset_utils.
  image_paths, labels, class_names = index_directory(
      directory,
      labels,
      formats=WHITELIST_FORMATS,
      class_names=class_names,
      shuffle=shuffle,
      seed=seed,
      follow_links=follow_links)

  if label_mode == 'binary' and len(class_names) != 2:
    raise ValueError(
        'When passing `label_mode="binary", there must exactly 2 classes. '
        'Found the following classes: %s' % (class_names,))

   #dataset_utils.
  image_paths, labels = get_training_or_validation_split(
      image_paths, labels, validation_split, subset)
  

  num_classes=len(class_names)
  path_ds = dataset_ops.Dataset.from_tensor_slices(image_paths)
  dataset = path_ds.map(
      lambda x: path_to_image(x, image_size, num_channels, interpolation))
  if label_mode:
     #dataset_utils.
    label_ds = labels_to_dataset(labels, label_mode, num_classes)
    dataset = dataset_ops.Dataset.zip((dataset, label_ds, path_ds))

#   dataset = paths_and_labels_to_dataset_v2(
#       image_paths=image_paths,
#       image_size=image_size,
#       num_channels=num_channels,
#       labels=labels,
#       label_mode=label_mode,
#       num_classes=len(class_names),
#       interpolation=interpolation)

  if shuffle:
    # Shuffle locally at each iteration
    dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
  dataset = dataset.batch(batch_size)
  # Users may need to reference `class_names`.
  dataset.class_names = class_names
  return dataset


# def paths_and_labels_to_dataset_v2(image_paths,
#                                 image_size,
#                                 num_channels,
#                                 labels,
#                                 label_mode,
#                                 num_classes,
#                                 interpolation):
#   """Constructs a dataset of images and labels."""
#   # TODO(fchollet): consider making num_parallel_calls settable
#   path_ds = dataset_ops.Dataset.from_tensor_slices(image_paths)
#   img_ds = path_ds.map(
#       lambda x: path_to_image(x, image_size, num_channels, interpolation))
#   if label_mode:
#      #dataset_utils.
#     label_ds = labels_to_dataset(labels, label_mode, num_classes)
#     img_ds = dataset_ops.Dataset.zip((img_ds, label_ds, path_ds))
#   return img_ds


def path_to_image(path, image_size, num_channels, interpolation):
  img = io_ops.read_file(path)
  img = image_ops.decode_image(
      img, channels=num_channels, expand_animations=False)
  img = image_ops.resize_images_v2(img, image_size, method=interpolation)
  img.set_shape((image_size[0], image_size[1], num_channels))
  return img


import tensorflow as tf


_RESIZE_METHODS = {
    'bilinear': tf.image.ResizeMethod.BILINEAR,
    'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    'bicubic': tf.image.ResizeMethod.BICUBIC,
    'area': tf.image.ResizeMethod.AREA,
    'lanczos3': tf.image.ResizeMethod.LANCZOS3,
    'lanczos5': tf.image.ResizeMethod.LANCZOS5,
    'gaussian': tf.image.ResizeMethod.GAUSSIAN,
    'mitchellcubic': tf.image.ResizeMethod.MITCHELLCUBIC
}

def get_interpolation(interpolation):
  interpolation = interpolation.lower()
  if interpolation not in _RESIZE_METHODS:
    raise NotImplementedError(
        'Value not recognized for `interpolation`: {}. Supported values '
        'are: {}'.format(interpolation, _RESIZE_METHODS.keys()))
  return _RESIZE_METHODS[interpolation]