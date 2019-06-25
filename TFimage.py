"""# Load images with tf.data
This tutorial provides a simple example of how to load an image dataset using `tf.data`.
The dataset used in this example is distributed as directories of images, with one class of image per directory.
"""

import tensorflow as tf
tf.enable_eager_execution()
import pathlib
import random
import os
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE

"""Download the dataset"""


data_root_orig = tf.keras.utils.get_file('flower_photos',
                                         'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                         untar=True)
data_root = pathlib.Path(data_root_orig)
print(data_root)

for item in data_root.iterdir():
  print(item)

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print(image_count)

print(all_image_paths[:10])

"""### Determine the label for each image

List the available labels:
"""

label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
print(label_names)

"""Assign an index to each label:"""

label_to_index = dict((name, index) for index,name in enumerate(label_names))
print(label_to_index)

"""Create a list of every file, and its label index"""

all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

print("First 10 labels indices: ", all_image_labels[:10])

"""### Load and format the images

TensorFlow includes all the tools you need to load and process images:
"""

img_path = all_image_paths[0]
img_path

"""here is the raw data:"""

img_raw = tf.read_file(img_path)

"""Decode it into an image tensor:"""

img_tensor = tf.image.decode_jpeg(img_raw)

print(img_tensor.shape)
print(img_tensor.dtype)

"""Resize it for your model:"""

img_final = tf.image.resize(img_tensor, [192, 192])
img_final = img_final/255.0
print(img_final.shape)
print(img_final.numpy().min())
print(img_final.numpy().max())

"""Wrap up these up in simple functions for later."""

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [192, 192])
  image /= 255.0  # normalize to [0,1] range

  return image

def load_and_preprocess_image(path):
  image = tf.read_file(path)
  return preprocess_image(image)

image_path = all_image_paths[0]
label = all_image_labels[0]

plt.imshow(load_and_preprocess_image(img_path))
plt.grid(False)
plt.title(label_names[label].title())
plt.show()

"""## Build a `tf.data.Dataset`

### A dataset of images

The easiest way to build a `tf.data.Dataset` is using the `from_tensor_slices` method.

Slicing the array of strings, results in a dataset of strings:
"""

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

"""The `output_shapes` and `output_types` fields describe the content of each item in the dataset. In this case it is a set of scalar binary-strings"""

print('shape: ', repr(path_ds.output_shapes))
print('type: ', path_ds.output_types)
print()
print(path_ds)

"""Now create a new dataset that loads and formats images on the fly by mapping `preprocess_image` over the dataset of paths."""

image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

"""### A dataset of `(image, label)` pairs

Using the same `from_tensor_slices` method we can build a dataset of labels
"""

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

for label in label_ds.take(10):
  print(label_names[label.numpy()])

"""Since the datasets are in the same order we can just zip them together to get a dataset of `(image, label)` pairs."""

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

"""The new dataset's `shapes` and `types` are tuples of shapes and types as well, describing each field:"""

print(image_label_ds)

"""Note: When you have arrays like `all_image_labels` and `all_image_paths` an alternative to `tf.data.dataset.Dataset.zip` is to slice the pair of arrays."""

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

# The tuples are unpacked into the positional arguments of the mapped function
def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label

image_label_ds = ds.map(load_and_preprocess_from_path_label)
image_label_ds

"""### Basic methods for training

To train a model with this dataset you will want the data:

* To be well shuffled.
* To be batched.
* To repeat forever.
* Batches to be available as soon as possible.

These features can be easily added using the `tf.data` api.
"""

BATCH_SIZE = 32

# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# `prefetch` lets the dataset fetch batches, in the background while the model is training.
ds = ds.prefetch(buffer_size=AUTOTUNE)
ds

"""There are a few things to note here:

1. The order is important.

  * A `.shuffle` before a `.repeat` would shuffle items across epoch boundaries (some items will be seen twice before others are seen at all).
  * A `.shuffle` after a `.batch` would shuffle the order of the batches, but not shuffle the items across batches.

1. We use a `buffer_size` the same size as the dataset for a full shuffle. Up to the dataset size, large values provide better randomization, but use more memory.

1. The shuffle buffer is filled before any elements are pulled from it. So a large `buffer_size` may cause a delay when your `Dataset` is starting.

1. The shuffeled dataset doesn't report the end of a dataset until the shuffle-buffer is completely empty. The `Dataset` is restarted by `.repeat`, causing another wait for the shuffle-buffer to be filled.

This last point can be addressed by using the `tf.data.Dataset.apply` method with the fused `tf.data.experimental.shuffle_and_repeat` function:
"""

ds = image_label_ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
ds

"""### Pipe the dataset to a model

Fetch a copy of MobileNet v2 from `tf.keras.applications`.

This will be used for a simple transfer learning example.

Set the MobileNet weights to be non-trainable:
"""

mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable=False

"""This model expects its input to be normalized to the `[-1,1]` range:

```
help(keras_applications.mobilenet_v2.preprocess_input)
```

<pre>
...
This function applies the "Inception" preprocessing which converts
the RGB values from [0, 255] to [-1, 1]
...
</pre>

So before the passing it to the MobilNet model, we need to convert the input from a range of `[0,1]` to `[-1,1]`.
"""

def change_range(image,label):
  return 2*image-1, label

keras_ds = ds.map(change_range)

"""The MobileNet returns a `6x6` spatial grid of features for each image.

Pass it a batch of images to see:
"""

# The dataset may take a few seconds to start, as it fills its shuffle buffer.
image_batch, label_batch = next(iter(keras_ds))

feature_map_batch = mobile_net(image_batch)
print(feature_map_batch.shape)

"""So build a model wrapped around MobileNet, and use `tf.keras.layers.GlobalAveragePooling2D` to average over those space dimensions, before the output `tf.keras.layers.Dense` layer:"""

model = tf.keras.Sequential([
  mobile_net,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(len(label_names))])

"""Now it produces outputs of the expected shape:"""

logit_batch = model(image_batch).numpy()

print("min logit:", logit_batch.min())
print("max logit:", logit_batch.max())
print()

print("Shape:", logit_batch.shape)

"""Compile the model to describe the training procedure:"""

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

"""There are 2 trainable variables: the Dense `weights` and `bias`:"""

len(model.trainable_variables)

model.summary()

"""Train the model.

Normally you would specify the real number of steps per epoch, but for demonstration purposes only run 3 steps.
"""

steps_per_epoch=tf.ceil(len(all_image_paths)/BATCH_SIZE).numpy()
steps_per_epoch

model.fit(ds, epochs=1, steps_per_epoch=3)

"""## Performance

Note: This section just shows a couple of easy tricks that may help performance. For an in depth guide see [Input Pipeline Performance](https://www.tensorflow.org/guide/performance/datasets).

The simple pipeline used above reads each file individually, on each epoch. This is fine for local training on CPU but may not be sufficient for GPU training, and is totally inappropriate for any sort of distributed training.

To investigate, first build a simple function to check the performance of our datasets:
"""

import time

def timeit(ds, batches=2*steps_per_epoch+1):
  overall_start = time.time()
  # Fetch a single batch to prime the pipeline (fill the shuffle buffer),
  # before starting the timer
  it = iter(ds.take(batches+1))
  next(it)

  start = time.time()
  for i,(images,labels) in enumerate(it):
    if i%10 == 0:
      print('.',end='')
  print()
  end = time.time()

  duration = end-start
  print("{} batches: {} s".format(batches, duration))
  print("{:0.5f} Images/s".format(BATCH_SIZE*batches/duration))
  print("Total time: {}s".format(end-overall_start))

"""The performance of the current dataset is:"""

ds = image_label_ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
ds

timeit(ds)

"""### Cache

Use `tf.data.Dataset.cache` to easily cache calculations across epochs. This is especially performant if the dataq fits in memory.

Here the images are cached, after being pre-precessed (decoded and resized):
"""

ds = image_label_ds.cache()
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
ds

timeit(ds)

"""One disadvantage to using an in memory cache is that the cache must be rebuilt on each run, giving the same startup delay each time the dataset is started:"""

timeit(ds)

"""If the data doesn't fit in memory, use a cache file:"""

ds = image_label_ds.cache(filename='./cache.tf-data')
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(1)
ds

timeit(ds)

"""The cache file also has the advantage that it can be used to quickly restart the dataset without rebuilding the cache. Note how much faster it is the second time:"""

timeit(ds)

"""### TFRecord File

#### Raw image data

TFRecord files are a simple format to store a sequence of binary blobs. By packing multiple examples into the same file, TensorFlow is able to read multiple examples at once, which is especially important for performance when using a remote storage service such as GCS.

First, build a TFRecord file from the raw image data:
"""

image_ds = tf.data.Dataset.from_tensor_slices(all_image_paths).map(tf.read_file)
tfrec = tf.data.experimental.TFRecordWriter('images.tfrec')
tfrec.write(image_ds)

"""Next build a dataset that reads from the TFRecord file and decodes/reformats the images using the `preprocess_image` function we defined earlier."""

image_ds = tf.data.TFRecordDataset('images.tfrec').map(preprocess_image)

"""Zip that with the labels dataset we defined earlier, to get the expected `(image,label)` pairs."""

ds = tf.data.Dataset.zip((image_ds, label_ds))
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds=ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
ds

timeit(ds)

"""This is slower than the `cache` version because we have not cached the preprocessing.

#### Serialized Tensors

To save some preprocessing to the TFRecord file, first make a dataset of the processed images, as before:
"""

paths_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = paths_ds.map(load_and_preprocess_image)
image_ds

"""Now instead of a dataset of `.jpeg` strings, this is a dataset of tensors.

To serialize this to a TFRecord file you first convert the dataset of tensors to a dataset of strings.
"""

ds = image_ds.map(tf.serialize_tensor)
ds

tfrec = tf.data.experimental.TFRecordWriter('images.tfrec')
tfrec.write(ds)

"""With the preprocessing cached, data can be loaded from the TFrecord file quite efficiently. Just remember to de-serialized tensor before trying to use it."""

ds = tf.data.TFRecordDataset('images.tfrec')

def parse(x):
  result = tf.parse_tensor(x, out_type=tf.float32)
  result = tf.reshape(result, [192, 192, 3])
  return result

ds = ds.map(parse, num_parallel_calls=AUTOTUNE)
ds

"""Now, add the labels and apply the same standard operations as before:"""

ds = tf.data.Dataset.zip((ds, label_ds))
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds=ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
ds

timeit(ds)