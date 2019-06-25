import tensorflow as tf

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))

print(dataset1.output_types)
print(dataset1.output_shapes)