import pathlib
import random

data_root = '/home/haichuan/Downloads/mould_data_set'
training_data_root = '/home/haichuan/Downloads/mould_data_set/training_data'
test_data_root = '/home/haichuan/Downloads/mould_data_set/test_data'
training_data_root = pathlib.Path(training_data_root)
print(training_data_root)

for item in training_data_root.iterdir():
    print(item)

all_image_paths = list(training_data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
print(len(all_image_paths))

random.shuffle(all_image_paths)
print(all_image_paths[:10])