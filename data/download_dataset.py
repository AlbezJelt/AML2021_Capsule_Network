import tensorflow_datasets as tfds
import shutil
import pathlib

base_path = pathlib.Path(__file__).parent.resolve()

tfds_data_dir = f'{base_path}/tensorflow_dataset'
zipped_files_dir = f'{base_path}/tensorflow_dataset/downloads'

print('Downloading and preparing the dataset using tensorflow_dataset...')
builder = tfds.builder(name='patch_camelyon', data_dir=tfds_data_dir)
builder.download_and_prepare()

print('Deleting the zipped files...')
shutil.rmtree(zipped_files_dir)

print('Done!')