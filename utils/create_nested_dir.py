from os import makedirs
from os.path import join, exists


def create_nested_dir(folder, sub_folder):
    if not exists(folder):
        makedirs(folder)
    if not exists(join(folder, sub_folder)):
        makedirs(join(folder, sub_folder))