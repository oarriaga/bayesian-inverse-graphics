import os
import json
import glob
import pickle
from pathlib import Path
from datetime import datetime


def print_progress(total, current_arg, message=''):
    # end = '' if current_arg == total else '\r'
    print(f'Progress {((current_arg + 1) / total):2.1%} {message}', end='\r')
    if current_arg == (total - 1):
        print('\n')


def build_directory(root='experiments', label=None):
    directory_name = build_directory_name(root, label)
    make_directory(directory_name)
    return directory_name


def build_directory_name(root, label=None):
    directory_name = [datetime.now().strftime("%d-%m-%Y_%H-%M-%S")]
    if label is not None:
        directory_name.extend([label])
    directory_name = '_'.join(directory_name)
    return os.path.join(root, directory_name)


def make_directory(directory_name):
    Path(directory_name).mkdir(parents=True, exist_ok=True)
    return directory_name


def load_parameters(wildcard, filename):
    filepath = find_path(wildcard)
    filepath = os.path.join(filepath, filename)
    filedata = open(filepath, 'r')
    parameters = json.load(filedata)
    return parameters


def find_path(wildcard):
    filenames = glob.glob(wildcard)
    filepaths = []
    for filename in filenames:
        if os.path.isdir(filename):
            filepaths.append(filename)
    return max(filepaths, key=os.path.getmtime)


def has_extension(fullpath, extension):
    return fullpath.endswith(extension)


def validate_extension(filename, extension):
    if not has_extension(filename, extension):
        raise ValueError(f'Filename {filename} missing extension {extension}')


def write_dictionary(dictionary, directory, filename):
    fielpath = os.path.join(directory, filename)
    filedata = open(fielpath, 'w')
    json.dump(dictionary, filedata, indent=4)


def write_pytree(x, directory, filename):
    fullpath = os.path.join(directory, filename)
    pickle.dump(x, open(fullpath, 'wb'))


def load_pytree(directory, filename):
    fullpath = os.path.join(directory, filename)
    return pickle.load(open(fullpath, 'rb'))


def write_trace(trace, directory):
    filepath = os.path.join(directory, 'trace.pkl')
    filedata = open(filepath, 'wb')
    pickle.dump(trace, filedata)


def write_summary(summary, directory):
    filepath = os.path.join(directory, 'summary.tex')
    summary.style.to_latex(filepath)
