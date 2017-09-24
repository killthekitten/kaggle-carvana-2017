import threading

from scipy.misc.pilutil import imread, imsave

from params import args
import numpy as np
import os

from utils import ThreadsafeIter


def average_strategy(images):
    return np.average(images, axis=0)


def hard_voting(images):
    rounded = np.round(images / 255.)
    return np.round(np.sum(rounded, axis=0) / images.shape[0]) * 255.

def ensemble_image(files, dirs, ensembling_dir, strategy):
    for file in files:
        images = []
        for dir in dirs:
            file_path = os.path.join(dir, file)
            if os.path.exists(file_path):
                images.append(imread(file_path, mode='L'))
        images = np.array(images)

        if strategy == 'average':
            ensembled = average_strategy(images)
        elif strategy == 'hard_voting':
            ensembled = hard_voting(images)
        else:
            raise ValueError('Unknown ensembling strategy')
        imsave(os.path.join(ensembling_dir, file), ensembled)


def ensemble(dirs, strategy, ensembling_dir, n_threads):
    files = ThreadsafeIter(os.listdir(dirs[0]))
    threads = [threading.Thread(target=ensemble_image, args=(files, dirs, ensembling_dir, strategy)) for i in range(n_threads)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()


if __name__ == '__main__':
    n_threads = args.ensembling_cpu_threads
    ensembling_dir = args.ensembling_dir
    strategy = args.ensembling_strategy
    dirs = args.dirs_to_ensemble
    folds_dir = args.folds_dir
    dirs = [os.path.join(folds_dir, d) for d in dirs]
    for d in dirs:
        if not os.path.exists(d):
            raise ValueError(d + " doesn't exist")
    ensemble(dirs, strategy, ensembling_dir, n_threads)
