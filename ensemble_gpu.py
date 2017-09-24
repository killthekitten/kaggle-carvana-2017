import os
import queue
import threading
from _ast import Lambda

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.core import Lambda
from keras.preprocessing.image import array_to_img, load_img, img_to_array
from tensorflow.python.client import device_lib

from params import args
from utils import ThreadsafeIter

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

gpus = [x.name for x in device_lib.list_local_devices() if x.name[:4] == '/gpu']

n_threads = args.ensembling_cpu_threads
ensembling_dir = args.ensembling_dir
strategy = args.ensembling_strategy
dirs = args.dirs_to_ensemble
folds_dir = args.folds_dir
dirs = [os.path.join(folds_dir, d) for d in dirs]
filenames = sorted(os.listdir(dirs[0]))
nb_samples = len(filenames)
for d in dirs:
    if not os.path.exists(d):
        raise ValueError(d + " doesn't exist")
prediction_dir = args.pred_mask_dir

batch_size = args.pred_batch_size

batch_indices = [(start, min(start + batch_size, len(filenames))) for start in range(0, len(filenames), batch_size)]

batch_indices = ThreadsafeIter(batch_indices)


def data_loader(q, ):
    for bi in batch_indices:
        start, end = bi
        x_batch = []
        filenames_batch = filenames[start:end]

        for filename in filenames_batch:
            imgs = []
            for d in dirs:
                img = img_to_array(load_img(os.path.join(d, filename), grayscale=True))
                imgs.append(np.squeeze(img))
            x_batch.append(np.array(imgs).transpose((1, 2, 0)))
        q.put((filenames_batch, np.array(x_batch)))

    for gpu in gpus:
        q.put((None, None))


def predictor(q, gpu, pq):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    with sess.as_default():
        model = create_model(gpu)
        while True:
            batch_fnames, x_batch = q.get()
            if x_batch is None:
                break

            preds = model.predict_on_batch(x_batch)

            for i, pred in enumerate(preds):
                filename = batch_fnames[i]
                pq.put((os.path.join(ensembling_dir, filename[:-4] + ".png"), pred))


def file_writer(q, ):
    while True:
        filename, img_array = q.get()
        if filename is None:
            break
        array_to_img(img_array * 255).save(os.path.join(ensembling_dir, filename[:-4] + ".png"))


q_size = 100


def create_model(gpu):
    with tf.device(gpu):
        input = Input((1280, 1918, len(dirs)))
        x = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(input)
        model = Model(input, x)
        model.summary()
    return model


print('Ensembling on {} samples with batch_size = {}...'.format(len(filenames), batch_size))
q = queue.Queue(maxsize=1000)
threads = [threading.Thread(target=data_loader, name='DataLoader', args=(q,)) for t in range(n_threads//2)]
writing_queue = queue.Queue(maxsize=1000)

for i in range(n_threads//2):
    threads.append(threading.Thread(target=file_writer, name='DataWriter', args=(writing_queue,)))


for gpu in gpus:
    print("Starting ensembler at device " + gpu)

    t = threading.Thread(target=predictor, name='Ensembler', args=(q, gpu, writing_queue))
    threads.append(t)

for t in threads:
    t.start()

for t in threads:
    t.join()
