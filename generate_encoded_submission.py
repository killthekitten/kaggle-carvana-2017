# from utils import encode_predictions
from params import args
from datetime import datetime
import time
import numpy as np
import os
import threading
import queue
import pandas as pd
from scipy import ndimage

def rle_encode(mask_image):
    pixels = mask_image.flatten()
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

def encoder(in_queue, threshold, generated_masks, time_counts):
    while True:
        img_name, mask_img_path = in_queue.get()

        if img_name is None:
            break

        t0 = time.clock()
        mask_img = ndimage.imread(mask_img_path, mode='L')
        mask_img[mask_img <= threshold] = 0
        mask_img[mask_img > threshold] = 1
        time_counts['time_read'].append(time.clock() - t0)

        t0 = time.clock()
        rle = rle_encode(mask_img)
        time_counts['time_rle'].append(time.clock() - t0)

        t0 = time.clock()
        rle_string = rle_to_string(rle)
        time_counts['time_stringify'].append(time.clock() - t0)

        generated_masks.append((img_name, rle_string))

def encode_predictions(predicted_dir, filenames, n_threads=None, threshold=127):
    print('Predicting RLE encoding masks using {} threads...'.format(n_threads))

    num_masks = 0

    # @TODO: Lacks proper concurrency primitives, refactor
    time_counts = {
        'time_read': [],
        'time_rle': [],
        'time_stringify': []
    }

    filepaths_queue = queue.Queue()
    generated_masks = []
    threads = []

    for i in range(n_threads):
        threads.append(threading.Thread(target=encoder,
                                        name='Encoder',
                                        args=(filepaths_queue,
                                              threshold,
                                              generated_masks,
                                              time_counts)))

    for img_name in filenames:
        mask_img_path = os.path.join(predicted_dir, img_name)

        filepaths_queue.put((img_name, mask_img_path))

    for thread in threads:
        filepaths_queue.put((None, None))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    time_read = sum(time_counts['time_read'])
    time_rle = sum(time_counts['time_rle'])
    time_stringify = sum(time_counts['time_stringify'])

    num_masks = len(filenames)
    print('Time spent reading mask images:', time_read, 's, =>', 1000 * (time_read / num_masks), 'ms per mask.')
    print('Time spent RLE encoding masks:', time_rle, 's, =>', 1000 * (time_rle / num_masks), 'ms per mask.')
    print('Time spent stringifying RLEs:', time_stringify, 's, =>', 1000 * (time_stringify / num_masks), 'ms per mask.')

    return generated_masks

def main():
    input_df = pd.read_csv(args.pred_sample_csv)
    input_filenames = input_df['img'].str.replace('.jpg', '.png')

    output_filename = 'submission-{:%Y-%m-%d_%H-%M-%S}.csv.gz'.format(datetime.now())
    output_path = os.path.join(args.submissions_dir, output_filename)
    print('{} masks found. Reading from {}, saving to {}'.format(len(input_filenames), args.pred_mask_dir, output_path))
    generated_masks = encode_predictions(args.pred_mask_dir, input_filenames, n_threads=args.pred_threads)

    output_df = pd.DataFrame(generated_masks, columns=['img', 'rle_mask'])
    output_df['img'] = output_df.img.str.replace('.png', '.jpg')
    output_df.to_csv(output_path, compression='gzip', index=False)


if __name__ == '__main__':
    main()
