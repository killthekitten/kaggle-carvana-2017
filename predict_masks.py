import os
from time import clock

import numpy as np
from keras.preprocessing.image import array_to_img, img_to_array, load_img

from models import get_unet_resnet
from params import args
from utils import preprocess_input_resnet

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

prediction_dir = args.pred_mask_dir


def predict():
    output_dir = args.pred_mask_dir
    model = get_unet_resnet((None, None, 3))
    model.load_weights(args.weights)
    batch_size = args.pred_batch_size
    nbr_test_samples = 100064

    filenames = [os.path.join(args.test_data_dir, f) for f in sorted(os.listdir(args.test_data_dir))]

    start_time = clock()
    for i in range(int(nbr_test_samples / batch_size) + 1):
        x = []
        for j in range(batch_size):
            if i * batch_size + j < len(filenames):
                img = load_img(filenames[i * batch_size + j], target_size=(args.img_height, args.img_width))
                x.append(preprocess_input_resnet(img_to_array(img)))
        x = np.array(x)
        batch_x = np.zeros((x.shape[0], 1280, 1920, 3))
        batch_x[:, :, 1:-1, :] = x
        preds = model.predict_on_batch(batch_x)
        for j in range(batch_size):
            filename = filenames[i * batch_size + j]
            prediction = preds[j][:, 1:-1, :]
            array_to_img(prediction * 255).save(os.path.join(output_dir, filename.split('/')[-1][:-4] + ".png"))
        time_spent = clock() - start_time
        print("predicted batch ", str(i))
        print("Time spent: {:.2f}  seconds".format(time_spent))
        print("Speed: {:.2f}  ms per image".format(time_spent / (batch_size * (i + 1)) * 1000))
        print("Elapsed: {:.2f} hours  ".format(time_spent / (batch_size * (i + 1)) / 3600 * (nbr_test_samples - (batch_size * (i + 1)))))

if __name__ == '__main__':
    predict()
