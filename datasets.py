import numpy as np
from params import args
from sklearn.model_selection import train_test_split, shuffle
from random_transform_mask import ImageWithMaskFunction

def generate_filenames(car_ids):
    return ['{}_{}'.format(id, str(angle + 1).zfill(2)) for angle in range(16) for id in ids_train_split]

def bootstrapped_split(car_ids, seed=args.seed):
    """
    # Arguments
        metadata: metadata.csv provided by Carvana (should include
        `train` column).

    # Returns
        A tuple (train_ids, test_ids)
    """
    all_ids = pd.Series(car_ids)
    train_ids, valid_ids = train_test_split(car_ids, test_size=args.test_size_float,
                                                     random_state=seed)

    np.random_seed(seed)
    bootstrapped_idx = np.random.random_integers(0, len(train_ids), seed=args.seed)
    bootstrapped_train_ids = train_ids[bootstrapped_idx]

    return generate_filenames(bootstrapped_train_ids.values),
           generate_filenames(valid_ids)

def build_batch_generator(filenames, dataset_dir=args.dataset_dir, batch_size=args.batch_size, shuffle=False, transformations=None,
                          out_size=None, crop_size=None, mask_dir=None, aug=False):
    mask_function = ImageWithMaskFunction(out_size=out_size, crop_size=crop_size, mask_dir=mask_dir)

    def batch_generator():
        while True:
            # @TODO: Should we fixate the seed here?
            if shuffle:
                filenames = shuffle(filenames)

            for start in range(0, len(filenames), batch_size):
                x_batch = []
                y_batch = []
                end = min(start + batch_size, len(filenames))
                train_batch = filenames[start:end]
                for filename in filenames:
                    img = cv2.imread(os.path.join(dataset_dir, 'train', '{}.jpg'.format(filename))

                    x_batch.append(img))
                x_batch = np.array(x_batch, np.float32) / 255

                yield mask_function.mask_pred(x_batch, filenames[start:end], range(batch_size), aug)

    return batch_generator
