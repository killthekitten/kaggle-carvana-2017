import keras.backend as K
from keras.backend.tensorflow_backend import _to_tensor
from keras.losses import binary_crossentropy


def dice_coef_clipped(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(K.round(y_true))
    y_pred_f = K.flatten(K.round(y_pred))
    intersection = K.sum(y_true_f * y_pred_f)
    return 100. * (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def bootstrapped_crossentropy(y_true, y_pred, bootstrap_type='hard', alpha=0.95):
    target_tensor = y_true
    prediction_tensor = y_pred
    _epsilon = _to_tensor(K.epsilon(), prediction_tensor.dtype.base_dtype)
    prediction_tensor = K.tf.clip_by_value(prediction_tensor, _epsilon, 1 - _epsilon)
    prediction_tensor = K.tf.log(prediction_tensor / (1 - prediction_tensor))

    if bootstrap_type == 'soft':
        bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * K.tf.sigmoid(prediction_tensor)
    else:
        bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * K.tf.cast(
            K.tf.sigmoid(prediction_tensor) > 0.5, K.tf.float32)
    return K.mean(K.tf.nn.sigmoid_cross_entropy_with_logits(
        labels=bootstrap_target_tensor, logits=prediction_tensor))


def online_bootstrapping(y_true, y_pred, pixels=512, threshold=0.5):
    """ Implements nline Bootstrapping crossentropy loss, to train only on hard pixels,
        see  https://arxiv.org/abs/1605.06885 Bridging Category-level and Instance-level Semantic Image Segmentation
        The implementation is a bit different as we use binary crossentropy instead of softmax
        SUPPORTS ONLY MINIBATCH WITH 1 ELEMENT!
    # Arguments
        y_true: A tensor with labels.

        y_pred: A tensor with predicted probabilites.

        pixels: number of hard pixels to keep

        threshold: confidence to use, i.e. if threshold is 0.7, y_true=1, prediction=0.65 then we consider that pixel as hard
    # Returns
        Mean loss value
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    difference = K.abs(y_true - y_pred)

    values, indices = K.tf.nn.top_k(difference, sorted=True, k=pixels)
    min_difference = (1 - threshold)
    y_true = K.tf.gather(K.gather(y_true, indices), K.tf.where(values > min_difference))
    y_pred = K.tf.gather(K.gather(y_pred, indices), K.tf.where(values > min_difference))

    return K.mean(K.binary_crossentropy(y_true, y_pred))


def dice_coef_loss_border(y_true, y_pred):
    return (1 - dice_coef_border(y_true, y_pred)) * 0.05 + 0.95 * dice_coef_loss(y_true, y_pred)

def bce_dice_loss_border(y_true, y_pred):
    return bce_border(y_true, y_pred) * 0.05 + 0.95 * dice_coef_loss(y_true, y_pred)


def dice_coef_border(y_true, y_pred):
    border = get_border_mask((21, 21), y_true)

    border = K.flatten(border)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.tf.gather(y_true_f, K.tf.where(border > 0.5))
    y_pred_f = K.tf.gather(y_pred_f, K.tf.where(border > 0.5))

    return dice_coef(y_true_f, y_pred_f)


def bce_border(y_true, y_pred):
    border = get_border_mask((21, 21), y_true)

    border = K.flatten(border)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.tf.gather(y_true_f, K.tf.where(border > 0.5))
    y_pred_f = K.tf.gather(y_pred_f, K.tf.where(border > 0.5))

    return binary_crossentropy(y_true_f, y_pred_f)


def get_border_mask(pool_size, y_true):
    negative = 1 - y_true
    positive = y_true
    positive = K.pool2d(positive, pool_size=pool_size, padding="same")
    negative = K.pool2d(negative, pool_size=pool_size, padding="same")
    border = positive * negative
    return border


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def dice_coef_loss_bce(y_true, y_pred, dice=0.5, bce=0.5, bootstrapping='hard', alpha=1.):
    return bootstrapped_crossentropy(y_true, y_pred, bootstrapping, alpha) * bce + dice_coef_loss(y_true, y_pred) * dice


def make_loss(loss_name):
    if loss_name == 'crossentropy':
        return K.binary_crossentropy
    elif loss_name == 'crossentropy_boot':
        def loss(y, p):
            return bootstrapped_crossentropy(y, p, 'hard', 0.9)
        return loss
    elif loss_name == 'dice':
        return dice_coef_loss
    elif loss_name == 'bce_dice':
        def loss(y, p):
            return dice_coef_loss_bce(y, p, dice=0.8, bce=0.2, bootstrapping='soft', alpha=1)

        return loss
    elif loss_name == 'boot_soft':
        def loss(y, p):
            return dice_coef_loss_bce(y, p, dice=0.8, bce=0.2, bootstrapping='soft', alpha=0.95)

        return loss
    elif loss_name == 'boot_hard':
        def loss(y, p):
            return dice_coef_loss_bce(y, p, dice=0.8, bce=0.2, bootstrapping='hard', alpha=0.95)

        return loss
    elif loss_name == 'online_bootstrapping':
        def loss(y, p):
            return online_bootstrapping(y, p, pixels=512 * 64, threshold=0.7)

        return loss
    elif loss_name == 'dice_coef_loss_border':
        return dice_coef_loss_border
    elif loss_name == 'bce_dice_loss_border':
        return bce_dice_loss_border
    else:
        ValueError("Unknown loss.")
