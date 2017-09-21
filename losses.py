import keras.backend as K
from keras.backend.tensorflow_backend import _to_tensor


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


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def dice_coef_loss_bce(y_true, y_pred, dice=0.5, bce=0.5, bootsrapping='hard', alpha=1.):
    return bootstrapped_crossentropy(y_true, y_pred, bootsrapping, alpha) * bce + dice_coef_loss(y_true, y_pred) * dice


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
            return dice_coef_loss_bce(y, p, dice=0.8, bce=0.2, bootsrapping='soft', alpha=1)

        return loss
    elif loss_name == 'boot_soft':
        def loss(y, p):
            return dice_coef_loss_bce(y, p, dice=0.8, bce=0.2, bootsrapping='soft', alpha=0.95)

        return loss
    elif loss_name == 'boot_hard':
        def loss(y, p):
            return dice_coef_loss_bce(y, p, dice=0.8, bce=0.2, bootsrapping='hard', alpha=0.95)

        return loss
    else:
        ValueError("Unknown loss.")
