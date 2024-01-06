import tensorflow as tf

class EvaluationMetrics:
    @staticmethod
    def dice_coef(y_true, y_pred, threshold=0.5):
        # Convert predicted values to binary using the threshold
        y_pred_binary = tf.cast(tf.math.greater(y_pred, threshold), dtype=tf.float32)

        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred_binary)

        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)

        return (2.0 * intersection + 1.0) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1.0)

    @staticmethod
    def IoU_coef(y_true, y_pred):
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (intersection + 1.0) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection + 1.0)

    @staticmethod
    def IoU_loss(y_true, y_pred):
        return -EvaluationMetrics.IoU_coef(y_true, y_pred)