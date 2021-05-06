from silence_tensorflow import silence_tensorflow
import numpy as np
import collections

# ログ抑制（tf の import 前に実行する）
silence_tensorflow()

import tensorflow as tf
import tensorflow_federated as tff


# TODO(b/148678573,b/148685415): must use the reference context because it
# supports unbounded references and tff.sequence_* intrinsics.
tff.backends.reference.set_reference_context()

# 訓練回数
NUM_TRAIN = 5

def my_training_model(federated_train_data, federated_test_data):

    # Debug
    # print(federated_train_data[5][-1]['y'])

    # Defining a loss function
    BATCH_SPEC = collections.OrderedDict(
        x=tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
        y=tf.TensorSpec(shape=[None], dtype=tf.int32))
    BATCH_TYPE = tff.to_type(BATCH_SPEC)

    MODEL_SPEC = collections.OrderedDict(
        weights=tf.TensorSpec(shape=[784, 10], dtype=tf.float32),
        bias=tf.TensorSpec(shape=[10], dtype=tf.float32))
    MODEL_TYPE = tff.to_type(MODEL_SPEC)

    # NOTE: `forward_pass` is defined separately from `batch_loss` so that it can
    # be later called from within another tf.function. Necessary because a
    # @tf.function  decorated method cannot invoke a @tff.tf_computation.

    @tf.function
    def forward_pass(model, batch):
        predicted_y = tf.nn.softmax(
            tf.matmul(batch['x'], model['weights']) + model['bias'])
        return -tf.reduce_mean(
            tf.reduce_sum(
                tf.one_hot(batch['y'], 10) * tf.math.log(predicted_y), axis=[1]))

    @tff.tf_computation(MODEL_TYPE, BATCH_TYPE)
    def batch_loss(model, batch):
        return forward_pass(model, batch)

    initial_model = collections.OrderedDict(
        weights=np.zeros([784, 10], dtype=np.float32),
        bias=np.zeros([10], dtype=np.float32))

    # Gradient descent on a single batch
    @tff.tf_computation(MODEL_TYPE, BATCH_TYPE, tf.float32)
    def batch_train(initial_model, batch, learning_rate):
        # Define a group of model variables and set them to `initial_model`. Must
        # be defined outside the @tf.function.
        model_vars = collections.OrderedDict([
            (name, tf.Variable(name=name, initial_value=value))
            for name, value in initial_model.items()
        ])
        optimizer = tf.keras.optimizers.SGD(learning_rate)

        @tf.function
        def _train_on_batch(model_vars, batch):
            # Perform one step of gradient descent using loss from `batch_loss`.
            with tf.GradientTape() as tape:
                loss = forward_pass(model_vars, batch)
            grads = tape.gradient(loss, model_vars)
            optimizer.apply_gradients(
                zip(tf.nest.flatten(grads), tf.nest.flatten(model_vars)))
            return model_vars

        return _train_on_batch(model_vars, batch)

    # Gradient descent on a sequence of local data
    print()
    print("----- Gradient descent on a sequence of local data -----")
    LOCAL_DATA_TYPE = tff.SequenceType(BATCH_TYPE)

    @tff.federated_computation(MODEL_TYPE, tf.float32, LOCAL_DATA_TYPE)
    def local_train(initial_model, learning_rate, all_batches):

        # Mapping function to apply to each batch.
        @tff.federated_computation(MODEL_TYPE, BATCH_TYPE)
        def batch_fn(model, batch):
            return batch_train(model, batch, learning_rate)

        return tff.sequence_reduce(all_batches, initial_model, batch_fn)

    # Local evaluation
    print()
    print("----- Local evaluation -----")

    @tff.federated_computation(MODEL_TYPE, LOCAL_DATA_TYPE)
    def local_eval(model, all_batches):
        # TODO(b/120157713): Replace with `tff.sequence_average()` once implemented.
        return tff.sequence_sum(
            tff.sequence_map(
                tff.federated_computation(
                    lambda b: batch_loss(model, b), BATCH_TYPE),
                all_batches))

    # Federated evaluation
    print()
    print("----- Federated evaluation -----")
    SERVER_MODEL_TYPE = tff.type_at_server(MODEL_TYPE)
    CLIENT_DATA_TYPE = tff.type_at_clients(LOCAL_DATA_TYPE)

    @tff.federated_computation(SERVER_MODEL_TYPE, CLIENT_DATA_TYPE)
    def federated_eval(model, data):
        return tff.federated_mean(
            tff.federated_map(local_eval, [tff.federated_broadcast(model), data]))

    # Federated training
    print()
    print("----- Federated training -----")
    SERVER_FLOAT_TYPE = tff.type_at_server(tf.float32)

    @tff.federated_computation(SERVER_MODEL_TYPE, SERVER_FLOAT_TYPE, CLIENT_DATA_TYPE)
    def federated_train(model, learning_rate, data):
        return tff.federated_mean(
            tff.federated_map(local_train, [
                tff.federated_broadcast(model),
                tff.federated_broadcast(learning_rate), data
            ]))

    model = initial_model
    learning_rate = 0.1
    li_loss = []
    for round_num in range(NUM_TRAIN):
        model = federated_train(model, learning_rate, federated_train_data)
        learning_rate = learning_rate * 0.9
        loss = federated_eval(model, federated_train_data)
        print('round {}, loss={}'.format(round_num, loss))

    print('initial_model test loss =', federated_eval(initial_model, federated_test_data))
    print('trained_model test loss =', federated_eval(model, federated_test_data))

    return model


# if __name__ == '__main__':
#    print(my_training_model(mnist_train, mnist_test))


