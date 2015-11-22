from ANN_v1 import blind_test
import basics.mnist_basics as mnist
import numpy as np
# import theano
# from theano import tensor as T
# from theano.sandbox.rng_mrg import MRG_RandomStreams as rs
# import sys

# srng = rs()


# def load_images():
#     global training_images, training_true_dist,testing_images, testing_true_dist
#     training_images, training_true_dist = load_image_arrays('training')
#     testing_images, testing_true_dist = load_image_arrays('testing')
#
#     training_images = scale_to_one(training_images)
#     testing_images = scale_to_one(testing_images)
#
# def scale_to_one(feature_set):
#     return (np.array(feature_set) / 255.0).tolist()
#
# def load_image_arrays(image_type):
#     image_array, labels = mnist.load_all_flat_cases(image_type)
#     np_image_array = np.array(image_array, dtype=theano.config.floatX)
#     np_true_dist = create_true_dist(labels)
#     return image_array, np_true_dist
#
# def create_true_dist(labels):
#     true_dist = np.zeros(shape=(len(labels), 10))
#     for i in range(len(labels)):
#         label_index = labels[i]
#         true_dist[i][label_index] = 1
#     return true_dist
#
# def dropout(X, p=0.):
#     if p > 0:
#         retain_prob = 1 - p
#         X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
#         X /= retain_prob
#         print(X)
#     return X

# load_images()

features, labels = mnist.load_flat_cases("demo_prep")

prediction_results = blind_test(features)
# image = mnist.reconstruct_image(features[27])
# print("Prediction:\t",prediction_results)
# print("Labels:\t\t", labels)
print("Blind test success rate:", np.mean(np.array(prediction_results) == np.array(labels)))
# t_i = np.array(training_images[0])
# print(t_i.shape, t_i.ctypes)
# image = mnist.reconstruct_image(t_i.tolist())
# print(dropout(t_i, 0.3))
#
# A = T.fmatrix
# y = A
# test = theano.function(inputs=[A], outputs=y, allow_input_downcast=True)
#
# print("func created")
# sys.stdout.flush()
# print(test(dropout(t_i, 0.3)))
#
#
# noisy_image = mnist.reconstruct_image(dropout(t_i, 0.3).tolist())
# mnist.show_digit_image(image)
# mnist.show_digit_image(noisy_image)
#
