import theano
from theano import tensor as T
import numpy as np
import json
import sys
import time
import pickle
import os
import random
import ai2048demo
from game2048 import load2048 as load2048
from game2048 import Game2048 as game2048

__author__ = 'markus&sondremare'

json_file_path = os.path.join(os.path.dirname(__file__), 'ANN_json_init.txt')
use_existing_net = False
number_of_epochs = 50
size_of_training_batch = 100
lmbda = 0.1

# Global variables defined in json file
global hidden_layers, possible_activation_funcs, activation_functions, learning_rate, possible_error_funcs
global error_func_index, activation_func_names, error_func_name


def main():
    # Get network setup from json file
    load_json_variables()

    init_print_variables()

    flush_print("Loading images")
    load_images()

    # Init weights
    global weights
    weights = init_weights(hidden_layers)

    flush_print("Preparing theano functions")
    init_theano_functions()

    if use_existing_net:
        load_weights()
        init_theano_functions()
    else:
        train_net()
        save_weights()


def init_print_variables():
    global activation_func_names, error_func_name
    activation_func_names = "[" + str(possible_activation_funcs[activation_functions[0]])
    for i in range(1, len(activation_functions)):
        activation_func_names += ", " + str(possible_activation_funcs[activation_functions[i]])
    activation_func_names += "]"
    error_func_name = possible_error_funcs[error_func_index]
    print("Creating ANN with hidden layers:", hidden_layers, "\t Learning rate:", learning_rate,
          "\tActivation functions:", activation_func_names, "\tError function:", error_func_name)


def flush_print(s):
    print(s)
    sys.stdout.flush()


def blind_test(feature_sets):
    feature_sets = np.array(feature_sets)
    feature_sets = scale_to_one(feature_sets)
    prediction_results = predict(feature_sets)
    return prediction_results.tolist()


def load_image_arrays(image_type):
    board_array, labels = load2048.load_cases(image_type)
    np_image_array = np.array(board_array, dtype=theano.config.floatX)
    np_true_dist = create_true_dist(labels)
    return np_image_array, np_true_dist


def create_true_dist(labels):
    true_dist = np.zeros(shape=(len(labels), 4))
    for i in range(len(labels)):
        label_index = labels[i]
        true_dist[i][label_index] = 1
    return true_dist


def scale_to_one(feature_set):
    return feature_set / 255.0


def load_json_variables():
    f = open(json_file_path, "r")
    variable_dict = json.load(f)
    globals().update(variable_dict)


def init_weights(hidden_node_config):
    from_nodes = 16
    weight_list = []
    for i in range(0, len(hidden_node_config)):
        to_nodes = hidden_node_config[i]
        weight_list.append(create_small_random_weights(from_nodes, to_nodes))
        from_nodes = to_nodes
    weight_list.append(create_small_random_weights(to_nodes, 4))
    return weight_list


def create_small_random_weights(from_nodes, to_nodes):
    return theano.shared(np.array(np.random.randn(from_nodes, to_nodes) * 0.05, dtype=theano.config.floatX))


def rectify(X):
    return T.maximum(X, 0.)


def get_activation_function(activation_func_index):
    activation_func_name = possible_activation_funcs[activation_func_index]
    if activation_func_name == "rectify":
        return rectify
    elif activation_func_name == "hyp_tang":
        return T.tanh
    elif activation_func_name == "sigmoid":
        return T.nnet.sigmoid


def squared_error(prob_dist, Y):
    return T.mean((Y - prob_dist) ** 2)


def cross_entropy(prob_dist, Y):
    return T.mean(T.nnet.categorical_crossentropy(prob_dist, Y))


def get_error_func():
    if error_func_name == "squared_errors":
        return squared_error
    elif error_func_name == "cross_entropy":
        return cross_entropy


def load_images():
    global training_images, training_true_dist, testing_images, testing_true_dist
    training_images, training_true_dist = load_image_arrays('training')
    testing_images, testing_true_dist = load_image_arrays('testing')


def predict_prob_dist(X):
    prev_layer = X
    hidden_layer_list = weights[:-1]
    for i in range(len(hidden_layer_list)):
        activation_func = get_activation_function(activation_functions[i])
        hidden_layer = hidden_layer_list[i]
        prev_layer = activation_func(T.dot(prev_layer, hidden_layer))

    # Lastly, softmax makes sure the sum of output probabilities sum to one
    return T.nnet.softmax(T.dot(prev_layer, weights[-1])) + 0.00000001


def gradient_descend(cost):
    gradients = T.grad(cost, weights)
    updates = []
    for weight, grad in zip(weights, gradients):
        updates.append([weight, weight - grad * learning_rate])
    return updates

def size(data):
    "Return the size of the dataset `data`."
    return data.shape[0]


def init_theano_functions():
    # Init theano function variables
    X = T.fmatrix("X")
    Y = T.fmatrix("Y")

    error_func = get_error_func()
    num_training_batches = size(training_images)/size_of_training_batch

    prob_dist = predict_prob_dist(X)
    l2_norm_squared = sum([(w**2).sum() for w in weights])
    cost = error_func(prob_dist, Y)
    cost += l2_norm_squared*lmbda/num_training_batches
    update = gradient_descend(cost)
    prediction = T.argmax(prob_dist, 1)

    global train, predict
    train = theano.function(inputs=[X, Y], outputs=[], updates=update, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=[prob_dist, prediction], allow_input_downcast=True)



def train_net():
    # Run ANN
    print("Running training sequence:")
    start_time_milli = int(round(time.time() * 1000))
    for i in range(number_of_epochs):
        batch_start = range(0, len(training_images), size_of_training_batch)
        batch_end = range(size_of_training_batch, len(training_images), size_of_training_batch)
        for start, end in zip(batch_start, batch_end):
            x = training_images[start: end]
            y = training_true_dist[start: end]
            train(x, y)
        prediction_results = predict(testing_images)[1]
        success_rate = np.mean(prediction_results == np.argmax(testing_true_dist, 1))
        print("Run #" + str(i), success_rate)
    print("Do you want to play a game?")
    play2048()

    # end_time_milli = int(round(time.time() * 1000))
    # run_time = (end_time_milli - start_time_milli) / 1000
    # print_final_stats(run_time, success_rate)


def setupGame():
    game = game2048.Game2048()
    return game

def scaleBoard(game):
    flattened = [row for sublist in game.game_matrix for row in sublist]
    highestValue = game.get_highest_value()
    scaled = np.log2(flattened)/np.log2(highestValue)
    scaled[scaled == -np.inf] = 0
    return np.asarray(scaled, dtype=theano.config.floatX)

def playMove(game):
    if not game.is_game_over():
        board = scaleBoard(game)
        prediction_results = predict([board])
        prob = prediction_results[0]
        move = np.argmax(prob, 1)[0]
        game.move(move)
        while (game.old_equals_new_game_matrix()):
            prob[0][move] = 0
            move = np.argmax(prob, 1)[0]
            game.move(move)
    else:
        print("game is over")

def play2048(rounds=50):
    n = 0
    own_results = []
    random_results= []
    while n < rounds:
        game = setupGame()
        while not game.is_game_over():
            playMove(game)
        own_results.append(game.get_highest_value())
        game = setupGame()
        while not game.is_game_over():
            random_move = random.randint(0, 4)
            game.move(random_move)
        random_results.append(game.get_highest_value())
        n += 1
    print(own_results)
    print(random_results)
    score = ai2048demo.welch(random_results, own_results)
    print("Demo score:",score)

def print_final_stats(run_time, success_rate):
    print("\n=================FINAL STATS==================")
    print("Hidden layer sizes:\t\t", hidden_layers, "\nLayer activation func:\t", activation_func_names,
          "\nError function:\t\t\t", error_func_name, "\nLearning rate:\t\t\t", learning_rate,
          "\nNumber of epochs:\t\t", number_of_epochs,
          "\nTotal time spent:\t\t", "%.3f" % run_time, "seconds\nTime per epoch:\t\t\t",
          "%.3f" % (run_time / number_of_epochs), "seconds",
          "\n\nFinal success rate:\t\t", success_rate)
    print("============================================")
    sys.stdout.flush()


def save_weights():
    with open("ann_weights", 'wb') as f:
        pickle.dump(weights, f, pickle.HIGHEST_PROTOCOL)


def load_weights():
    global weights
    with open("ann_weights", 'rb') as f:
        weights = pickle.load(f)


main()
