import pickle
import numpy as np
import os

training_data_start = 10
training_data_end = 11
validation_data_start = 12
validation_data_end = 13
testing_data_start = 15
testing_data_end = 16

__training_path__ = os.path.join(os.path.dirname(__file__), '')

def load2(start, stop):
    boards = []
    moves = []

    for i in range(start, stop):
        file = os.path.join(__training_path__, 'training_data'+str(i))
        with open(file, 'rb') as f:
            rotated_boards = []
            rotated_moves = []
            training_data = pickle.load(f)
            training_boards = training_data[0]
            training_moves = training_data[1]
            for j, board in enumerate(training_boards):
                move = training_moves[j]
                b,m = generate_rotated_boards_and_moves(board, move)
                rotated_boards += b
                rotated_moves += m
            boards.append(rotated_boards)
            moves.append(rotated_moves)
    flattened_boards = [board for sublist in boards for board in sublist]
    flattened_moves = [move for sublist in moves for move in sublist]
    return [np.array(flattened_boards), np.array(flattened_moves)]

def load(start, stop):
    boards = []
    moves = []
    for i in range(start, stop):
        file = os.path.join(__training_path__, 'markus_data'+str(i)+'.txt')
        with open(file, 'r') as f:
            lines = f.read().splitlines()
            for i, val in enumerate(lines):
                values = val.split(";")
                move = np.array(values[1]).astype(int)
                board = np.array(values[0].split(",")).astype(int)
                boards.append(board)
                moves.append(move)
    return [np.array(boards), np.array(moves)]


def unflatten(flat_array,dims=(4,4)):
    array = np.array(flat_array)
    array = np.reshape(array,dims)
    return array

def flatten(array):
    return [row for sublist in array for row in sublist]

def scale_boards(boards, log2):
    scaled_boards = []
    for i, board in enumerate(boards):
        scaled_boards.append(scale(board, log2))
    return scaled_boards

def generate_rotated_boards_and_moves(board, move):
    rotated_boards = []
    rotated_moves = []
    #for i in range(4):
        #board = rotate_board_clockwise(board)
        #rotated_boards.append(board)
        #move = rotated_move_clockwise(move)
        #rotated_moves.append(move)
    rotated_moves.append(move)
    rotated_boards.append(board)
    return rotated_boards, rotated_moves


def rotate_board_clockwise(board):
    unflattened_board = unflatten(board)
    rotated_board = np.asarray(list(zip(*unflattened_board[::-1])))
    return flatten(rotated_board)

def rotated_move_clockwise(move):
    if move == 0: #UP
        return 3
    if move == 3: #RIGHT
        return 1
    if move == 1: #DOWN
        return 2
    if move == 2: #LEFT
        return 0

def scale(board, log2):
    highest_value = 0.0
    for j, val in enumerate(board):
        if val > highest_value:
            highest_value = val
    if log2:
        board = board/highest_value
    else:
        board = (2**board)/(2**highest_value)
    return board

def load_training_data(log2=True):
    data = load(training_data_start, training_data_end)
    boards = scale_boards(data[0], log2)
    moves = data[1]
    print("Training data loaded, "+str(len(boards))+" cases")
    return boards, moves


def load_testing_data(log2=True):
    data = load(testing_data_start, testing_data_end)
    boards = scale_boards(data[0], log2)
    moves = data[1]
    print("Testing data loaded, "+str(len(boards))+" cases")
    return boards, moves

def load_validation_data(log2=True):
    data = load(validation_data_start, validation_data_end)
    boards = scale_boards(data[0], log2)
    moves = data[1]
    print("validation data loaded, "+str(len(boards))+" cases")
    return [boards, moves]

def load_cases(data_type, log2=True):
    if data_type == 'training':
        return load_training_data(log2)
    else:
        return load_testing_data(log2)




