import numpy as np
import random
import math


class Game2048:
    def __init__(self):
        self.matrix_size = 4
        self.game_matrix = np.zeros(shape=(self.matrix_size, self.matrix_size), dtype=int)
        self.old_game_matrix = None
        self.spawn_random_tile()
        self.spawn_random_tile()

    # Functions for spawning new random tiles
    # -----------------------------------------
    def spawn_random_tile(self):
        empty_cells = self.find_all_empty_cells()
        if empty_cells:
            random_cell = self.select_random_cell(empty_cells)
            self.spawn_in_cell(random_cell)

    def find_all_empty_cells(self):
        empty_cells = []
        for x in range(self.matrix_size):
            for y in range(self.matrix_size):
                if self.game_matrix[x][y] == 0:
                    empty_cells.append([x, y])

        return empty_cells

    def select_random_cell(self, cell_list):
        return random.choice(cell_list)

    # cell_pos is a tuple with the x and y coordinates of the cell
    def spawn_in_cell(self, cell_pos):
        new_tile_value = self.get_random_tile_value()
        self.game_matrix[cell_pos[0], cell_pos[1]] = new_tile_value

    def get_random_tile_value(self):
        random_float = random.uniform(0, 1)
        if random_float < 0.9:
            return 2
        else:
            return 4

    # ------------------------------------------

    # Functions checking if the game is over
    # ------------------------------------------
    def is_game_over(self):
        return self.all_cells_filled() and self.no_possible_moves_left()

    def all_cells_filled(self):
        for x in range(self.matrix_size):
            for y in range(self.matrix_size):
                if self.game_matrix[x][y] == 0:
                    return False
        return True

    def no_possible_moves_left(self):
        for x in range(self.matrix_size):
            for y in range(self.matrix_size):
                if self.can_merge_with_neighbor(x, y):
                    return False
        return True

    def can_merge_with_neighbor(self, x, y):
        value = self.game_matrix[x][y]
        for deltaX in range(-1, 2):
            for deltaY in range(-1,2):
                newX = x + deltaX
                newY = y + deltaY
                is_in_bounds = newX < self.matrix_size and newX >= 0 and newY < self.matrix_size and newY >= 0
                is_same_position = newX == x and newY == y
                is_diagonal = abs(deltaX) == 1 and abs(deltaY) == 1
                is_equal = is_in_bounds and self.game_matrix[newX][newY] == value;
                if is_in_bounds and not is_same_position and not is_diagonal and is_equal:
                    return True
        return False


    # General merging and shifting functions
    # ------------------------------------------
    def try_to_shift_tiles(self, shift_from_pos, shift_to_pos):
        shift_from_value = self.get_value_at_pos(shift_from_pos)
        shift_to_value = self.get_value_at_pos(shift_to_pos)
        if shift_from_value != 0:
            if shift_to_value == 0:
                self.swap_tiles(shift_from_pos, shift_to_pos)
            return True
        return False

    def swap_tiles(self, shift_from_pos, shift_to_pos):
        temp = self.get_value_at_pos(shift_to_pos)
        self.game_matrix[shift_to_pos[0]][shift_to_pos[1]] = self.get_value_at_pos(shift_from_pos)
        self.game_matrix[shift_from_pos[0]][shift_from_pos[1]] = temp

    # Merging_horizontally is a boolean. True when merging left/right, False when merging up/down
    def merge_tiles(self, current_tile_pos, prev_tile_pos, merging_horizontally):
        current_tile_value = self.get_value_at_pos(current_tile_pos)
        if -1 not in prev_tile_pos:
            prev_tile_value = self.get_value_at_pos(prev_tile_pos)
        else:
            prev_tile_value = -1

        if prev_tile_value != 0 and current_tile_value == prev_tile_value:
            self.force_merge(current_tile_pos, prev_tile_pos)
            prev_tile_pos[merging_horizontally] = -1
        elif current_tile_value != 0:
            prev_tile_pos[merging_horizontally] = current_tile_pos[merging_horizontally]

        return prev_tile_pos

    def force_merge(self, remove_tile_pos, keep_tile_pos):
        self.game_matrix[keep_tile_pos[0]][keep_tile_pos[1]] = self.get_value_at_pos(remove_tile_pos) * 2
        self.game_matrix[remove_tile_pos[0]][remove_tile_pos[1]] = 0

    # Tile_pos is a list of size 2 with the x and y position of the tile
    def get_value_at_pos(self, tile_pos):
        return self.game_matrix[tile_pos[0]][tile_pos[1]]

    def get_highest_value(self):
        return self.game_matrix.max()

    # ---------------------------------------------------------------------------------------

    # Functions for moving in different directions
    # ---------------------------------------------------------------------------------------
    def move_right(self):
        self.old_game_matrix = np.copy(self.game_matrix)
        for x in range(self.matrix_size):
            self.merge_tiles_right(x)
            self.shift_row_right(x)
        self.complete_move()

    def merge_tiles_right(self, x):
        prev_tile_pos = [x, -1]
        for y in range(self.matrix_size - 1, -1, -1):
            current_tile_pos = [x, y]
            prev_tile_pos = self.merge_tiles(current_tile_pos, prev_tile_pos, True)

    def shift_row_right(self, x):
        number_of_moved_tiles = 0
        max_index = self.matrix_size - 1
        y = max_index
        while y >= 0:
            if self.try_to_shift_tiles([x, y], [x, max_index - number_of_moved_tiles]):
                number_of_moved_tiles += 1
                y = max_index - number_of_moved_tiles
            else:
                y -= 1

    def move_left(self):
        self.old_game_matrix = np.copy(self.game_matrix)
        for x in range(self.matrix_size):
            self.merge_tiles_left(x)
            self.shift_row_left(x)
        self.complete_move()

    def merge_tiles_left(self, x):
        prev_tile_pos = [x, -1]
        for y in range(self.matrix_size):
            current_tile_pos = [x, y]
            prev_tile_pos = self.merge_tiles(current_tile_pos, prev_tile_pos, True)

    def shift_row_left(self, x):
        number_of_moved_tiles = 0
        y = 1
        while y < self.matrix_size:
            if self.try_to_shift_tiles([x, y], [x, number_of_moved_tiles]):
                number_of_moved_tiles += 1
                y = number_of_moved_tiles
            else:
                y += 1

    def move_up(self):
        self.old_game_matrix = np.copy(self.game_matrix)
        for x in range(self.matrix_size):
            self.merge_tiles_up(x)
            self.shift_row_up(x)
        self.complete_move()

    def merge_tiles_up(self, y):
        prev_tile_pos = [-1, y]
        for x in range(self.matrix_size):
            current_tile_pos = [x, y]
            prev_tile_pos = self.merge_tiles(current_tile_pos, prev_tile_pos, False)

    def shift_row_up(self, y):
        number_of_moved_tiles = 0
        x = 1
        while x < self.matrix_size:
            if self.try_to_shift_tiles([x, y], [number_of_moved_tiles, y]):
                number_of_moved_tiles += 1
                x = number_of_moved_tiles
            else:
                x += 1

    def move_down(self):
        self.old_game_matrix = np.copy(self.game_matrix)
        for x in range(self.matrix_size):
            self.merge_tiles_down(x)
            self.shift_row_down(x)
        self.complete_move()

    def merge_tiles_down(self, y):
        prev_tile_pos = [-1, y]
        for x in range(self.matrix_size - 1, -1, -1):
            current_tile_pos = [x, y]
            prev_tile_pos = self.merge_tiles(current_tile_pos, prev_tile_pos, False)

    def shift_row_down(self, y):
        number_of_moved_tiles = 0
        max_index = self.matrix_size - 1
        x = max_index
        while x >= 0:
            if self.try_to_shift_tiles([x, y], [max_index - number_of_moved_tiles, y]):
                number_of_moved_tiles += 1
                x = max_index - number_of_moved_tiles
            else:
                x -= 1

    def move(self, direction):
        if direction == 0:
            self.move_up()
        elif direction == 1:
            self.move_down()
        elif direction == 2:
            self.move_left()
        elif direction == 3:
            self.move_right()

    # ---------------------------------------------------------------------------------------

    def complete_move(self):
        if not self.old_equals_new_game_matrix():
            self.spawn_random_tile()

    def old_equals_new_game_matrix(self):
        for x in range(self.matrix_size):
            for y in range(self.matrix_size):
                old_val = self.old_game_matrix[x][y]
                new_val = self.game_matrix[x][y]
                if old_val != new_val:
                    return False
        return True


    def print(self):
        print_string = ""
        for x in range(self.matrix_size):
            print_string += "["
            for y in range(self.matrix_size):
                print_string += str(self.game_matrix[x][y]) + " "
            print_string += "]\n"

        print(print_string)



