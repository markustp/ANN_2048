import Game2048

game = Game2048()

while not game.is_game_over():
    game.print()
    key = input()
    if key == 'w':
        game.move_up()
    if key == 's':
        game.move_down()
    if key == 'a':
        game.move_left()
    if key == 'd':
        game.move_right()


print("GAME OVER")