import numpy as np
import random


# Monty Hall Problem:
# You're on a game show, and you're given the choice of three doors:
# Behind one door is a car; behind the others, goats. You pick a door.

# The host must always open a door that was not selected by the contestant.
# The host must always open a door to reveal a goat and never the car.
# The host must always offer the chance to switch between the door chosen originally and the closed door remaining.

# Is it to your advantage to switch your choice?


def Monty_Hall_Problem_always_stay(K):
    actual_position_of_car = np.array(random.choices([0, 1, 2], k=K))
    player_choice = np.array(random.choices([0, 1, 2], k=K))

    print(np.mean(actual_position_of_car - player_choice == 0))


def Monty_Hall_Problem_always_switch(K):
    actual_position_of_car = np.array(random.choices([0, 1, 2], k=K))
    player_choice = np.array(random.choices([0, 1, 2], k=K))

    actual_situation = np.vstack((actual_position_of_car, player_choice))
    switch_result = np.apply_along_axis(switch_door, 0, actual_situation)

    print(np.mean(switch_result == 1))


def switch_door(situation):
    player_choice, car_position = situation
    if player_choice == car_position:  # player chose the car door, however he must change them :c
        return 0
    else:  # player chose the goat door, however he must change them for car one,
        # because the host is obligated to reveal goat door
        return 1


Monty_Hall_Problem_always_stay(10000)
Monty_Hall_Problem_always_switch(10000)
