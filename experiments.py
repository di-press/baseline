import graphics
from pathlib import Path

#userKNN with cosine:
def exp_baseline_1():

    baseline_cosine_userknn_file = Path.cwd().joinpath('UserKNN_cosine_1_to_50_neighbors.txt')
    MRSE_values_list = graphics.MRSE_values_from_file(baseline_cosine_userknn_file)
    #print(MRSE_values_list)

    graphics.MRSE_graphic('1', 50, MRSE_values_list, "UserKNN", 10, "cosine","teal")


#UserKNN hamming
def exp_baseline_2():

    baseline_hamming_userknn_file = Path.cwd().joinpath('UserKNN_hamming_1_to_50_neighbors.txt')
    MRSE_values_list = graphics.MRSE_values_from_file(baseline_hamming_userknn_file)
    #print(MRSE_values_list)

    graphics.MRSE_graphic('2', 50, MRSE_values_list, "UserKNN", 10, "hamming","darkgreen")


def exp_baseline_3():

    baseline_cosine_itemknn_file = Path.cwd().joinpath('ItemKNN_cosine_1_to_50_neighbors.txt')
    MRSE_values_list = graphics.MRSE_values_from_file(baseline_cosine_itemknn_file)
    #print(MRSE_values_list)

    graphics.MRSE_graphic('3', 50, MRSE_values_list, "ItemKNN", 10, "cosine","darkmagenta")


def exp_baseline_4():

    baseline_hamming_itemknn_file = Path.cwd().joinpath('ItemKNN_hamming_1_to_50_neighbors.txt')
    MRSE_values_list = graphics.MRSE_values_from_file(baseline_hamming_itemknn_file)
    #print(MRSE_values_list)

    graphics.MRSE_graphic('4', 50, MRSE_values_list, "ItemKNN", 10, "hamming","deeppink")

# UserKNN minkowski
def exp_baseline_5():

    file = Path.cwd().joinpath('UserKNN_minkowski_1_to_50_neighbors.txt')
    MRSE_values_list = graphics.MRSE_values_from_file(file)
    #print(MRSE_values_list)

    graphics.MRSE_graphic('5', 50, MRSE_values_list, "UserKNN", 10, "minkowski","deeppink")

# UserKNN chebyshev
def exp_baseline_6():

    file = Path.cwd().joinpath('UserKNN_chebyshev_1_to_50_neighbors.txt')
    MRSE_values_list = graphics.MRSE_values_from_file(file)
    #print(MRSE_values_list)

    graphics.MRSE_graphic('6', 50, MRSE_values_list, "UserKNN", 10, "chebyshev","chartreuse")

# UserKNN manhattan
def exp_baseline_7():

    file = Path.cwd().joinpath('UserKNN_manhattan_1_to_50_neighbors.txt')
    MRSE_values_list = graphics.MRSE_values_from_file(file)
    #print(MRSE_values_list)

    graphics.MRSE_graphic('7', 50, MRSE_values_list, "UserKNN", 10, "manhattan","indigo")

# UserKNN mahalanobis
def exp_baseline_8():

    file = Path.cwd().joinpath('UserKNN_mahalanobis_1_to_50_neighbors.txt')
    MRSE_values_list = graphics.MRSE_values_from_file(file)
    #print(MRSE_values_list)

    graphics.MRSE_graphic('8', 50, MRSE_values_list, "UserKNN", 10, "mahalanobis","mediumvioletred")

# ItemKNN minkowski
def exp_baseline_9():

    file = Path.cwd().joinpath('ItemKNN_minkowski_1_to_50_neighbors.txt')
    MRSE_values_list = graphics.MRSE_values_from_file(file)
    #print(MRSE_values_list)

    graphics.MRSE_graphic('9', 50, MRSE_values_list, "ItemKNN", 10, "minkowski","steelblue")


# ItemKNN chebyshev
def exp_baseline_10():

    file = Path.cwd().joinpath('ItemKNN_chebyshev_1_to_50_neighbors.txt')
    MRSE_values_list = graphics.MRSE_values_from_file(file)
    #print(MRSE_values_list)

    graphics.MRSE_graphic('10', 50, MRSE_values_list, "ItemKNN", 10, "chebyshev","palegreen")

# ItemKNN manhattan
def exp_baseline_10():

    file = Path.cwd().joinpath('ItemKNN_manhattan_1_to_50_neighbors.txt')
    MRSE_values_list = graphics.MRSE_values_from_file(file)
    #print(MRSE_values_list)

    graphics.MRSE_graphic('10', 50, MRSE_values_list, "ItemKNN", 10, "manhattan","bisque")

# ItemKNN mahalanobis
def exp_baseline_11():

    file = Path.cwd().joinpath('ItemKNN_mahalanobis_1_to_50_neighbors.txt')
    MRSE_values_list = graphics.MRSE_values_from_file(file)
    #print(MRSE_values_list)

    graphics.MRSE_graphic('11', 50, MRSE_values_list, "ItemKNN", 10, "mahalanobis","plum")


if __name__ == '__main__':

    #exp_baseline_1()

    #exp_baseline_2()

    #exp_baseline_3()

    #exp_baseline_4()