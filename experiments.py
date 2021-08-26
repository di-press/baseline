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



if __name__ == '__main__':

    #exp_baseline_1()

    exp_baseline_2()