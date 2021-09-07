# files with MRSE values

from pathlib import Path

user_KNN_file = Path.cwd().joinpath('UserKNN_hamming_1_to_50_neighbors.txt')

item_KNN_file = Path.cwd().joinpath('ItemKNN_hamming_1_to_50_neighbors.txt')

matrix_fact_file = Path.cwd().joinpath("matrix_fact_various_factors.txt")


proposal_1_file = Path.cwd().joinpath('no_genre_standard_n=50_cv=10_hamming_exp2.csv')

proposal_2_file = Path.cwd().joinpath('with_genre_normalization_n=50_cv=10_hammingexp23.csv')

proposal_3_file = Path.cwd().joinpath('no_genre_normalization_n=50_cv=10_hammingexp25.csv')

proposal_4_file = Path.cwd().joinpath('no_genre_normalization_n=50_cv=10_hammingexp27.csv')


UserATTRKNN_file_chebyshev = Path.cwd().joinpath('UserAttrKnn_chebyshev_1_to_50_neighbors.txt')
UserATTRKNN_file_manhattan = Path.cwd().joinpath('UserAttrKnn_manhattan_1_to_50_neighbors.txt')
UserATTRKNN_file_minkowski = Path.cwd().joinpath('UserAttrKnn_minkowski_1_to_50_neighbors.txt')


ItemATTRKNN_file_chebyshev = Path.cwd().joinpath('ItemAttrKNN_chebyshev_1_to_50_neighbors.txt')
ItemATTRKNN_file_manhattan = Path.cwd().joinpath('ItemAttrKNN_manhattan_1_to_50_neighbors.txt')
ItemATTRKNN_file_minkowski = Path.cwd().joinpath('ItemAttrKNN_minkowski_1_to_50_neighbors.txt')


# MSRE values for baselines, from its file:
def MRSE_values_from_file_baselines(filename):

    MSRE_values = []

    with open(filename) as f:
        
        for line in f:

            current_line = line.split()

            if current_line:
                
                if current_line[0] == 'Mean::':

                    current_MRSE = float(current_line[4])
                    MSRE_values.append(current_MRSE)
        
    #print("number of MSRE values: ", len(MSRE_values))

    return MSRE_values
