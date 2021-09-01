from matplotlib.pyplot import plot
import graphics
from pathlib import Path
import pandas as pd

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



# ItemKNN minkowski
def exp_baseline_8():

    file = Path.cwd().joinpath('ItemKNN_minkowski_1_to_50_neighbors.txt')
    MRSE_values_list = graphics.MRSE_values_from_file(file)
    #print(MRSE_values_list)

    graphics.MRSE_graphic('8', 50, MRSE_values_list, "ItemKNN", 10, "minkowski","steelblue")


# ItemKNN chebyshev
def exp_baseline_9():

    file = Path.cwd().joinpath('ItemKNN_chebyshev_1_to_50_neighbors.txt')
    MRSE_values_list = graphics.MRSE_values_from_file(file)
    #print(MRSE_values_list)

    graphics.MRSE_graphic('9', 50, MRSE_values_list, "ItemKNN", 10, "chebyshev","palegreen")

# ItemKNN manhattan (cityblock)
def exp_baseline_10():

    file = Path.cwd().joinpath('ItemKNN_manhattan_1_to_50_neighbors.txt')
    MRSE_values_list = graphics.MRSE_values_from_file(file)
    #print(MRSE_values_list)

    graphics.MRSE_graphic('10', 50, MRSE_values_list, "ItemKNN", 10, "manhattan","crimson")


#UserKNN jaccard:
def exp_baseline_11():

    file = Path.cwd().joinpath('UserKNN_jaccard_1_to_50_neighbors.txt')
    MRSE_values_list = graphics.MRSE_values_from_file(file)
    #print(MRSE_values_list)

    graphics.MRSE_graphic('11', 50, MRSE_values_list, "UserKNN", 10, "jaccard","tomato")

#ItemKNN jaccard:
def exp_baseline_12():

    file = Path.cwd().joinpath('ItemKNN_jaccard_1_to_50_neighbors.txt')
    MRSE_values_list = graphics.MRSE_values_from_file(file)
    #print(MRSE_values_list)

    graphics.MRSE_graphic('12', 50, MRSE_values_list, "ItemKNN", 10, "jaccard","dodgerblue")


def all_graphs_UserKNN():

    plot_info = []

    baseline_cosine_userknn_file = Path.cwd().joinpath('UserKNN_cosine_1_to_50_neighbors.txt')
    MRSE_values_list_cosine = graphics.MRSE_values_from_file(baseline_cosine_userknn_file)
    plot_0 = {'RMSE_values': MRSE_values_list_cosine, 'legend': 'UserKNN - cosseno', 'color': 'teal'}

    plot_info.append(plot_0)

    baseline_hamming_userknn_file = Path.cwd().joinpath('UserKNN_hamming_1_to_50_neighbors.txt')
    MRSE_values_list_hamming = graphics.MRSE_values_from_file(baseline_hamming_userknn_file)
    plot_1 = {'RMSE_values': MRSE_values_list_hamming, 'legend': 'UserKNN - hamming', 'color': 'orange'}

    plot_info.append(plot_1)

    file = Path.cwd().joinpath('UserKNN_minkowski_1_to_50_neighbors.txt')
    MRSE_values_list_minkowski = graphics.MRSE_values_from_file(file)
    plot_2 = {'RMSE_values': MRSE_values_list_minkowski, 'legend': 'UserKNN - minkowski', 'color': 'deeppink'}
    
    plot_info.append(plot_2)

    file = Path.cwd().joinpath('UserKNN_chebyshev_1_to_50_neighbors.txt')
    MRSE_values_list_chebyshev = graphics.MRSE_values_from_file(file)
    plot_3 = {'RMSE_values': MRSE_values_list_chebyshev, 'legend': 'UserKNN - chebyshev', 'color': 'chartreuse'}
    
    plot_info.append(plot_3)


    file = Path.cwd().joinpath('UserKNN_manhattan_1_to_50_neighbors.txt')
    MRSE_values_list_manhattan = graphics.MRSE_values_from_file(file)
    plot_4 = {'RMSE_values': MRSE_values_list_manhattan, 'legend': 'UserKNN - manhattan', 'color': 'indigo'}
    
    plot_info.append(plot_4)

    graphics.MRSE_graphic_all_exps(50, plot_info, 10, 'UserKNN')


def all_graphs_ItemKNN():

    plot_info = []

    baseline_cosine_itemknn_file = Path.cwd().joinpath('ItemKNN_cosine_1_to_50_neighbors.txt')
    MRSE_values_list_cosine = graphics.MRSE_values_from_file(baseline_cosine_itemknn_file)
    plot_0 = {'RMSE_values': MRSE_values_list_cosine, 'legend': 'ItemKNN - cosseno', 'color': 'darkmagenta'}

    plot_info.append(plot_0)

    baseline_hamming_userknn_file = Path.cwd().joinpath('ItemKNN_hamming_1_to_50_neighbors.txt')
    MRSE_values_list_hamming = graphics.MRSE_values_from_file(baseline_hamming_userknn_file)
    plot_1 = {'RMSE_values': MRSE_values_list_hamming, 'legend': 'ItemKNN - hamming', 'color': 'deeppink'}

    plot_info.append(plot_1)

    file = Path.cwd().joinpath('ItemKNN_minkowski_1_to_50_neighbors.txt')
    MRSE_values_list_minkowski = graphics.MRSE_values_from_file(file)
    plot_2 = {'RMSE_values': MRSE_values_list_minkowski, 'legend': 'ItemKNN - minkowski', 'color': 'steelblue'}
    
    plot_info.append(plot_2)

    file = Path.cwd().joinpath('ItemKNN_chebyshev_1_to_50_neighbors.txt')
    MRSE_values_list_chebyshev = graphics.MRSE_values_from_file(file)
    plot_3 = {'RMSE_values': MRSE_values_list_chebyshev, 'legend': 'ItemKNN - chebyshev', 'color': 'springgreen'}
    
    plot_info.append(plot_3)


    file = Path.cwd().joinpath('ItemKNN_manhattan_1_to_50_neighbors.txt')
    MRSE_values_list_manhattan = graphics.MRSE_values_from_file(file)
    plot_4 = {'RMSE_values': MRSE_values_list_manhattan, 'legend': 'ItemKNN - manhattan', 'color': 'crimson'}
    
    plot_info.append(plot_4)

    graphics.MRSE_graphic_all_exps(50, plot_info, 10, 'ItemKNN')

    
def all_baselines_versus_proposal1():

    plot_info=[]

    baseline_hamming_userknn_file = Path.cwd().joinpath('UserKNN_hamming_1_to_50_neighbors.txt')
    MRSE_values_list_hamming = graphics.MRSE_values_from_file(baseline_hamming_userknn_file)
    plot_1 = {'RMSE_values': MRSE_values_list_hamming[:-1], 'legend': 'UserKNN - hamming', 'color': 'lightseagreen'}

    plot_info.append(plot_1)

    baseline_hamming_itemknn_file = Path.cwd().joinpath('ItemKNN_hamming_1_to_50_neighbors.txt')
    MRSE_values_list_hamming = graphics.MRSE_values_from_file(baseline_hamming_itemknn_file)
    plot_2 = {'RMSE_values': MRSE_values_list_hamming[:-1], 'legend': 'ItemKNN - hamming', 'color': 'royalblue'}

    plot_info.append(plot_2)

    proposal_file = Path.cwd().joinpath('no_genre_standard_n=50_cv=10_hamming_exp2.csv')


    df = pd.read_csv(proposal_file,
                        index_col= False,
                        sep=",", warn_bad_lines=True, 
                        error_bad_lines=True,
                        engine='python',
                        header=0,
                        usecols = ['MSRE']

    )

    MRSE_values_list_hamming = list(df['MSRE'])
    plot_3 = {'RMSE_values': MRSE_values_list_hamming, 'legend': 'Proposta 1 - hamming', 'color': 'orange'}

    plot_info.append(plot_3)

    graphics.baselines_versus_proposal(1, 49, plot_info)

# perso user + perso movie + genres
def all_baselines_versus_proposal2():

    plot_info=[]

    baseline_hamming_userknn_file = Path.cwd().joinpath('UserKNN_hamming_1_to_50_neighbors.txt')
    MRSE_values_list_hamming = graphics.MRSE_values_from_file(baseline_hamming_userknn_file)
    plot_1 = {'RMSE_values': MRSE_values_list_hamming[:-1], 'legend': 'UserKNN - hamming', 'color': 'lightseagreen'}

    plot_info.append(plot_1)

    baseline_hamming_itemknn_file = Path.cwd().joinpath('ItemKNN_hamming_1_to_50_neighbors.txt')
    MRSE_values_list_hamming = graphics.MRSE_values_from_file(baseline_hamming_itemknn_file)
    plot_2 = {'RMSE_values': MRSE_values_list_hamming[:-1], 'legend': 'ItemKNN - hamming', 'color': 'royalblue'}

    plot_info.append(plot_2)

    proposal_file = Path.cwd().joinpath('with_genre_normalization_n=50_cv=10_hammingexp23.csv')


    df = pd.read_csv(proposal_file,
                        index_col= False,
                        sep=",", warn_bad_lines=True, 
                        error_bad_lines=True,
                        engine='python',
                        header=0,
                        usecols = ['MSRE']

    )

    MRSE_values_list_hamming = list(df['MSRE'])
    plot_3 = {'RMSE_values': MRSE_values_list_hamming, 'legend': 'Proposta 2 - hamming', 'color': 'limegreen'}

    plot_info.append(plot_3)

    graphics.baselines_versus_proposal(2, 49, plot_info)


# perso user 
def all_baselines_versus_proposal3():

    plot_info=[]

    baseline_hamming_userknn_file = Path.cwd().joinpath('UserKNN_hamming_1_to_50_neighbors.txt')
    MRSE_values_list_hamming = graphics.MRSE_values_from_file(baseline_hamming_userknn_file)
    plot_1 = {'RMSE_values': MRSE_values_list_hamming[:-1], 'legend': 'UserKNN - hamming', 'color': 'lightseagreen'}

    plot_info.append(plot_1)

    baseline_hamming_itemknn_file = Path.cwd().joinpath('ItemKNN_hamming_1_to_50_neighbors.txt')
    MRSE_values_list_hamming = graphics.MRSE_values_from_file(baseline_hamming_itemknn_file)
    plot_2 = {'RMSE_values': MRSE_values_list_hamming[:-1], 'legend': 'ItemKNN - hamming', 'color': 'royalblue'}

    plot_info.append(plot_2)

    proposal_file = Path.cwd().joinpath('no_genre_normalization_n=50_cv=10_hammingexp25.csv')


    df = pd.read_csv(proposal_file,
                        index_col= False,
                        sep=",", warn_bad_lines=True, 
                        error_bad_lines=True,
                        engine='python',
                        header=0,
                        usecols = ['MSRE']

    )

    MRSE_values_list_hamming = list(df['MSRE'])
    plot_3 = {'RMSE_values': MRSE_values_list_hamming, 'legend': 'Proposta 3 - hamming', 'color': 'lightcoral'}

    plot_info.append(plot_3)

    graphics.baselines_versus_proposal(3, 49, plot_info)


# perso movie 
def all_baselines_versus_proposal4():

    plot_info=[]

    baseline_hamming_userknn_file = Path.cwd().joinpath('UserKNN_hamming_1_to_50_neighbors.txt')
    MRSE_values_list_hamming = graphics.MRSE_values_from_file(baseline_hamming_userknn_file)
    plot_1 = {'RMSE_values': MRSE_values_list_hamming[:-1], 'legend': 'UserKNN - hamming', 'color': 'lightseagreen'}

    plot_info.append(plot_1)

    baseline_hamming_itemknn_file = Path.cwd().joinpath('ItemKNN_hamming_1_to_50_neighbors.txt')
    MRSE_values_list_hamming = graphics.MRSE_values_from_file(baseline_hamming_itemknn_file)
    plot_2 = {'RMSE_values': MRSE_values_list_hamming[:-1], 'legend': 'ItemKNN - hamming', 'color': 'royalblue'}

    plot_info.append(plot_2)

    proposal_file = Path.cwd().joinpath('no_genre_normalization_n=50_cv=10_hammingexp27.csv')


    df = pd.read_csv(proposal_file,
                        index_col= False,
                        sep=",", warn_bad_lines=True, 
                        error_bad_lines=True,
                        engine='python',
                        header=0,
                        usecols = ['MSRE']

    )

    MRSE_values_list_hamming = list(df['MSRE'])
    plot_3 = {'RMSE_values': MRSE_values_list_hamming, 'legend': 'Proposta 4 - hamming', 'color': 'indigo'}

    plot_info.append(plot_3)

    graphics.baselines_versus_proposal(4, 49, plot_info)


def all_baselines_versus_all_proposals():

    plot_info=[]

    baseline_hamming_userknn_file = Path.cwd().joinpath('UserKNN_hamming_1_to_50_neighbors.txt')
    MRSE_values_list_hamming = graphics.MRSE_values_from_file(baseline_hamming_userknn_file)
    plot_1 = {'RMSE_values': MRSE_values_list_hamming[:-1], 'legend': 'UserKNN - hamming', 'color': 'lightseagreen'}

    plot_info.append(plot_1)

    baseline_hamming_itemknn_file = Path.cwd().joinpath('ItemKNN_hamming_1_to_50_neighbors.txt')
    MRSE_values_list_hamming = graphics.MRSE_values_from_file(baseline_hamming_itemknn_file)
    plot_2 = {'RMSE_values': MRSE_values_list_hamming[:-1], 'legend': 'ItemKNN - hamming', 'color': 'royalblue'}

    plot_info.append(plot_2)

    proposal_file = Path.cwd().joinpath('no_genre_normalization_n=50_cv=10_hammingexp27.csv')


    df = pd.read_csv(proposal_file,
                        index_col= False,
                        sep=",", warn_bad_lines=True, 
                        error_bad_lines=True,
                        engine='python',
                        header=0,
                        usecols = ['MSRE']

    )

    MRSE_values_list_hamming = list(df['MSRE'])
    plot_3 = {'RMSE_values': MRSE_values_list_hamming, 'legend': 'Proposta 4 - hamming', 'color': 'indigo'}

    plot_info.append(plot_3)


    proposal_file = Path.cwd().joinpath('no_genre_standard_n=50_cv=10_hamming_exp2.csv')


    df = pd.read_csv(proposal_file,
                        index_col= False,
                        sep=",", warn_bad_lines=True, 
                        error_bad_lines=True,
                        engine='python',
                        header=0,
                        usecols = ['MSRE']

    )

    MRSE_values_list_hamming = list(df['MSRE'])
    plot_4 = {'RMSE_values': MRSE_values_list_hamming, 'legend': 'Proposta 1 - hamming', 'color': 'orange'}

    plot_info.append(plot_4)

    proposal_file = Path.cwd().joinpath('with_genre_normalization_n=50_cv=10_hammingexp23.csv')


    df = pd.read_csv(proposal_file,
                        index_col= False,
                        sep=",", warn_bad_lines=True, 
                        error_bad_lines=True,
                        engine='python',
                        header=0,
                        usecols = ['MSRE']

    )

    MRSE_values_list_hamming = list(df['MSRE'])
    plot_5 = {'RMSE_values': MRSE_values_list_hamming, 'legend': 'Proposta 2 - hamming', 'color': 'limegreen'}

    plot_info.append(plot_5)

    proposal_file = Path.cwd().joinpath('no_genre_normalization_n=50_cv=10_hammingexp25.csv')


    df = pd.read_csv(proposal_file,
                        index_col= False,
                        sep=",", warn_bad_lines=True, 
                        error_bad_lines=True,
                        engine='python',
                        header=0,
                        usecols = ['MSRE']

    )

    MRSE_values_list_hamming = list(df['MSRE'])
    plot_6 = {'RMSE_values': MRSE_values_list_hamming, 'legend': 'Proposta 3 - hamming', 'color': 'lightcoral'}

    plot_info.append(plot_6)


    graphics.baselines_versus_proposal(5, 49, plot_info)

if __name__ == '__main__':

    #exp_baseline_1()

    #exp_baseline_2()

    #exp_baseline_3()

    #exp_baseline_4()

    #exp_baseline_5()

    #exp_baseline_6()

    #exp_baseline_7()

    #exp_baseline_8()

    #exp_baseline_9()

    #exp_baseline_10()

    #exp_baseline_11()

    #exp_baseline_12()

    #all_graphs_UserKNN()

    #all_graphs_ItemKNN()

    #all_baselines_versus_proposal1()

    #all_baselines_versus_proposal2()

    #all_baselines_versus_proposal3()

    #all_baselines_versus_proposal4()

    all_baselines_versus_all_proposals()