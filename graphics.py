import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd


# MSRE values for baseline?
def MRSE_values_from_file(filename):

    MSRE_values = []

    with open(filename) as f:
        
        for line in f:

            current_line = line.split()

            if current_line:
                
                if current_line[0] == 'Mean::':

                    current_MRSE = float(current_line[4])
                    MSRE_values.append(current_MRSE)
        
    print("number of MSRE values: ", len(MSRE_values))

    return MSRE_values

#function to read the MRSE results of regressive knn:
#(not baseline)
def read_csv_regr_knn(filename):

    csv_file = Path.cwd().joinpath(filename)


    df = pd.read_csv(csv_file,
                        index_col= False,
                        sep=",", warn_bad_lines=True, 
                        error_bad_lines=True,
                        engine='python',
                        header=0,
                        usecols = ['k_values',
                                   'MSRE',
                                   'r2_score',
                                   ]                                                                        
                    )
    
    
    df.reset_index(drop=True, inplace=False)

    print(len(list(df['MSRE'])))
    return list(df['MSRE'])

# teal e coral:
def graphic_userknn_comparison(distance, color_graphic_baseline, color_graphic_proposal):

    if distance == 'cosine':


        baseline_cosine_userknn_file = Path.cwd().joinpath('UserKNN_cosine_1_to_50_neighbors.txt')
        MRSE_values_list_baseline = MRSE_values_from_file(baseline_cosine_userknn_file)

        proposal_cosine_userknn_file= Path.cwd().joinpath('no_genre_normalization_n=50_cv=10_cosineexp26.csv')
        MRSE_values_list_proposal = read_csv_regr_knn(proposal_cosine_userknn_file)
    
    if distance == 'hamming':

        baseline_hamming_userknn_file = Path.cwd().joinpath('UserKNN_hamming_1_to_50_neighbors.txt')
        MRSE_values_list_baseline = MRSE_values_from_file(baseline_hamming_userknn_file)

        proposal_hamming_userknn_file= Path.cwd().joinpath('no_genre_normalization_n=50_cv=10_hammingexp25.csv')
        MRSE_values_list_proposal = read_csv_regr_knn(proposal_hamming_userknn_file)

    baseline_proposal_graphic(MRSE_values_list_baseline, MRSE_values_list_proposal, 50, 
                                "UserKNN (baseline)", "regressive KNN - only user personality",
                                10, distance, color_graphic_baseline, color_graphic_proposal)



def graphic_itemknn_comparison(distance, color_graphic_baseline, color_graphic_proposal):

    if distance == 'cosine':


        baseline_cosine_userknn_file = Path.cwd().joinpath('ItemKNN_cosine_1_to_50_neighbors.txt')
        MRSE_values_list_baseline = MRSE_values_from_file(baseline_cosine_userknn_file)

        proposal_cosine_userknn_file= Path.cwd().joinpath('no_genre_normalization_n=50_cv=10_cosineexp28.csv')
        MRSE_values_list_proposal = read_csv_regr_knn(proposal_cosine_userknn_file)
    
    if distance == 'hamming':

        baseline_hamming_userknn_file = Path.cwd().joinpath('ItemKNN_hamming_1_to_50_neighbors.txt')
        MRSE_values_list_baseline = MRSE_values_from_file(baseline_hamming_userknn_file)

        proposal_hamming_userknn_file= Path.cwd().joinpath('no_genre_normalization_n=50_cv=10_hammingexp27.csv')
        MRSE_values_list_proposal = read_csv_regr_knn(proposal_hamming_userknn_file)

    baseline_proposal_graphic(MRSE_values_list_baseline, MRSE_values_list_proposal, 50, 
                                "ItemKNN (baseline)", "regressive KNN - only item personality",
                                10, distance, color_graphic_baseline, color_graphic_proposal)
#-------------------------------------------------------------------------
#"coral" "aqua"
def MRSE_graphic(id_exp, num_neighbors_max, MSRE_values, nome_legenda, k_folds, dist, cor):

    plt.style.use('seaborn')

    titulo = '''MSRE obtido para cada valor de vizinho K  do baseline \n 
                com validação em ''' + str(k_folds) + ' folds - métrica ' + str(dist) 

    plt.title(titulo)

    plt.xlabel("K (vizinhos)")
    plt.ylabel("MSRE")

    k_values = list(range(1, num_neighbors_max+1))
    x = k_values
    y = MSRE_values


    plt.plot(x, y, label = nome_legenda, color = cor, marker = ".", linestyle = "solid", markersize = "10")    

    plt.legend()

    filename = 'MRSE_cv='+ str(k_folds) + '_' + str(dist) + '_' + id_exp + '.png'
    plt.savefig(filename)
    plt.close()


#------------------------------------------------------------------------------------

def baseline_proposal_graphic(MRSE_values_baseline, MRSE_values_proposal, num_neighbors_max, 
                                legend_name_baseline, legend_name_proposal, k_folds, dist,
                                color_graphic_baseline, color_graphic_proposal):


    plt.style.use('seaborn')

    titulo = '''comparação do baseline com a proposta - MSRE obtido para cada valor de vizinho K \n 
                com validação em ''' + str(k_folds) + ' folds - métrica ' + str(dist) 

    plt.title(titulo)

    plt.xlabel("K (vizinhos)")
    plt.ylabel("RMSE - Root Minimun Squared Error")

    k_values = list(range(1, num_neighbors_max))
    x = k_values
    # basleine is with 50 neighbors, but proposal is with 49!
    # extract the last value for baseline:
    y_baseline = MRSE_values_baseline[:-1]

    #baseline plot:
    plt.plot(x, y_baseline, label = legend_name_baseline, color = color_graphic_baseline, marker = ".", linestyle = "solid", markersize = "10")    

    k_values = list(range(1, num_neighbors_max))
    x = k_values
    y_proposal = MRSE_values_proposal

    #proposal plot:
    plt.plot(x, y_proposal, label = legend_name_proposal, color = color_graphic_proposal, marker = ".", linestyle = "solid", markersize = "10")    

    plt.legend()

    filename = 'MRSE_cv='+ str(k_folds) + '_' + str(dist) + '_' + str(legend_name_proposal) + "_vs_regr_knn" + '.png'
    plt.savefig(filename)
    plt.close()

#plots_info: (dict) for each plot, contains:
# RMSE values, legend name, dist, color of plot
def MRSE_graphic_all_exps(num_neighbors_max, plots_info , k_folds, algorithm_type):

    plt.style.use('seaborn')

    titulo = '''RMSE obtido para cada valor de vizinho K  dos diversos baselines do tipo ''' +algorithm_type +'''\n  com validação em ''' + str(k_folds) + ' folds, para diversas distâncias'

    plt.title(titulo)

    plt.xlabel("K (vizinhos)")
    plt.ylabel("RMSE")

    k_values = list(range(1, num_neighbors_max+1))
    x = k_values

    plot_0 = plots_info[0]
    plt.plot(x, plot_0['RMSE_values'], label = plot_0['legend'], color = plot_0['color'], marker = ".", linestyle = "solid", markersize = "7")    

    plot_1 = plots_info[1]
    plt.plot(x, plot_1['RMSE_values'], label = plot_1['legend'], color = plot_1['color'], marker = ".", linestyle = "solid", markersize = "7")    

    plot_2 = plots_info[2]
    plt.plot(x, plot_2['RMSE_values'], label = plot_2['legend'], color = plot_2['color'], marker = ".", linestyle = "solid", markersize = "7")   

    plot_3 = plots_info[3]
    plt.plot(x, plot_3['RMSE_values'], label = plot_3['legend'], color = plot_3['color'], marker = ".", linestyle = "solid", markersize = "7")     

    plot_4 = plots_info[4]
    plt.plot(x, plot_4['RMSE_values'], label = plot_4['legend'], color = plot_4['color'], marker = ".", linestyle = "solid", markersize = "7")    
   
    plt.legend()

    if algorithm_type == 'UserKNN':
        filename = 'UserKNN_comparison.png'

    if algorithm_type == 'ItemKNN':
        filename = 'ItemKNN_comparison.png'

    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":

    
    #MAP5_graph()
    #MAP10_graph()

    
    MSRE_values = [0.08,
                    0.06,
                    0.06,
                    0.07,
                    0.065,
                    0.08,
                    0.06,
                    0.06,
                    0.07,
                    0.065,
                    0.08,
                    0.06,
                    0.06,
                    0.07,
                    0.065,
                    0.08,
                    0.06,
                    0.06,
                    0.07,
                    0.065,
                    0.08,
                    0.06,
                    0.06,
                    0.07,
                    0.065,
                    0.08,
                    0.06,
                    0.06,
                    0.07,
                    0.065,
                    0.08,
                    0.06,
                    0.06,
                    0.07,
                    0.065,
                    0.08,
                    0.06,
                    0.06,
                    0.07,
                    0.065,
                    0.08,
                    0.06,
                    0.06,
                    0.07,
                    0.065,
                    0.08,
                    0.06,
                    0.06,
                    0.07,
                    0.065,
                   ]

    r2_values = MSRE_values
    #MRSE_graph(50, MSRE_values, "KNN",5, 'Hamiing', "aqua")
    #r2_score_graph(50, r2_values, "KNN", 5, 'Manhattan', "blueviolet" )
    #MRSE_graph(50, MSRE_values, "KNN",5, 'Euclidean', "mediumblue")
    #r2_score_graph(50, r2_values, "KNN", 5, 'Canberra', "coral" )

    #baseline_cosine_userknn_file = Path.cwd().joinpath('UserKNN_cosine_1_to_50_neighbors.txt')
    #MRSE_values_list = MRSE_values_from_file(baseline_cosine_userknn_file)
    #print(MRSE_values_list)

    #MRSE_graphic('1', 50, MRSE_values_list, "UserKNN", 10, "cosine","teal")

    #df_test = read_csv_regr_knn("no_genre_normalization_n=50_cv=10_cosineexp26.csv")

    #print(df_test)


    #graphic_userknn_comparison("cosine", "teal", "coral")
    #graphic_userknn_comparison("hamming", "lightseagreen", "lightcoral")

    graphic_itemknn_comparison("cosine", "mediumseagreen", "darkorange")
    graphic_itemknn_comparison("hamming", "mediumaquamarine", "orange")