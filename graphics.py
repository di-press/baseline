import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


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

#-----------------------------------------------------------------------------------
# "blueviolet" "mediumblue"
#AINDA N CODEI A FUNÇÂO ABAIXO!S
def r2_score_graphic(id_exp, num_neighbors_max, r2_values, nome_legenda, k_folds, dist, cor):


    plt.style.use('seaborn')

    titulo = '''Coeficiente de determinação (R²) obtido para cada valor de vizinho K \n 
                com validação em ''' + str(k_folds) + ' folds - métrica ' + str(dist)

    plt.title(titulo)

    plt.xlabel("K (vizinhos)")
    plt.ylabel("coeficiente de determinação (R²)")

    k_values = list(range(1, num_neighbors_max+1))
    x = k_values
    y = r2_values


    plt.plot(x, y, label = nome_legenda, color = cor, marker = ".", linestyle = "solid", markersize = "10")    

    plt.legend()
    filename = 'R2_cv='+ str(k_folds) + '_' + str(dist) +"_" + id_exp +'.png'
    plt.savefig(filename)

    plt.close()

    #plt.show()


#------------------------------------------------------------------------------------

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

    baseline_cosine_userknn_file = Path.cwd().joinpath('UserKNN_cosine_1_to_50_neighbors.txt')
    MRSE_values_list = MRSE_values_from_file(baseline_cosine_userknn_file)
    print(MRSE_values_list)