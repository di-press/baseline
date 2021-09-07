
import pandas as pd
from pathlib import Path
from caserec.utils.cross_validation import CrossValidation
from caserec.recommenders.rating_prediction.item_attribute_knn import ItemAttributeKNN
import graphics
import files
def generate_metadata_file_df():
    
    csv_file = Path.cwd().joinpath('final_shuffled_dataset.csv')


    df = pd.read_csv(csv_file,
                        index_col= False,
                        sep=",", warn_bad_lines=True, 
                        error_bad_lines=True,
                        engine='python',
                        header=0,
                        usecols = ['user_id',
                        'movielens_Id',
                        'movie_rating',
                        'movie_extroversion',
                        'movie_neuroticism',
                        'movie_agreeableness',
                        'movie_conscientiousness',
                        'movie_openess']                                                                        
                    )
    
    
    df.reset_index(drop=True, inplace=False)

    user_id_list = list(df['user_id'])

    new_users_ids = user_hashes_to_id(user_id_list)

    #df.drop('user_id', axis='columns', inplace=True)

    items_tuples = []
    baseline_df_tuples = []

    for index, row in df.iterrows():

        current_hash_id = row['user_id']
        new_numerical_id = new_users_ids[current_hash_id]
        item_id = row['movielens_Id']
        movie_rating = row['movie_rating']
        
        current_movie_extroversion = row['movie_extroversion']
        current_movie_neuroticism = row['movie_neuroticism']
        current_movie_agreeableness = row['movie_agreeableness']
        current_movie_conscientiousness = row['movie_conscientiousness']
        current_movie_openess = row['movie_openess']
        
        # row values of movie metadata:
        row_values = (item_id, current_movie_extroversion,
                     current_movie_neuroticism, current_movie_agreeableness,
                     current_movie_conscientiousness, current_movie_openess)

        user_item_rating_row_values = (new_numerical_id, 
                                        item_id,
                                        movie_rating
                                        )

        items_tuples.append(row_values)
        baseline_df_tuples.append(user_item_rating_row_values)
        #print(row_values)

    final_df = pd.DataFrame(items_tuples,
                        columns = ['movielens_Id',
                                    'movie_extroversion',
                                    'movie_neuroticism',
                                    'movie_agreeableness',
                                    'movie_conscientiousness',
                                    'movie_openess'
                                   ]                                       
                                )

    # data containing the users and its personality:
    path_temp_csv = Path.cwd().joinpath('movie_perso_metadata.csv')
    # index = True star indexing the csv file with 0:
    final_df.to_csv(path_temp_csv, sep='\t', index = False, header=False)


    user_interaction_df = pd.DataFrame(baseline_df_tuples,
                        columns = ['new_user_id',
                                   'movielens_Id',
                                   'movie_rating']                                       
                                )

    # data containing the interaction for usrAttKNN:
    path_temp_csv = Path.cwd().joinpath('baseline_ItemAtrrKNN_df.csv')
    # index = True star indexing the csv file with 0:
    user_interaction_df.to_csv(path_temp_csv, sep='\t', index = False, header=False)

def user_hashes_to_id(user_id_list):

    id_set = set(user_id_list)

    new_user_id = list(range( 0, len(id_set)))

    meu_zip = list(zip(new_user_id, id_set))

    hash_id_dict = {}

    for item in meu_zip:

        hash_id_dict[item[1]] = item[0]

    #print(hash_id_dict)
    return hash_id_dict

#use this function with > filename.txt
def item_attr_knn_rating_prediction(metric):

    #complete database file, without splitting
    input_file = Path.cwd().joinpath('baseline_ItemAtrrKNN_df.csv') 
    input_file = str(input_file)

    prediction_dir = Path.cwd().joinpath('ItemKnnAtrr')
    prediction_dir  = str(prediction_dir)

    only_movie_perso = Path.cwd().joinpath('movie_perso_metadata.csv') 
    print("metric: ", metric)

    for num_neighbors in range(1, 51):

        print("k neighbors= ", num_neighbors)
        recommender = ItemAttributeKNN(similarity_metric=metric, metadata_file=only_movie_perso,
                                        k_neighbors= num_neighbors)

        CrossValidation(input_file, recommender, prediction_dir, k_folds = 10, header=None,
                        write_predictions = True).compute()


def ItemKNN_ItemAttrKNN_graphic():

    plot_info = []

    baseline_ItemKNN = files.item_KNN_file
    MRSE_values_list_itemKNN = graphics.MRSE_values_from_file(baseline_ItemKNN)
    plot_0 = {'RMSE_values': MRSE_values_list_itemKNN, 'legend': 'itemKNN - baseline', 'color': 'rebeccapurple'}

    plot_info.append(plot_0)

    proposal_ItemAttrKNN = files.ItemATTRKNN_file_chebyshev
    MRSE_values_list_ItemAttrKNN = graphics.MRSE_values_from_file(proposal_ItemAttrKNN)
    plot_1 = {'RMSE_values': MRSE_values_list_ItemAttrKNN, 'legend': 'ItemAttrKNN - chebyshev', 'color': 'blue'}

    plot_info.append(plot_1)

    proposal_ItemAttrKNN = files.ItemATTRKNN_file_manhattan
    MRSE_values_list_ItemAttrKNN = graphics.MRSE_values_from_file(proposal_ItemAttrKNN)
    plot_2 = {'RMSE_values': MRSE_values_list_ItemAttrKNN, 'legend': 'ItemAttrKNN - manhattan', 'color': 'tomato'}

    plot_info.append(plot_2)


    proposal_ItemAttrKNN = files.ItemATTRKNN_file_minkowski
    MRSE_values_list_ItemAttrKNN = graphics.MRSE_values_from_file(proposal_ItemAttrKNN)
    plot_3 = {'RMSE_values': MRSE_values_list_ItemAttrKNN, 'legend': 'ItemAttrKNN - minkowski', 'color': 'gold'}

    plot_info.append(plot_3)

    graphics.new_proposals_graphic(7, 50, plot_info, "Comparação do baseline ItemKNN com as propostas de ItemAttrKNN")


def item_atrr_knn_experiments(similarity, num_neighbors):

    #complete database file, without splitting
    input_file = Path.cwd().joinpath('baseline_ItemAtrrKNN_df.csv.csv') 
    input_file = str(input_file)

    prediction_dir = Path.cwd().joinpath('ItemAttrKNN' + similarity)
    prediction_dir  = str(prediction_dir)


    only_movie_perso = Path.cwd().joinpath('movie_perso_metadata.csv') 
    print("metric: ", similarity)

    recommender = ItemAttributeKNN(similarity_metric= similarity, metadata_file= only_movie_perso, 
                                    k_neighbors= num_neighbors)


    CrossValidation(input_file, recommender, prediction_dir, k_folds = 10, 
                    header=1, recommender_verbose=False, as_table=False,
                    evaluation_in_fold_verbose=False,del_folds=True,
                    write_predictions = False).compute()


if __name__ == '__main__':

    #generate_metadata_file_df()

    #item_attr_knn_rating_prediction('minkowski')

    #item_attr_knn_rating_prediction('cityblock')

    #item_attr_knn_rating_prediction('chebyshev')

    ItemKNN_ItemAttrKNN_graphic()