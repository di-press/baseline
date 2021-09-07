
import pandas as pd
from pathlib import Path
from caserec.utils.cross_validation import CrossValidation
from caserec.recommenders.rating_prediction.user_attribute_knn import UserAttributeKNN
import files
import graphics


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
                        'user_extroversion',
                        'user_neuroticism',
                        'user_agreeableness',
                        'user_conscientiousness',
                        'user_openess']                                                                        
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
        
        current_user_extroversion = row['user_extroversion']
        current_user_neuroticism = row['user_neuroticism']
        current_user_agreeableness = row['user_agreeableness']
        current_user_conscientiousness = row['user_conscientiousness']
        current_user_openess = row['user_openess']
        
        row_values = (new_numerical_id, current_user_extroversion,
                     current_user_neuroticism, current_user_agreeableness,
                     current_user_conscientiousness, current_user_openess)

        user_item_rating_row_values = (new_numerical_id, 
                                        item_id,
                                        movie_rating
                                        )

        items_tuples.append(row_values)
        baseline_df_tuples.append(user_item_rating_row_values)
        #print(row_values)

    final_df = pd.DataFrame(items_tuples,
                        columns = ['new_user_id',
                                    'user_extroversion',
                                    'user_neuroticism',
                                    'user_agreeableness',
                                    'user_conscientiousness',
                                    'user_openess'
                                   ]                                       
                                )

    # data containing the users and its personality:
    path_temp_csv = Path.cwd().joinpath('user_perso_metadata.csv')
    # index = True star indexing the csv file with 0:
    final_df.to_csv(path_temp_csv, sep='\t', index = False, header=False)


    user_interaction_df = pd.DataFrame(baseline_df_tuples,
                        columns = ['new_user_id',
                                   'movielens_Id',
                                   'movie_rating']                                       
                                )

    # data containing the interaction for usrAttKNN:
    path_temp_csv = Path.cwd().joinpath('baseline_UserAtrrKNN_df.csv')
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
def user_attr_knn_rating_prediction(metric):

    #complete database file, without splitting
    input_file = Path.cwd().joinpath('baseline_UserAtrrKNN_df.csv') 
    input_file = str(input_file)

    prediction_dir = Path.cwd().joinpath('UserKnnAtrr')
    prediction_dir  = str(prediction_dir)

    only_user_perso = Path.cwd().joinpath('user_perso_metadata.csv') 
    print("metric: ", metric)

    for num_neighbors in range(1, 51):

        print("k neighbors= ", num_neighbors)
        recommender = UserAttributeKNN(similarity_metric=metric, metadata_file=only_user_perso,
                                        k_neighbors= num_neighbors)

        CrossValidation(input_file, recommender, prediction_dir, k_folds = 10, header=None,
                        write_predictions = True).compute()


def UserKNN_UserAttrKNN_graphic():

    plot_info = []

    baseline_UserKNN = files.user_KNN_file
    MRSE_values_list_UserKNN = graphics.MRSE_values_from_file(baseline_UserKNN)
    plot_0 = {'RMSE_values': MRSE_values_list_UserKNN, 'legend': 'UserKNN - baseline', 'color': 'rebeccapurple'}

    plot_info.append(plot_0)

    proposal_UserAttrKNN = files.UserATTRKNN_file_chebyshev
    MRSE_values_list_UserAttrKNN = graphics.MRSE_values_from_file(proposal_UserAttrKNN)
    plot_1 = {'RMSE_values': MRSE_values_list_UserAttrKNN, 'legend': 'UserAttrKNN - chebyshev', 'color': 'blue'}

    plot_info.append(plot_1)

    proposal_UserAttrKNN = files.UserATTRKNN_file_manhattan
    MRSE_values_list_UserAttrKNN = graphics.MRSE_values_from_file(proposal_UserAttrKNN)
    plot_2 = {'RMSE_values': MRSE_values_list_UserAttrKNN, 'legend': 'UserAttrKNN - manhattan', 'color': 'tomato'}

    plot_info.append(plot_2)


    proposal_UserAttrKNN = files.UserATTRKNN_file_minkowski
    MRSE_values_list_UserAttrKNN = graphics.MRSE_values_from_file(proposal_UserAttrKNN)
    plot_3 = {'RMSE_values': MRSE_values_list_UserAttrKNN, 'legend': 'UserAttrKNN - minkopwski', 'color': 'gold'}

    plot_info.append(plot_3)

    graphics.new_proposals_graphic(6, 50, plot_info, "Comparação do baseline UserKNN com as propostas de UserAttrKNN")

    
if __name__ == '__main__':

    #generate_metadata_file_df()

    #user_attr_knn_rating_prediction('minkowski')

    #user_attr_knn_rating_prediction('cityblock')

    #user_attr_knn_rating_prediction('chebyshev')

    UserKNN_UserAttrKNN_graphic()