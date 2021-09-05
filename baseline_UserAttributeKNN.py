
import pandas as pd
from pathlib import Path
from caserec.utils.cross_validation import CrossValidation
from caserec.recommenders.rating_prediction.user_attribute_knn import UserAttributeKNN


def metadata_file_df():
    
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
    final_df.to_csv(path_temp_csv, sep='\t', index = False, header=True)


    user_interaction_df = pd.DataFrame(baseline_df_tuples,
                        columns = ['new_user_id',
                                   'movielens_Id',
                                   'movie_rating']                                       
                                )

    # data containing the interaction for usrAttKNN:
    path_temp_csv = Path.cwd().joinpath('baseline_UserAtrrKNN_df.csv')
    # index = True star indexing the csv file with 0:
    user_interaction_df.to_csv(path_temp_csv, sep='\t', index = False, header=True)

def user_hashes_to_id(user_id_list):

    id_set = set(user_id_list)

    new_user_id = list(range( 0, len(id_set)))

    meu_zip = list(zip(new_user_id, id_set))

    hash_id_dict = {}

    for item in meu_zip:

        hash_id_dict[item[1]] = item[0]

    #print(hash_id_dict)
    return hash_id_dict

def user_knn_rating_prediction_cosine(metric):

    #complete database file, without splitting
    input_file = Path.cwd().joinpath('baseline_df.csv') 
    input_file = str(input_file)

    prediction_dir = Path.cwd().joinpath('UserKnnAtrr')
    prediction_dir  = str(prediction_dir)

    only_user_perso = Path.cwd().joinpath('user_perso_metadata.csv') 

    recommender = UserAttributeKNN(similarity_metric=metric, metadata_file=only_user_perso )

    CrossValidation(input_file, recommender, prediction_dir, k_folds = 10, header=1,
                    write_predictions = True).compute()


if __name__ == '__main__':

    metadata_file_df()