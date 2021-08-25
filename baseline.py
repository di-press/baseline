import pandas as pd
from pathlib import Path
from caserec.utils.cross_validation import CrossValidation
from caserec.recommenders.rating_prediction.userknn import UserKNN


def df_for_baseline():
    
    csv_file = Path.cwd().joinpath('final_shuffled_dataset.csv')


    df = pd.read_csv(csv_file,
                        index_col= False,
                        sep=",", warn_bad_lines=True, 
                        error_bad_lines=True,
                        engine='python',
                        header=0,
                        usecols = ['user_id',
                        'movielens_Id',
                        'movie_rating']                                                                        
                    )
    
    
    df.reset_index(drop=True, inplace=False)

    user_id_list = list(df['user_id'])

    new_users_ids = user_hashes_to_id(user_id_list)

    #df.drop('user_id', axis='columns', inplace=True)

    items_tuples = []

    for index, row in df.iterrows():

        current_hash_id = row['user_id']
        new_numerical_id = new_users_ids[current_hash_id]
        
        current_movielens_id = row['movielens_Id']
        current_movie_rating = row['movie_rating']
        
        row_values = (new_numerical_id, current_movielens_id, current_movie_rating) 
        items_tuples.append(row_values)
        #print(row_values)

    final_df = pd.DataFrame(items_tuples,
                        columns = ['new_user_id',
                                   'movielens_Id',
                                   'movie_rating']                                       
                                )

    # data containing the plots and its personality:
    path_temp_csv = Path.cwd().joinpath('baseline_df.csv')
    # index = True star indexing the csv file with 0:
    final_df.to_csv(path_temp_csv, sep='\t', index = False, header=True)
    


def user_hashes_to_id(user_id_list):

    id_set = set(user_id_list)

    new_user_id = list(range( 0, len(id_set)))

    meu_zip = list(zip(new_user_id, id_set))

    hash_id_dict = {}

    for item in meu_zip:

        hash_id_dict[item[1]] = item[0]

    #print(hash_id_dict)
    return hash_id_dict


def df_for_test():

    csv_file = Path.cwd().joinpath('final_shuffled_dataset.csv')


    df = pd.read_csv(csv_file,
                        index_col= False,
                        sep=",", warn_bad_lines=True, 
                        error_bad_lines=True,
                        engine='python',
                        header=0,
                        usecols = ['user_id',
                        'movielens_Id',
                        'movie_rating']                                                                        
                    )
    
    
    df.reset_index(drop=True, inplace=False)

    path_temp_csv = Path.cwd().joinpath('baseline_df.csv')
    # index = True star indexing the csv file with 0:
    df.to_csv(path_temp_csv, sep='\t', index = False, header=True)

# the file     baseline_df.csv need to exist before executing
# the function above!
# the file can be created with the function df_for_baseline()
def knn_rating_prediction():

    #complete database file, without splitting
    input_file = Path.cwd().joinpath('baseline_df.csv') 
    input_file = str(input_file)

    prediction_dir = Path.cwd().joinpath('knn_rating_prediction')
    prediction_dir  = str(prediction_dir)

    recommender = UserKNN()

    CrossValidation(input_file, recommender, prediction_dir, k_folds = 10, header=1,
                    write_predictions = True).compute()


if __name__ == '__main__':

    df_for_baseline()
    
    
    knn_rating_prediction()
