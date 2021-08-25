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

    path_temp_csv = Path.cwd().joinpath('shuffled_baseline_df.csv')
    # index = True star indexing the csv file with 0:
    df.to_csv(path_temp_csv, index = False, header=False)

# the file shuffled_baseline_df.csv need to exist before executing
# the function above!
# the file can be created with the function df_for_baseline()
def knn_rating_prediction():

    #complete database file, without splitting
    input_file = Path.cwd().joinpath('shuffled_baseline_df.csv') 
    prediction_dir = Path.cwd().joinpath('knn_rating_prediction.txt')
    recommender = UserKNN()

    CrossValidation(input_file, recommender, prediction_dir, k_folds = 10, header=1,
                    write_predictions = True).compute()


if __name__ == '__main__':

    #df_for_baseline()
    #print("oi")
    
    knn_rating_prediction()