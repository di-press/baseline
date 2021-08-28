import pandas as pd
from pathlib import Path
from caserec.utils.cross_validation import CrossValidation
from caserec.recommenders.rating_prediction.itemknn import ItemKNN


# uses cosine
def item_knn_rating_prediction_cosine():

    #complete database file, without splitting
    input_file = Path.cwd().joinpath('baseline_df.csv') 
    input_file = str(input_file)

    prediction_dir = Path.cwd().joinpath('knn_rating_prediction')
    prediction_dir  = str(prediction_dir)

    recommender = ItemKNN(similarity_metric="hamming", k_neighbors=3)

    CrossValidation(input_file, recommender, prediction_dir, k_folds = 10, header=1,
                    write_predictions = True).compute()


# uses hamming
def item_knn_rating_prediction_hamming():

    #complete database file, without splitting
    input_file = Path.cwd().joinpath('baseline_df.csv') 
    input_file = str(input_file)

    prediction_dir = Path.cwd().joinpath('knn_rating_prediction')
    prediction_dir  = str(prediction_dir)

    recommender = ItemKNN(similarity_metric="hamming", k_neighbors=3)

    CrossValidation(input_file, recommender, prediction_dir, k_folds = 10, header=1,
                    write_predictions = True).compute()


def item_knn_experiments(similarity, num_neighbors):

    #complete database file, without splitting
    input_file = Path.cwd().joinpath('baseline_df.csv') 
    input_file = str(input_file)

    prediction_dir = Path.cwd().joinpath('knn_rating_prediction')
    prediction_dir  = str(prediction_dir)

    recommender = ItemKNN(similarity_metric= similarity, k_neighbors= num_neighbors)

    CrossValidation(input_file, recommender, prediction_dir, k_folds = 10, 
                    header=1, recommender_verbose=False, as_table=False,
                    evaluation_in_fold_verbose=False,del_folds=True,
                    write_predictions = False).compute()


def generate_up_to_50_neighbors_RMSE(similarity):

    #neighbors from 1 to 50:
    for num_neighbors in range(1, 51):

        print("k= ", num_neighbors)
        item_knn_experiments(similarity, num_neighbors)

if __name__ == '__main__':

    #item_knn_rating_prediction_cosine()
    #item_knn_rating_prediction_hamming()

    #generate_up_to_50_neighbors_RMSE("cosine")

    #generate_up_to_50_neighbors_RMSE("hamming")

    #generate_up_to_50_neighbors_RMSE("minkowski")

    #generate_up_to_50_neighbors_RMSE("chebyshev")

    generate_up_to_50_neighbors_RMSE("cityblock")

    #generate_up_to_50_neighbors_RMSE("jaccard")