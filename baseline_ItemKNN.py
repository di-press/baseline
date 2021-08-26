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


if __name__ == '__main__':

    item_knn_rating_prediction_cosine()
    #item_knn_rating_prediction_hamming()