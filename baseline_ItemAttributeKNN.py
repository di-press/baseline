
import pandas as pd
from pathlib import Path
from caserec.utils.cross_validation import CrossValidation
from caserec.recommenders.rating_prediction.userknn import ItemAttributeKNN

def item_atrr_knn_experiments(similarity, num_neighbors):

    #complete database file, without splitting
    input_file = Path.cwd().joinpath('baseline_df.csv') 
    input_file = str(input_file)

    prediction_dir = Path.cwd().joinpath('knn_rating_prediction')
    prediction_dir  = str(prediction_dir)

    recommender = ItemAttributeKNN(similarity_metric= similarity, k_neighbors= num_neighbors)

    CrossValidation(input_file, recommender, prediction_dir, k_folds = 10, 
                    header=1, recommender_verbose=False, as_table=False,
                    evaluation_in_fold_verbose=False,del_folds=True,
                    write_predictions = False).compute()