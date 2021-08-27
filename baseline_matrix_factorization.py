from caserec.recommenders.rating_prediction.matrixfactorization import MatrixFactorization
from pathlib import Path
from caserec.utils.cross_validation import CrossValidation

def matrix_fact_rating_prediction():

    #complete database file, without splitting
    input_file = Path.cwd().joinpath('baseline_df.csv') 
    input_file = str(input_file)

    prediction_dir = Path.cwd().joinpath('matrix_fact_rating_prediction')
    prediction_dir  = str(prediction_dir)

    recommender = MatrixFactorization()

    CrossValidation(input_file, recommender, prediction_dir, k_folds = 10, header=1,
                    write_predictions = True).compute()


if __name__ == '__main__':

    matrix_fact_rating_prediction()