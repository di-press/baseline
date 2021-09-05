	
# Wilcoxon signed-rank test

import graphics
from pathlib import Path
from numpy.random import seed
from numpy.random import randn
from scipy.stats import wilcoxon


baseline_hamming_userknn_file = Path.cwd().joinpath('UserKNN_hamming_1_to_50_neighbors.txt')
MRSE_values_list_baseline = graphics.MRSE_values_from_file(baseline_hamming_userknn_file)

data_UserKNN_baseline = MRSE_values_list_baseline[:-1]


proposal_hamming_userknn_file= Path.cwd().joinpath('no_genre_normalization_n=50_cv=10_hammingexp25.csv')
MRSE_values_list_proposal = graphics.read_csv_regr_knn(proposal_hamming_userknn_file)

data_proposal = MRSE_values_list_proposal
# compare samples
stat, p = wilcoxon(data_UserKNN_baseline, data_proposal)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')