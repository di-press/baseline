	
# Wilcoxon signed-rank test

import graphics
#from pathlib import Path
import files
from numpy.random import seed
from numpy.random import randn
from scipy.stats import wilcoxon


def wilcoxon_rank_test(MRSE_baseline_values, MRSE_proposal_values, title):

	stat, p = wilcoxon(MRSE_baseline_values, MRSE_proposal_values)
	print(title, "\n")
	print('Statistics=%.7f, p=%.15f' % (stat, p))
	# interpret
	alpha = 0.05
	if p > alpha:
		print('Same distribution (fail to reject H0)')
	else:
		print('Different distribution (reject H0)')


def all_p_values_proposal_1():

	proposal_1_file= files.proposal_1_file
	MRSE_values_list_proposal = graphics.read_csv_regr_knn(proposal_1_file)
	MRSE_proposal_values = MRSE_values_list_proposal


	baseline_hamming_userknn_file = files.user_KNN_file
	MRSE_values_list_baseline = graphics.MRSE_values_from_file(baseline_hamming_userknn_file)
	MRSE_baseline_values = MRSE_values_list_baseline[:-1]

	# proposal 1 X UserKNN
	wilcoxon_rank_test(MRSE_baseline_values, MRSE_proposal_values,
						"Proposta 1 (perso. usuário + perso. filme) x UserKNN")

	item_knn_file = files.item_KNN_file
	MRSE_baseline_values = graphics.MRSE_values_from_file(item_knn_file)
	MRSE_baseline_values = MRSE_baseline_values[:-1]

	# Proposal 1 X ItemKNN
	wilcoxon_rank_test(MRSE_baseline_values, MRSE_proposal_values,
						"Proposta 1 (perso. usuário + perso. filme) x ItemKNN")


	baseline_matrix_fact = files.matrix_fact_file
	MRSE_values_list_matr_fact = graphics.MRSE_values_from_file(baseline_matrix_fact)
	MRSE_baseline_values = MRSE_values_list_matr_fact[:-1]

	# Proposal 1 x matrix factorization
	wilcoxon_rank_test(MRSE_baseline_values, MRSE_proposal_values,
						"Proposta 1 (perso. usuário + perso. filme) x Fatoração de Matrizes")
	

def all_p_values_proposal_2():

	proposal_2_file= files.proposal_2_file
	MRSE_values_list_proposal = graphics.read_csv_regr_knn(proposal_2_file)
	MRSE_proposal_values = MRSE_values_list_proposal


	baseline_hamming_userknn_file = files.user_KNN_file
	MRSE_values_list_baseline = graphics.MRSE_values_from_file(baseline_hamming_userknn_file)
	MRSE_baseline_values = MRSE_values_list_baseline[:-1]

	# proposal 2 X UserKNN
	wilcoxon_rank_test(MRSE_baseline_values, MRSE_proposal_values,
						"Proposta 2 (perso. usuário + perso. filme + gêneros) x UserKNN")

	item_knn_file = files.item_KNN_file
	MRSE_baseline_values = graphics.MRSE_values_from_file(item_knn_file)
	MRSE_baseline_values = MRSE_baseline_values[:-1]

	# Proposal 2 X ItemKNN
	wilcoxon_rank_test(MRSE_baseline_values, MRSE_proposal_values,
						"Proposta 2 (perso. usuário + perso. filme + gêneros) x ItemKNN")

	baseline_matrix_fact = files.matrix_fact_file
	MRSE_values_list_matr_fact = graphics.MRSE_values_from_file(baseline_matrix_fact)
	MRSE_baseline_values = MRSE_values_list_matr_fact[:-1]

	# Proposal 2 x matrix factorization
	wilcoxon_rank_test(MRSE_baseline_values, MRSE_proposal_values,
						"Proposta 2 (perso. usuário + perso. filme + gêneros) x Fatoração de Matrizes")

# Proposal 3 :only user perso:
def all_p_values_proposal_3():

	proposal_3_file= files.proposal_3_file
	MRSE_values_list_proposal = graphics.read_csv_regr_knn(proposal_3_file)
	MRSE_proposal_values = MRSE_values_list_proposal


	baseline_hamming_userknn_file = files.user_KNN_file
	MRSE_values_list_baseline = graphics.MRSE_values_from_file(baseline_hamming_userknn_file)
	MRSE_baseline_values = MRSE_values_list_baseline[:-1]

	# proposal 3 X UserKNN
	wilcoxon_rank_test(MRSE_baseline_values, MRSE_proposal_values,
						"Proposta 3 (só perso. usuário) x UserKNN")

	item_knn_file = files.item_KNN_file
	MRSE_baseline_values = graphics.MRSE_values_from_file(item_knn_file)
	MRSE_baseline_values = MRSE_baseline_values[:-1]

	# Proposal 3 X ItemKNN
	wilcoxon_rank_test(MRSE_baseline_values, MRSE_proposal_values,
						"Proposta 3 (só perso. usuário) x ItemKNN")

	baseline_matrix_fact = files.matrix_fact_file
	MRSE_values_list_matr_fact = graphics.MRSE_values_from_file(baseline_matrix_fact)
	MRSE_baseline_values = MRSE_values_list_matr_fact[:-1]
	
	# Proposal 3 x matrix factorization
	wilcoxon_rank_test(MRSE_baseline_values, MRSE_proposal_values,
					"Proposta 3 (só perso. usuário) x Fatoração de Matrizes")

def all_p_values_proposal_4():

	proposal_4_file= files.proposal_4_file
	MRSE_values_list_proposal = graphics.read_csv_regr_knn(proposal_4_file)
	MRSE_proposal_values = MRSE_values_list_proposal


	baseline_hamming_userknn_file = files.user_KNN_file
	MRSE_values_list_baseline = graphics.MRSE_values_from_file(baseline_hamming_userknn_file)
	MRSE_baseline_values = MRSE_values_list_baseline[:-1]

	# proposal 4 X UserKNN
	wilcoxon_rank_test(MRSE_baseline_values, MRSE_proposal_values,
						"Proposta 4 (só perso. filme) x UserKNN")

	item_knn_file = files.item_KNN_file
	MRSE_baseline_values = graphics.MRSE_values_from_file(item_knn_file)
	MRSE_baseline_values = MRSE_baseline_values[:-1]

	# Proposal 4 X ItemKNN
	wilcoxon_rank_test(MRSE_baseline_values, MRSE_proposal_values,
						"Proposta 4 (só perso. filme) x ItemKNN")


	baseline_matrix_fact = files.matrix_fact_file
	MRSE_values_list_matr_fact = graphics.MRSE_values_from_file(baseline_matrix_fact)
	MRSE_baseline_values = MRSE_values_list_matr_fact[:-1]

	# Proposal 4 x matrix factorization
	wilcoxon_rank_test(MRSE_baseline_values, MRSE_proposal_values,
						"Proposta 4 (só perso. filme) x Fatoração de Matrizes")

if __name__ == '__main__':

	#all_p_values_proposal_1()

	#all_p_values_proposal_2()

	#all_p_values_proposal_3()

	all_p_values_proposal_4()

