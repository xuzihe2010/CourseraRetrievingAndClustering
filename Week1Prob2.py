"""
Created on Aug 30 2017 3:08 PM 

@author: ZiheTony
@project: CourseraRetrievingAndClustering
@file: Week1Prob2
"""

import numpy as np
import pandas as pd
import json
from scipy.sparse import csr_matrix
from collections import defaultdict, namedtuple
from itertools import combinations
from sklearn.metrics.pairwise import pairwise_distances
import time
import matplotlib.pyplot as plt
from copy import copy


ROOT = 'C:/Users/ZiheTony/Documents/Coursera Machine Learning Assignment Data'

LSHModel = namedtuple( 'LSHModel', ['bin_index_bits', 'bin_indices', 'table', 'random_vectors', 'num_vector'] )


class RetrieveWeek2Prob2Data( object ):

    def __init__(self):
        self.wiki_data = pd.read_csv( '/'.join( [ ROOT, 'people_wiki.csv' ] ) )

        with open( '/'.join( [ ROOT, 'people_wiki_map_index_to_word.json' ] ) ) as map_idx_to_word_data:
            self.map_idx_to_word = json.load( map_idx_to_word_data )

        self.wiki_data.reset_index(inplace = True)
        self.wiki_data.rename( columns = { 'index' : 'id' }, inplace = True )
        self.corpus = self.load_sparse_csr( '/'.join( [ ROOT, 'people_wiki_tf_idf.npz' ] ) )

    def load_sparse_csr(self, file_name):
        ''' Load npz file to create scipy sparse matrix for building model. '''
        loader = np.load( file_name )

        data = loader[ 'data' ]
        indices = loader[ 'indices' ]
        indptr = loader[ 'indptr' ]
        shape = loader[ 'shape' ]

        return csr_matrix( ( data, indices, indptr ), shape )

    def return_data(self):
        return self.wiki_data, self.map_idx_to_word, self.corpus


class RetrieveWeek1Prob2( object ):

    def __init__(self, wiki_data, tf_idf):
        self.wiki_data = wiki_data
        self.tf_idf = tf_idf

    @staticmethod
    def norm(x):
        sum_sq = x.dot(x.T)
        norm = np.sqrt(sum_sq)
        return (norm)

    @staticmethod
    def generate_random_vectors(num_vector, dim):
        return np.random.randn(dim, num_vector)

    @staticmethod
    def visualize_random_vectors():
        # Generate 3 random vectors of dimension 5, arranged into a single 5 x 3 matrix.
        np.random.seed(0)
        print RetrieveWeek1Prob2.generate_random_vectors(num_vector = 3, dim = 5)

    def train_LSH_model(self, num_vector = 16, seed = None):

        dim = self.tf_idf.shape[1]

        if not seed is None:
            np.random.seed( seed )

        random_vectors = RetrieveWeek1Prob2.generate_random_vectors(num_vector, dim)
        powers_of_two = (1 << np.arange(num_vector - 1, -1, -1))

        hash_table = defaultdict(list)

        bin_index_bits = (self.tf_idf.dot(random_vectors) >= 0)
        bin_indices = bin_index_bits.dot(powers_of_two)

        for i, bin_index in enumerate(bin_indices):
            hash_table[bin_index].append(i)

        return LSHModel(bin_index_bits, bin_indices, hash_table, random_vectors, num_vector)

    def get_person_index(self, name):
        return self.wiki_data[self.wiki_data['name'] == name].iloc[0]['id']

    def get_bin_index(self, model, index):
        return model.bin_indices[ index ]

    def count_bit_match(self, model, name1, name2, num_vectors):
        index1 = self.get_person_index(name1)
        index2 = self.get_person_index(name2)

        overlap = (self.get_bin_index(model, index1) ^ self.get_bin_index(model, index2))
        count = 0
        while overlap:
            if 1 & overlap != 0:
                count += 1
            overlap >>= 1

        return num_vectors - count

    def get_docs_in_same_bin(self, model, name):
        index = self.get_person_index(name)
        bin_index = self.get_bin_index(model, index)

        return self.wiki_data.iloc[ model.table[ bin_index ] ]

    @staticmethod
    def cosine_distance(x, y):
        dot_multi = x.dot(y.transpose())
        dist = dot_multi / ( RetrieveWeek1Prob2.norm(x) * RetrieveWeek1Prob2.norm(y) )

        return 1 - dist[0, 0]

    @staticmethod
    def search_nearby_bins(query_bin_bits, table, search_radius = 2, initial_candidates = None):
        if not initial_candidates:
            initial_candidates = set()

        num_vector = len(query_bin_bits)
        powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)

        # Allow the user to provide an initial set of candidates.
        candidate_set = copy(initial_candidates)

        for different_bits in combinations(range(num_vector), search_radius):
            # Flip the bits (n_1,n_2,...,n_r) of the query bin to produce a new bit vector.
            ## Hint: you can iterate over a tuple like a list
            alternate_bits = copy(query_bin_bits)
            for i in different_bits:
                alternate_bits[i] = False if query_bin_bits[i] else True

            # Convert the new bit vector to an integer index
            nearby_bin = alternate_bits.dot(powers_of_two)

            # Fetch the list of documents belonging to the bin indexed by the new bit vector.
            # Then add those documents to candidate_set
            candidate_set.update(table.get(nearby_bin, []))

        return candidate_set

    def query_nearest_neighbor(self, query_vector, model, k, max_search_radius):
        table = model.table
        random_vectors = model.random_vectors

        # Compute bin index for the query vector, in bit representation.
        bin_index_bits = (query_vector.dot(random_vectors) >= 0).flatten()

        # Search nearby bins and collect candidates
        candidate_set = set()
        for search_radius in xrange(max_search_radius + 1):
            candidate_set = RetrieveWeek1Prob2.search_nearby_bins(bin_index_bits, table, search_radius,
                                                                  initial_candidates=candidate_set)

        # Sort candidates by their true distances from the query
        nearest_neighbors = pd.DataFrame({'id': list( candidate_set ) })
        candidates = self.tf_idf[np.array(list(candidate_set)), :]
        nearest_neighbors['distance'] = pairwise_distances(candidates, query_vector, metric='cosine').flatten()

        return nearest_neighbors.sort('distance', ascending=True).iloc[:k+1], len(candidate_set)

    # Print functions - print out intermediate step results
    def print_cosine_distance(self, name1, name2):
        first_tf_idf = self.tf_idf[self.get_person_index(name1)]
        second_tf_idf = self.tf_idf[self.get_person_index(name2)]

        res = RetrieveWeek1Prob2.cosine_distance(first_tf_idf, second_tf_idf)
        print '================= Cosine distance from %s' % name1
        print '%s - {0:24s}: {1:f}'.format(name2, res) % name1

    def print_similarity_in_same_bin(self, model, name = 'Barack Obama'):
        docs = self.get_docs_in_same_bin(model, name)

        for person in docs[ 'name' ]:
            if person != name:
                self.print_cosine_distance(name, person)

    def print_obama_neighbors(self, model, index = 35817, k = 10, max_search_radius = 3):
        result, num_candidates_considered = self.query_nearest_neighbor(self.tf_idf[index], model, k, max_search_radius)

        print pd.merge(result, self.wiki_data[['id', 'name']], on = 'id', how = 'inner').sort('distance')

    # Checkpoints: check if the intermediate results are correct
    def check_point1(self, model):
        table = model.table

        if 0 in table and table[0] == [39583] and 143 in table and table[143] == [19693, 28277, 29776, 30399]:
            print 'Passed check point1'
        else:
            print 'Failed check point1'

    def check_point2(self, model, name = 'Barack Obama'):
        bin_index = model.bin_index_bits[ self.get_person_index(name) ]
        candidate_set = RetrieveWeek1Prob2.search_nearby_bins(bin_index, model.table, search_radius=0)

        if candidate_set == set([35817, 21426, 53937, 39426, 50261]):
            print 'Passed check point2'
        else:
            print 'Failed check point2'
            print 'Docs in the same bin with %s is: ' % name, candidate_set

        return candidate_set

    def check_point3(self, model, name = 'Barack Obama'):
        bin_index = model.bin_index_bits[self.get_person_index(name)]
        initial_set = self.check_point2(model, name)
        candidate_set = RetrieveWeek1Prob2.search_nearby_bins(  bin_index,
                                                                model.table,
                                                                search_radius=1,
                                                                initial_candidates = initial_set)

        if candidate_set == set([39426, 38155, 38412, 28444, 9757, 41631, 39207, 59050, 47773, 53937, 21426, 34547,
                                 23229, 55615, 39877, 27404, 33996, 21715, 50261, 21975, 33243, 58723, 35817, 45676,
                                 19699, 2804, 20347]):
            print 'Passed check point3'
        else:
            print 'Failed check point3'

    # Experiments: testing LSH performance and other features
    def experiment_with_LSH(self, model, exp_radius = 17):
        num_candidates_history = []
        query_time_history = []
        max_distance_from_query_history = []
        min_distance_from_query_history = []
        average_distance_from_query_history = []

        for max_search_radius in xrange(exp_radius):
            start = time.time()
            # Perform LSH query using Barack Obama, with max_search_radius
            result, num_candidates = self.query_nearest_neighbor(self.tf_idf[35817, :], model, k=10,
                                                                 max_search_radius=max_search_radius)
            end = time.time()
            query_time = end - start  # Measure time

            print 'Radius:', max_search_radius
            # Display 10 nearest neighbors, along with document ID and name
            print pd.merge(result, self.wiki_data[['id', 'name']], on='id', how = 'inner').sort('distance')

            # Collect statistics on 10 nearest neighbors
            average_distance_from_query = result['distance'][1:].mean()
            max_distance_from_query = result['distance'][1:].max()
            min_distance_from_query = result['distance'][1:].min()

            num_candidates_history.append(num_candidates)
            query_time_history.append(query_time)
            average_distance_from_query_history.append(average_distance_from_query)
            max_distance_from_query_history.append(max_distance_from_query)
            min_distance_from_query_history.append(min_distance_from_query)

        print [ 'radius %i : avg distance %.5f' % (radius, avg_dis) for radius, avg_dis in zip(range(exp_radius), average_distance_from_query_history) ]
        self.plot_radius_growth(num_candidates_history,
                                query_time_history,
                                average_distance_from_query_history,
                                max_distance_from_query_history,
                                min_distance_from_query_history)

    def plot_radius_growth(self, num_candidates_history,
                           query_time_history,
                           average_distance_from_query_history,
                           max_distance_from_query_history,
                           min_distance_from_query_history):
        plt.figure(figsize=(7, 4.5))
        plt.plot(num_candidates_history, linewidth=4)
        plt.xlabel('Search radius')
        plt.ylabel('# of documents searched')
        plt.rcParams.update({'font.size': 16})
        plt.tight_layout()

        plt.figure(figsize=(7, 4.5))
        plt.plot(query_time_history, linewidth=4)
        plt.xlabel('Search radius')
        plt.ylabel('Query time (seconds)')
        plt.rcParams.update({'font.size': 16})
        plt.tight_layout()

        plt.figure(figsize=(7, 4.5))
        plt.plot(average_distance_from_query_history, linewidth=4, label='Average of 10 neighbors')
        plt.plot(max_distance_from_query_history, linewidth=4, label='Farthest of 10 neighbors')
        plt.plot(min_distance_from_query_history, linewidth=4, label='Closest of 10 neighbors')
        plt.xlabel('Search radius')
        plt.ylabel('Cosine distance of neighbors')
        plt.legend(loc='best', prop={'size': 15})
        plt.rcParams.update({'font.size': 16})
        plt.tight_layout()

    def brute_force_query(self, vector, k):
        num_data_points = self.tf_idf.shape[0]

        # Compute distances for ALL data points in training set
        nearest_neighbors = pd.DataFrame({'id': range(num_data_points)})
        nearest_neighbors['distance'] = pairwise_distances(self.tf_idf, vector, metric='cosine').flatten()

        return nearest_neighbors.sort('distance', ascending=True).iloc[:k+1]

    def experiment_with_LSH_compare_with_brute_force(self, model, max_radius = 17):

        precision = {i: [] for i in xrange(max_radius)}
        average_distance = {i: [] for i in xrange(max_radius)}
        query_time = {i: [] for i in xrange(max_radius)}

        np.random.seed(0)
        num_queries = 10
        for i, ix in enumerate(np.random.choice(self.tf_idf.shape[0], num_queries, replace=False)):
            print('%s / %s' % (i, num_queries))
            ground_truth = set(self.brute_force_query(self.tf_idf[ix, :], k=25)['id'])
            # Get the set of 25 true nearest neighbors

            for r in xrange(1, max_radius):
                start = time.time()
                result, num_candidates = self.query_nearest_neighbor(self.tf_idf[ix, :], model, k=10, max_search_radius=r)
                end = time.time()

                query_time[r].append(end - start)
                # precision = (# of neighbors both in result and ground_truth)/10.0
                precision[r].append(len(set(result['id']) & ground_truth) / 10.0)
                average_distance[r].append(result['distance'][1:].mean())

        self.plot_LSH_compare_with_brute_force(average_distance, precision, query_time)

    def plot_LSH_compare_with_brute_force(self, average_distance,
                                          precision,
                                          query_time):
        plt.figure(figsize=(7, 4.5))
        plt.plot(range(1, 17), [np.mean(average_distance[i]) for i in xrange(1, 17)], linewidth=4,
                 label='Average over 10 neighbors')
        plt.xlabel('Search radius')
        plt.ylabel('Cosine distance')
        plt.legend(loc='best', prop={'size': 15})
        plt.rcParams.update({'font.size': 16})
        plt.tight_layout()

        plt.figure(figsize=(7, 4.5))
        plt.plot(range(1, 17), [np.mean(precision[i]) for i in xrange(1, 17)], linewidth=4, label='Precison@10')
        plt.xlabel('Search radius')
        plt.ylabel('Precision')
        plt.legend(loc='best', prop={'size': 15})
        plt.rcParams.update({'font.size': 16})
        plt.tight_layout()

        plt.figure(figsize=(7, 4.5))
        plt.plot(range(1, 17), [np.mean(query_time[i]) for i in xrange(1, 17)], linewidth=4, label='Query time')
        plt.xlabel('Search radius')
        plt.ylabel('Query time (seconds)')
        plt.legend(loc='best', prop={'size': 15})
        plt.rcParams.update({'font.size': 16})
        plt.tight_layout()

    # Answer questions in the assignment
    def question1(self, name = 'Barack Obama'):
        return self.get_person_index(name)

    def question2(self, model, name = 'Barack Obama'):
        index = self.get_person_index(name)
        return self.get_bin_index(model, index)

    def question3(self, model, name1 = 'Barack Obama', name2 = 'Joe Biden'):
        return self.count_bit_match(model, name1, name2, 16)

    # question 4 and question 5 can be solved by running experiment_with_LSH function, and observing the results

    def dummy_func_for_testing_git(self):
        print 'This function is added to test git.'


# if __name__ == '__main__':
#     data_generator = RetrieveWeek2Prob2Data()
#     sol = RetrieveWeek1Prob2(data_generator.wiki_data, data_generator.corpus)
#     lsh_model = sol.train_LSH_model(seed = 143)
#
#     # Run check points
#     sol.check_point1(lsh_model)
#     sol.check_point2(lsh_model)
#     sol.check_point3(lsh_model)
#
#     # Run assignment questions:
#     print "Question 1: ", sol.question1()
#     print "Question 2: ", sol.question2(lsh_model)
#     print "Question 3: ", sol.question3(lsh_model)
#
#     # Run experiments:
#     sol.experiment_with_LSH(lsh_model)
#     sol.experiment_with_LSH_compare_with_brute_force(lsh_model)































