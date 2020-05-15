import gudhi
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D


def compute_similarity_matrix(simplex_trees, dimension=0, e=.001, save=False, filtration='rips'):
    print('Computing {}d persistence diagrams bottleneck distance matrix'.format(dimension))
    sim_matrix = np.zeros((len(simplex_trees), len(simplex_trees)))
    counter = 0
    for i in range(len(simplex_trees)):
        i_persistence = simplex_trees[i].persistence_intervals_in_dimension(dimension)
        for j in range(i + 1, 300):
            if not counter % 5000:
                print('Computing bottleneck distance {}/{}'
                      .format(counter, (len(simplex_trees) * (len(simplex_trees) - 1)) // 2))

            j_persistence = simplex_trees[j].persistence_intervals_in_dimension(dimension)
            bot_dist = gudhi.bottleneck_distance(i_persistence, j_persistence, e=e)
            sim_matrix[i][j] = bot_dist
            sim_matrix[j][i] = bot_dist
            counter += 1

    if save:
        np.savetxt('intermediary_data/similarity_matrices/{}_{}_similarity_matrix.csv'
                   .format(filtration, dimension), sim_matrix, delimiter=',')

    return sim_matrix


def plot_similarity_matrix_embeddings(sim_matrix, mda_dimension=2, diagram_dimension=0, filtration='rips', save=True):
    if mda_dimension not in (2, 3):
        raise ValueError('Parameter mda_dimension must be either 2 or 3')

    mds = MDS(n_components=mda_dimension, max_iter=3000, eps=1e-9, dissimilarity="precomputed", n_jobs=10)
    sim_matrix_embed = mds.fit(sim_matrix).embedding_

    if mda_dimension == 2:
        plt.scatter(sim_matrix_embed[0:100, 0], sim_matrix_embed[0:100, 1], label='A', color='red')
        plt.scatter(sim_matrix_embed[100:200, 0], sim_matrix_embed[100:200, 1], label='B', color='blue')
        plt.scatter(sim_matrix_embed[200:300, 0], sim_matrix_embed[200:300, 1], label='C', color='green')

    elif mda_dimension == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(sim_matrix_embed[0:100, 0], sim_matrix_embed[0:100, 1], sim_matrix_embed[0:100, 2],
                   label='A', color='red', marker='o', s=7)
        ax.scatter(sim_matrix_embed[100:200, 0], sim_matrix_embed[100:200, 1], sim_matrix_embed[100:200, 2],
                   label='B', color='blue', marker='o', s=7)
        ax.scatter(sim_matrix_embed[200:300, 0], sim_matrix_embed[200:300, 1], sim_matrix_embed[200:300, 2],
                   label='C', color='green', marker='o', s=7)

    plt.title('{}d embedding of the {}d persistence diagram bottlneck \n'
              'distance matrix for the {} filtration'.format(mda_dimension, diagram_dimension, filtration))
    plt.legend(title='Pedestrian', loc='lower left')

    if save:
        plt.savefig('plots/{}_{}d_diagram_{}d_embedding.png'.format(filtration, diagram_dimension, mda_dimension))

    plt.show()


def compute_similarities(simplex_trees, filtration='rips', save=True):
    sim_matrix_0 = compute_similarity_matrix(simplex_trees, dimension=0, e=.001, save=save)
    plot_similarity_matrix_embeddings(sim_matrix_0, mda_dimension=2, diagram_dimension=0,
                                      filtration=filtration, save=save)
    plot_similarity_matrix_embeddings(sim_matrix_0, mda_dimension=3, diagram_dimension=0,
                                      filtration=filtration, save=save)

    sim_matrix_1 = compute_similarity_matrix(simplex_trees, dimension=1, e=.001, save=save)
    plot_similarity_matrix_embeddings(sim_matrix_1, mda_dimension=2, diagram_dimension=1,
                                      filtration=filtration, save=save)
    plot_similarity_matrix_embeddings(sim_matrix_1, mda_dimension=3, diagram_dimension=1,
                                      filtration=filtration, save=save)

