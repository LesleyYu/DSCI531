I have dataset Caltech.mat. Now I run the following script in ipynb and I get this result:

```Python
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from collections import Counter

caltech_mat = loadmat('Data/Caltech.mat')
caltech_adj = caltech_mat['A']
caltech_info = caltech_mat['local_info']


print("Caltech graph: ")
print(f"number of nodes={caltech_adj.shape[0]}")
print(f"number of edges={int(np.sum(caltech_adj)//2)}")
print()

caltech_gender = np.zeros(caltech_adj.shape[0], dtype=int)
caltech_gender[caltech_info[:, 1] == 1] = 1
caltech_gender[caltech_info[:, 1] == 2] = 2

print("")
print("caltech genders: \n\t#0 = {} \n\t#1 = {} \n\t#2 = {}".format(len(caltech_gender[caltech_gender == 0]), len(caltech_gender[caltech_gender == 1]), len(caltech_gender[caltech_gender == 2])))
```

Caltech graph: 
number of nodes=769
number of edges=16656


caltech genders: 
	#0 = 66 
	#1 = 228 
	#2 = 475

Now I am doing this task:
We want to perform link prediction on it. In link prediction, we have positive edges and negative edges. Positive edges are edges which are in the graph, and negative edges are edges which are not in the graph. In other words, negative edges are edges in complement of the graph. We train the link prediction model on a fraction of positive edges, and we test the model on how well it can retrieve the rest positive edges.

For evaluation, for each node, we first retrieve the top-k incident edges as ranked by scores given by the model, and then count how many of the retrieved edges are in the test edges, thus obtaining precision@k on this node. The average precision@k over all the nodes is used to evaluate the model’s performance on the entire graph.

1. Train-Test split: Use 75% of positive edges (at random) as training edges. Test edges would be the rest 25% of positive edges.
2. Algorithm: Perform link prediction Adamic-Adar and Jaccard Coefficient. The algorithms should output the scores for target edges.
3. Evaluation: Implement the evaluation metric for link prediction based on average precision@k over nodes in the graph. Report the performance on all nodes, on gender1 nodes, and on gender2 nodes. We want to see which algorithm has less bias for genders. On which gender the algorithms give better precision? Which algorithm is more fair?

Below is the code I have, help me complete them. Only write code where it says "TODO":

```Python
def compare_nodes_centrality(centrality_name, chi_measure, chi_gender, cal_measure, cal_gender):
    """
    Show information about centrality measure
    :centrality_name (string): the title of the plot i.e.: PageRank, Degree Centrality, ...
    :chi_measure (dict): Chicago nodes centrality measures. keys are nodes, values are centrality measure
    :chi_gender (list): Gender of Chicago nodes -- chicago_gender array
    :cal_measure (dict): Caltech nodes centrality measures. keys are nodes, values are centrality measure
    :cal_gender (list): Gender of Caltech nodes -- caltech_gender array
    """
    
    
    plt.clf()
    fig, axs = plt.subplots(1, 3)
    
    chi_all = list(chi_measure.values())
    cal_all = list(cal_measure.values())
    axs[0].hist(chi_all, bins=50, color='red', alpha=0.6, label='Chicago', density=False)
    axs[0].hist(cal_all, bins=50, color='blue', alpha=0.6, label='Caltech', density=False)
    axs[0].set_yscale('log')
    axs[0].set_xlabel('value')
    axs[0].set_ylabel('frequency')
    axs[0].legend(loc='upper right')
    axs[0].set_title("All nodes " + centrality_name)
    
    chi_gen_1 = [val for key, val in chi_measure.items() if key in np.where(chi_gender == 1)[0]]
    cal_gen_1 = [val for key, val in cal_measure.items() if key in np.where(cal_gender == 1)[0]]
    axs[1].hist(chi_gen_1, bins=50, color='red', alpha=0.6, label='Chicago', density=False)
    axs[1].hist(cal_gen_1, bins=50, color='blue', alpha=0.6, label='Caltech', density=False)
    axs[1].set_yscale('log')
    axs[1].set_xlabel('value')
    axs[1].legend(loc='upper right')
    axs[1].set_title("Gender 1 " + centrality_name)
    
    chi_gen_2 = [val for key, val in chi_measure.items() if key in np.where(chi_gender == 2)[0]]
    cal_gen_2 = [val for key, val in cal_measure.items() if key in np.where(cal_gender == 2)[0]]
    axs[2].hist(chi_gen_2, bins=50, color='red', alpha=0.6, label='Chicago', density=False)
    axs[2].hist(cal_gen_2, bins=50, color='blue', alpha=0.6, label='Caltech', density=False)
    axs[2].set_yscale('log')
    axs[2].set_xlabel('value')
    axs[2].legend(loc='upper right')
    axs[2].set_title("Gender 2 " + centrality_name)
    
    fig.set_size_inches(18.5, 6.5)
    plt.show()
    
    print("{} mean and std:".format(centrality_name))
    
    print("\tChicago:")
    print("\t\tAll = {0:.3f} (+- {1:.4f})".format(np.mean(chi_all), np.std(chi_all)))
    print("\t\tGender 1 = {0:.3f} (+- {1:.4f})".format(np.mean(chi_gen_1), np.std(chi_gen_1)))
    print("\t\tGender 2 = {0:.3f} (+- {1:.4f})".format(np.mean(chi_gen_2), np.std(chi_gen_2)))
    
    print("\tCaltech:")
    print("\t\tAll = {0:.3f} (+- {1:.4f})".format(np.mean(cal_all), np.std(cal_all)))
    print("\t\tGender 1 = {0:.3f} (+- {1:.4f})".format(np.mean(cal_gen_1), np.std(cal_gen_1)))
    print("\t\tGender 2 = {0:.3f} (+- {1:.4f})".format(np.mean(cal_gen_2), np.std(cal_gen_2)))
    
    print ("")


import networkx as nx

# create the networkx objects for the two graphs
G_chicago = nx.from_scipy_sparse_array(chicago_adj)
G_caltech = nx.from_scipy_sparse_array(caltech_adj)
```


## Analysis of Link Prediction
### Conduct link prediction on Caltech

```Python
edges = list(G_caltech.edges)

# note that G_caltech is an undirected graph
# if there is an edge between node a and b, 
# the edge list will only contain (a, b) or (b, a) to save space, where a<=b or b<=a respectively
edges[:10]

# from edges, sample 75% training edges and 25% test edges
# both edges_train and edges_test should be a list of edges
# TODO. 2pts.
edges_train = 
edges_test = 


print(len(edges_train))
print(len(edges_test))

# construct a training Graph based on training edges
G_train = nx.from_edgelist(edges_train)
G_train.add_nodes_from(G_caltech.nodes)
print(G_train)

# we want the algorithm to score all the possible edges except those in the training set
G_complete = nx.complete_graph(G_caltech.number_of_nodes())
edges_complete = list(G_complete.edges)
edges_to_score = list(set(edges_complete) - set(edges_train))
print(len(edges_to_score))

# link prediction using Adamic-Adar index
# you need to score every edge in edges_to_score
# edge_score_ada should be a dictionary {(u, v): score}
# TODO. 3pts
edge_score_ada = 

print(len(edge_score_ada))

# plot a histogram of all the scores in edge_score_ada
# TODO. 2pts


# link prediction using Jaccard Coefficient
# you need to score every edge in edges_to_score
# edge_score_jac should be a dictionary {(u, v): score}
# TODO. 3pts.
edge_score_jac = 


print(len(edge_score_jac))

# plot a histogram of all the scores in edge_score_jac
# TODO. 2pts


# the link prediction evaluation of the entire graph is based on that of each node
# we only evaluate nodes that have a degree >= 20 and have gender label 1 or 2
from collections import Counter
deg = Counter(list(sum(edges_to_score, ())))
nodes_to_eval = [k for k, v in deg.items() if v >= 20]
nodes_to_eval = [node for node in nodes_to_eval if caltech_gender[node] != 0]

from collections import Counter
print(f'Gender count in nodes_to_eval: {Counter(caltech_gender[nodes_to_eval])}')

def eval_link_prediction(nodes_to_eval, edge_score, gender, edges_test, k):
    """
    :nodes_to_eval (list): a list of nodes to evaluate
    :edge_score (dict): the predicted scores for edges in edges_to_score
    :gender (list): the gender list for all nodes in the graph
    :edges_test (list): the edge list of the test edges
    :k (int): the k in precision@k
    Return the average precision on nodes, the average precision for gender1 nodes, 
    and the average precision for gender2 nodes.
    
    Hint:
    For each node in nodes_to_eval, 
    retrieve the top-k incident edges on this node ranked by the predicted scores given in edge_score.
    Count how many retrieved edges are actually in edges_test.
    In this way, we can obtain the precision of predicted links for this node.
    Compute the precisions for all nodes in nodes_to_eval, return the avearge precision.
    Similarly, compute the average precision for only nodes with gender1/gender2 in nodes_to_eval.

    """
    #TODO. 10pts.
    
    
    return precision, precision_gender1, precision_gender2


# a test case
# this test graph has six nodes [0, 1, 2, 3, 4, 5]
nodes_to_eval_example = [0,1,2,4]
edge_score_example = {(0, 1): 0.27, (0, 2): 0.39, (0, 3): 0.32, 
                      (1, 2): 0.3, (1, 4): 0.7,  (1, 5): 0.67, 
                      (2, 4): 0.45, (2, 5): 0.58, (4, 5): 0.9}
gender_example = [2,1,1,2,2,1]
edges_test_example = [(0, 3), (1,4), (1, 5), (2, 3), (2, 5), (3, 4)]
print(eval_link_prediction(nodes_to_eval_example, edge_score_example, gender_example, edges_test_example, 1))
print(eval_link_prediction(nodes_to_eval_example, edge_score_example, gender_example, edges_test_example, 2))

for algo in ['adamic-adar', 'jaccard']:
    print(f'Evaluating link prediction on {algo}')
    for k in [5, 10, 20]:
        edge_score = edge_score_ada if algo == 'adamic-adar' else edge_score_jaccard
        precision, precision_gender1, precision_gender2 = eval_link_prediction(nodes_to_eval, edge_score, caltech_gender, edges_test, k)
        print(f'precision@{k}: {precision:.3f}, precision@{k} for gender1: {precision_gender1: .3f}, precision@{k} for gender2: {precision_gender2:.3f}, precision@{k} diff: {precision_gender1-precision_gender2: .3f}')
    print()
```