import os
import functools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.traversal.depth_first_search import dfs_tree



def leaves(graph):
    """
    Returns the terminal leaves of a tree

    Parameters
    ----------
    graph : nx.DiGraph()
        tree (directed acyclic graph)

    Returns
    -------
    list : list of node objects
        list of node labels of leaves

    """
    return [node for node in graph.nodes() if graph.out_degree(node) == 0]


def root(graph):
    """
    Returns the root of a tree
    
    Parameters
    ----------
    graph : nx.DiGrap()
        tree (directed acyclic graph)

    Returns
    -------
    n : node object
        label of the root of graph

    """
    for n in graph.nodes():
        if graph.in_degree(n) == 0:
            return n


def subtree_size(graph, node):
    """
    Returns the size of the subtree of graph with given root node

    Parameters
    ----------
    graph : nx.DiGraph()
        tree
    node : root of the subtree

    Returns
    -------
    int
        Size of the subtree

    """
    return len(dfs_tree(graph,node).nodes())

def imbalance(graph):
    """
    Return the list of imbalance data for a tree

    Parameters
    ----------
    graph : nx.DiGrap()
        tree

    Returns
    -------
    distrib : list of 2-uples
        list for all non-leave nodes of graph of tuples (m,n) with
            m : number of descendants of the node
            n : number of node of the largest descendant branch

    """
    inner_nodes=[n for n in graph.nodes() if len(list(graph.neighbors(n))) > 1]
    distrib = []
    for n in inner_nodes:
        descendants_nb = subtree_size(graph, n)
        largest_subtree = max([subtree_size(graph, nn) for nn in graph.neighbors(n)])
        distrib.append((descendants_nb-1, largest_subtree))
    return distrib

def imbalance_root(graph):
    '''
    Returns the root imbalance proportion of a tree, defined as the ration between 
    the total number of offspring on the root and the number of nodes in the largest
    root branch
    '''
    r = root(graph)
    descendants_nb = subtree_size(graph, r) -1
    largest_subtree = max([subtree_size(graph, nn) for nn in graph.neighbors(r)])
    return largest_subtree/descendants_nb

def random_digraph_uniform(n):
    """
    Returns a random rooted tree drawn with uniform probability over the set of rooted tree of size n

    Parameters
    ----------
    n : int
        number of nodes of the tree

    Returns
    -------
    G : nx.DiGraph()
        random tree

    """
    G = nx.DiGraph()
    G.add_node(0)
    for k in range(1, n):
        root = np.random.choice(list(G.nodes()))
        G.add_node(k)
        G.add_edge(root, k)
    nx.set_node_attributes(G, {k:f'{k}' for k in range(n)}, 'schmilblick')
    return G

def generate_tree(lda, mu, Nact, Ninact):
    """
    Generate a tree (arbre réel) according to birth death model.

    Parameters
    ----------
    lda : float
        birth rate of new node per node per iteration
    mu : float
        death rate of nodes per node per per iteration
    Nact : int
        number of iterations of the active reproduction phase
    Ninact : int
        number of iterations of the pure death phase (lda is set to 0)

    Returns
    -------
    G : nx.DiGraph()
        networkx graph object of the generated tree with following node attributes:
            'state' : boolean, True if node living at the end of simulation
            'birth_time' : int
            'death_time' : int

    """
    currentID = 0
    G = nx.DiGraph()
    G.add_node(currentID)
    living = {0:True}

    birth_time = {0:0}
    death_time = {}

    pop = 1
    prob_birth = lda
    prob_death = mu

    for t in range(Nact):
        for current_node in list(G.nodes()):
            r = np.random.rand()
            if r < prob_birth and living[current_node]:
                currentID += 1
                G.add_node(currentID)
                G.add_edge(current_node, currentID)
                living[currentID] = True
                pop += 1
                birth_time[currentID] = t
            if prob_birth < r and r < (prob_birth + prob_death) and living[current_node]:
                living[current_node] =  False
                pop -= 1
                death_time[current_node] = t
        if pop == 0:
            break

    for t in range(Ninact):
        for current_node in list(G.nodes()):
            r = np.random.rand()
            if r <  prob_death and living[current_node]:
                living[current_node] =  False
                pop -= 1
                death_time[current_node] = t + Nact
            if pop == 0:
                break

    nx.set_node_attributes(G, living, 'state')
    nx.set_node_attributes(G, birth_time, 'birth_time')
    nx.set_node_attributes(G, death_time, 'death_time')
    return G

def generate_stemma(GG):
    """
    Returns the stemma of a tradition generated from generate_tree

    Parameters
    ----------
    GG : nx.DiGraph()
        tree object with at least node attributes 'state' given

    Returns
    -------
    G : nx.DiGraph()
        stemma obtained from GG.

    """
    G = nx.DiGraph(GG)
    living = {n: G.nodes[n]['state'] for n in list(G.nodes())}
    
    # recursivelly remove dead leaves until all terminal nodes are living witnesses
    
    terminal_dead_nodes = [n for n in leaves(G) if not living[n]]
    while terminal_dead_nodes != []:
        for n in terminal_dead_nodes:
            G.remove_node(n)
        terminal_dead_nodes = [n for n in leaves(G) if not living[n]]

    # remove non-branching consecutive dead nodes
    
    unwanted_virtual_nodes = [n for n in list(G.nodes()) if living[n] == False and G.out_degree(n) == 1 and G.in_degree(n) == 1]
    while unwanted_virtual_nodes != []:
        for n in unwanted_virtual_nodes:
            G.add_edge(list(G.predecessors(n))[0], list(G.neighbors(n))[0])
            G.remove_node(n)
        unwanted_virtual_nodes = [n for n in list(G.nodes()) if living[n] == False and G.in_degree(n) == 1 and G.out_degree(n) ==1]

    if not living[root(G)] and G.out_degree(root(G)) == 1:
        G.remove_node(root(G))
        
    return G

def draw_tree(G, filename):
    """
    Draw a tree generated from generate_tree or generate_stemma as svg file 
    with living nodes colored in red and grey dead nodes

    Parameters
    ----------
    G : nx.DiGraph()
        tree object with at least 'state' attribute given
    filename : string
        

    Returns
    -------
    None.

    """
    living = nx.get_node_attributes(G, 'state')
    color_map = {node: 'red' if state else 'grey' for node, state in zip(living.keys(), living.values())}
    nx.set_node_attributes(G, color_map, name='color')
    nx.nx_pydot.write_dot(G, 'graph.dot')
    os.system('dot -Tsvg graph.dot > {}.svg'.format(filename))
    
def csv_dump(G, filename):
    """
    Generate a csv file representing a tree obtained from generate_tree with layout
                
                label, parent, birth_time, death_time
                
                parent is -1 if node is root
                death_time is -1 if node living at the end of simulation
    """
    out = 'label,parent,birth_time,death_time\n'
    for n in G.nodes():
        bt = G.nodes[n]['birth_time']
        dt = G.nodes[n]['death_time'] if not G.nodes['state'] else -1
        pred = G.predecessors(n)
        par = pred[0] if pred != [] else -1
        out.append(f'{n},{par},{bt},{dt}\n')
    with open(f'{filename}.csv', 'w') as f:
        f.write(out)
        
def plot_heatmap(data, title, xticks, yticks, precision):
    '''
    Plots 2D phase diagrams with printed values

    Parameters
    ----------
    data : 2D array
        data to be plotted
    title : string
        title of the plot
    xticks : 1D float array
        list of ticks values displayed on x-axis
    yticks : 1D float array
        list of ticks values displayed on y-axis
    precision : int
        Nb of significant digits printed on plots
        
    Returns
    -------
    None.
    '''
    fig, ax = plt.subplots()
    im = ax.imshow(data, interpolation='nearest', cmap='viridis')
    for i in range(len(yticks)):
        for j in range(len(xticks)):
            text = ax.text(j, i, f'%.{precision}f'%data[i][j],
                           ha="center", va="center", color="w")
            
    ax.set_xticks(np.arange(len(xticks)), labels=xticks)
    ax.set_yticks(np.arange(len(yticks)), labels=yticks)
    
    ax.set_xlabel(r'$\lambda ~~~~ (10^{-3})$')
    ax.set_ylabel(r'$\mu ~~~~ (10^{-3})$')
     
    plt.colorbar(im)
    ax.set_title(title)
    plt.show()
    
def bootstraped(estimator):
    '''
    Decorator function returning the bootstraped mean and 10% confidence interval of an estimator

    Parameters
    ----------
    estimator : function(1D-array) -> float

    Returns
    -------
    (float,float,float)
        (bootstrap mean, 5th centile of bootstrap distribution , 95th centile)

    '''
    @functools.wraps(estimator)
    def wrapper_bootstraped(dataset):
        sample_size  = len(dataset)
        sample_nb = 100
        
        list_estimator = []
        
        for k in range(sample_nb):
            sample = np.random.choice(np.asarray(dataset, dtype ="object"), sample_size, replace = True)
            list_estimator.append(estimator(sample))
            
        estimated_beta = np.mean(list_estimator)
        lb = np.quantile(list_estimator, np.linspace(0,1,20))[0]
        ub = np.quantile(list_estimator, np.linspace(0,1,20))[19]
        return (estimated_beta, lb, ub)
    return wrapper_bootstraped
