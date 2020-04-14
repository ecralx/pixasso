import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import hdbscan
from functools import wraps

def fitted(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self._fitted:
            return func(self, *args, **kwargs)
        else:
            raise RuntimeError("Model not fitted yet")
    return wrapper


class Palette:
    """Colors palette generation based on HDBSCAN clustering of an image
    colors and tree exploration

    Args:
        n_colors (int): fixed number of colors for palette
        min_cluster_size (int): smallest amount of pixel to create a cluster
        **kwargs (dict): additional hdbscan.HDBSCAN keyed args

    Attributes:
        values (numpy.ndarray): (n_colors, 3) array for colors RGB values
        nodes_idx (list[int]): corresponding nodes in clustering graph
        hdbscan (hdbscan.HDBSCAN): hierarchical density clustering object
        _fitted (bool): True if model is fitted
    """
    N_PIXELS = 'n_pixels'
    COLOR_SUM = 'color_sum'

    def __init__(self, n_colors, min_cluster_size=15, **kwargs):
        self._n_colors = n_colors
        self.values = np.zeros((n_colors, 3))
        self.nodes_idx = [None] * n_colors
        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                       gen_min_span_tree=True,
                                       **kwargs)
        self._fitted = False

    @fitted
    def __getitem__(self, idx):
        """Returns palette color as dictionnary

            {'color': np.array([0.3, 0.2, 0.1]),       # color value in RGB coordinates
             'node': 78600,                            # index of corresponding node in self.g
             'n_pixels': 34,                           # number of pixel in corresponding cluster
             'color_sum': np.array([10.2, 6.8, 3.4]),  # cumulated pixel values in cluster
             }

        Args:
            idx (int)
        """
        value = self.values[idx]
        node = self.nodes_idx[idx]
        if node:
            node_attributes = self.g.nodes[node]
            output = {'color': value, 'node': node, **node_attributes}
            return output
        else:
            raise IndexError("Undefined color in palette")

    def __len__(self):
        return self._n_colors

    def __repr__(self):
        output = '\n'.join(["Color Palette",
                            f"Nb of colors : {len(self)}",
                            f"Color values {self.values}"])
        return output

    def display_colors(self):
        """Displays palette colors
        """
        fig, ax = plt.subplots(1, len(self))
        for i, a in enumerate(ax):
            a.imshow(np.tile(self.values[i], 25).reshape(5, 5, 3))
            a.axis('off')
        plt.show()

    def fit(self, image):
        """Fits hdbscan on image and derives filled colors clustering tree

        Args:
            image (numpy.ndarray): (W, H, 3)
        """
        # Flatten image to (W*H, 3) size to perform clustering on colors only
        img_colors = image.reshape(-1, 3)

        # Fit hierarchical density based clustering model
        self.hdbscan.fit(img_colors)

        # Setup corresponding clutering graph and dataframe
        self._setup_dataframes()
        leaves = self._setup_networkx()
        self._init_leaves_attributes(img_colors, leaves)
        self._flow_back_attributes(leaves)
        self._fitted = True
        self._populate_values(img_colors)

    def _setup_dataframes(self):
        """Creates dataframes based on clustering tree

        single_points_df : dataframe of isolated pixels (non belonging to any cluster)
        clusters_df : dataframe of clusters relationships

        in single_points_df, the child index is the index in the dataset of that pixel

        Details on dataframe format at https://hdbscan.readthedocs.io/en/latest/advanced_hdbscan.html
        """
        tree_df = self.hdbscan.condensed_tree_.to_pandas()
        self.single_points_df = tree_df.sort_values(by='lambda_val')[tree_df.child_size == 1]
        self.clusters_df = tree_df.sort_values(by='lambda_val')[tree_df.child_size > 1]

    def _setup_networkx(self):
        """Creates DiGraph based on clustering tree and isolates leaves indices

        Details on graph format at https://hdbscan.readthedocs.io/en/latest/advanced_hdbscan.html
        """
        # Initialize corresponding DiGraph
        self.g = nx.from_pandas_edgelist(self.clusters_df,
                                         source='parent',
                                         target='child',
                                         create_using=nx.DiGraph())

        # Select nodes being unique to children or parent sets
        all_parents = set(self.clusters_df.parent.values)
        all_child = set(self.clusters_df.child.values)
        leaves = all_parents.symmetric_difference(all_child)

        # Remove root to obtain nodes being children but never parents i.e. leaves
        root = int(self.clusters_df.iloc[0].parent)
        leaves.remove(root)
        return leaves

    def _init_leaves_attributes(self, img_colors, leaves):
        """Records number of pixels and their cummulated RGB values for each
        leaf cluster

        Args:
            img_colors (np.ndarray): (H*W, 3) flattened array of colors
            leaves (set[int]): indices of graph leaves
        """
        for leaf in leaves:
            # Select pixels of the leaf cluster
            leaf_pixels = self.single_points_df[self.single_points_df.parent == leaf].child.values
            pixels_values = img_colors[leaf_pixels]

            # Compute attributes
            n_pixels = len(leaf_pixels)
            sum_pixels = np.sum(pixels_values, axis=0)

            # Set attributes
            self.g.nodes[leaf][self.N_PIXELS] = n_pixels
            self.g.nodes[leaf][self.COLOR_SUM] = sum_pixels

    def _flow_back_attributes(self, leaves):
        """Transmits child attributes upward to its parents clusters

        Args:
            leaves (set[int]): indices of graph leaves
        """
        for leaf in leaves:
            # Retrieve ancestors
            ancestors = nx.ancestors(self.g, leaf)

            # Flow back leaf attributes to ancestors
            for ancester in ancestors:
                if not self.g.nodes[ancester]:
                    self.g.nodes[ancester][self.N_PIXELS] = 0
                    self.g.nodes[ancester][self.COLOR_SUM] = np.zeros(3)

                # Increment attributes with leaf values
                self.g.nodes[ancester][self.N_PIXELS] += self.g.nodes[leaf][self.N_PIXELS]
                self.g.nodes[ancester][self.COLOR_SUM] += self.g.nodes[leaf][self.COLOR_SUM]

    def _populate_values(self, img_colors):
        """Populates the palette values by traversing the graph. 
        TODO: expliquer comment ça marche
        """
        values = []
        nodes_idx = []
        root = [n for n,d in self.g.in_degree() if d == 0].pop(0)
        successors = self.g.successors(root)
        while len(nodes_idx) < self._n_colors:
            # First element
            element1 = next(successors, False) # avoid StopIteration
            if not element1: break
            element1_successors = self.g.successors(element1)
            
            child_11 = next(element1_successors, False)
            child_12 = next(element1_successors, False)

            if not child_11 or not child_12:
                distance1 = 0
            else:
                sum_color_11 = self.g.nodes[child_11][self.COLOR_SUM]
                sum_color_12 = self.g.nodes[child_12][self.COLOR_SUM]
                distance1 = np.linalg.norm(sum_color_11 - sum_color_12)
            
            # Second element
            element2 = next(successors, False)
            if not element2: break
            element2_successors = self.g.successors(element2)
            
            child_21 = next(element2_successors, False)
            child_22 = next(element2_successors, False)
            if not child_21 or not child_22:
                distance2 = 0
            else:
                sum_color_21 = self.g.nodes[child_21][self.COLOR_SUM]
                sum_color_22 = self.g.nodes[child_22][self.COLOR_SUM]
                distance2 = np.linalg.norm(sum_color_21 - sum_color_22)

            if distance1 > 0 or distance2 > 0:
                if distance1 < distance2: # we take second element children
                    # remove parent if exists
                    if element2 in nodes_idx:
                        nodes_idx.remove(element2)
                    nodes_idx.append(child_21)
                    nodes_idx.append(child_22)
                    successors = self.g.successors(element2)
                else: # we take first element children
                    if element1 in nodes_idx:
                        nodes_idx.remove(element1)
                    nodes_idx.append(child_11)
                    nodes_idx.append(child_12)
                    successors = self.g.successors(element1)
        
        for id in nodes_idx:
            node = self.g.nodes[id]
            values.append(node[self.COLOR_SUM] / node[self.N_PIXELS])
        
        self.values = np.array(values)
        self.nodes_idx = nodes_idx
        