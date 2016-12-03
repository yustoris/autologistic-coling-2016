import networkx
import pickle


class PhylogeneticGraphGenerator():
    def generate_graph(self, languages, pickle_file_name):
        """
        Create phylogenetical neighbor graph
        """
        empty_vectors = {}
        for i, language in enumerate(languages):
            empty_vectors[i] = []

        phylogenetic_graph = networkx.Graph(empty_vectors)
        for i, language1 in enumerate(languages):
            for j, language2 in enumerate(languages):
                if language1['id'] == language2['id']:
                    continue

                if language1['ph_group'] == language2['ph_group'] and \
                   not language1['ph_group'] == 'NA':
                    phylogenetic_graph.add_edge(
                        i,
                        j,
                        weight=1.0
                    )

        with open(pickle_file_name, 'wb') as f:
            pickle.dump(phylogenetic_graph, f)

        return phylogenetic_graph
