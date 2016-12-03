import math
import networkx
import pickle


class SpatialGraphGenerator():
    def _distance(self, x1, y1, x2, y2):
        """
        Calculate a distance between 2 points
        from whose longitude and latitude
        """
        A = 6378137.0
        B = 6356752.314140
        dy = math.radians(y1 - y2)
        dx = math.radians(x1 - x2)

        if(dx < -math.pi):
            dx += 2*math.pi
        if(dx > math.pi):
            dx -= 2*math.pi

        my = math.radians((y1 + y2) / 2)
        E2 = (A**2 - B**2) / A**2
        Mnum = A * (1 - E2)
        w = math.sqrt(1 - E2 * math.sin(my)**2)
        m = Mnum / w**3
        n = A / w
        return math.sqrt((dy * m)**2 + (dx * n * math.cos(my))**2)

    def generate_graph(self, languages, pickle_file_name):
        """
        Create spatial neighbor graph
        """

        empty_vectors = {}
        for i, language in enumerate(languages):
            empty_vectors[i] = []
        spatial_graph = networkx.Graph(empty_vectors)

        for i, language1 in enumerate(languages):
            for j, language2 in enumerate(languages):
                if language1['id'] == language2['id']:
                    continue
                distance = self._distance(language1['longitude'],
                                          language1['latitude'],
                                          language2['longitude'],
                                          language2['latitude'])
                # in 1000km
                DISTANCE_THRESHOLD = 1000000
                if distance <= DISTANCE_THRESHOLD:
                    w = 1.0-distance/DISTANCE_THRESHOLD+1.0e-6
                    spatial_graph.add_edge(i,
                                           j,
                                           weight=w)

        with open(pickle_file_name, 'wb') as f:
            pickle.dump(spatial_graph, f)

        return spatial_graph
