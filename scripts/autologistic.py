import numpy
from collections import Counter


class Autologistic:
    """Execute multinomial Autologistic model

    Args:
        x (numpy vector):
            Vectors whose elements denote typological feature values
            of the target languages. If the element value is -1, which is
            regarded as a missing value
        spatial_graph, phylogenetical_graph (networkx graph):
            Neighbor graphs on spatial and phylogenetical
        max_value (int):
            Max value of the feature values
            corresponding to the number of feature types
    """
    def __init__(self, x, spatial_graph, phylogenetical_graph, max_value):
        self.__max_value = max_value
        self.__value_list = list(range(self.__max_value+1))

        self.__p_theta, self.__p_lambda = 0.0, 0.0

        # Initialize p_beta with the distribution probability
        # of each feature value
        value_dist = Counter()
        for i in numpy.where(x != -1)[0]:
            value_dist[x[i]] += 1
        _sum = sum(value_dist.values())

        value_dist = dict([(i, value_dist[i] / _sum+1) for i in range(self.__max_value+1)])
        self.__p_beta = numpy.array([numpy.log(value_dist[i]) for i in range(self.__max_value+1)])
        print("initial beta", self.__p_beta)

        self.__spatial_graph = spatial_graph
        self.__phylogenetical_graph = phylogenetical_graph

        self.__x = x.copy()
        self.__s_original = self.__neighbor_sum(self.__x, self.__spatial_graph)
        self.__t_original = self.__neighbor_sum(self.__x, self.__phylogenetical_graph)
        self.__u_original = numpy.array([self.__sum_vector(self.__x, i) for i in range(self.__max_value+1)])

    def estimate_with_missing_value(self, test_data=None):
        """
        Estimate parameters on incomplete data

        Args:
            test_data (numpy vector):
                Vector for checking the accuracy (hit rate)
                The accuracy (hit rate) is calculated
                by comparing estimated x and this

        Returns:
            tuple: estimated x and parameters

        """
        print('Initialize the missing values')
        missing_indexes = self.__init_missing_values()

        self.__s_original = self.__neighbor_sum(self.__x, self.__spatial_graph)
        self.__t_original = self.__neighbor_sum(self.__x, self.__phylogenetical_graph)
        self.__u_original = numpy.array([self.__sum_vector(self.__x, i) for i in range(self.__max_value+1)])

        initial_hit_rate = 0
        if test_data:
            difference = self.__x[test_data['target_indexes']] - test_data['answers']
            initial_hit_rate = len(numpy.where(difference == 0)[0]) / len(difference)
            print('Initial hit rate', initial_hit_rate, self.__x[test_data['target_indexes']], test_data['answers'])

        print('Start the estimation')

        # Estimate parameters
        self.estimate(missing_indexes, test_data)

        # Reestimate vector x on missing indices with calculated parameters
        sampled_xs = numpy.empty((0, len(self.__x)), int)
        sampled_xs = numpy.append(sampled_xs, numpy.array([self.__x]), axis=0)
        change_rate = 0
        ITER_REESTIMATE = 200
        BURN_IN = 50

        for i in range(ITER_REESTIMATE):
            new_x = self.__x.copy()
            for j in missing_indexes:
                probs = numpy.zeros(self.__max_value + 1)

                for k in range(self.__max_value + 1):
                    s = self.__sum_vector(sampled_xs[-1][self.__spatial_graph.neighbors(j)], k)
                    t = self.__sum_vector(sampled_xs[-1][self.__phylogenetical_graph.neighbors(j)], k)
                    probs[k] = numpy.exp(self.__p_beta[k] + self.__p_theta*s + self.__p_lambda*t)
                denominator_p = numpy.sum(probs)
                probs = probs / denominator_p

                new_x[j] = numpy.random.choice(self.__value_list, p=probs)

            sampled_xs = numpy.append(sampled_xs, numpy.array([new_x]), axis=0)

            if i >= BURN_IN:
                print('Change rate', (sampled_xs[-2] != sampled_xs[-1]).sum() / len(self.__x))
                change_rate += (sampled_xs[-2] != sampled_xs[-1]).sum() / len(self.__x)
        print()
        print('Change rate mean', change_rate/(ITER_REESTIMATE-BURN_IN))

        # Pickup most occured feature value in the sampling on each language
        for i in missing_indexes:
            sampled_values_dist = Counter(sampled_xs[BURN_IN:, i])
            print('Sampled value distribution', i, sampled_values_dist)
            self.__x[i] = sampled_values_dist.most_common(1)[0][0]

        # Output estimated parameters
        if test_data:
            hit_rate = (self.__x[test_data['target_indexes']] == test_data['answers']).sum() / len(test_data['answers'])
            print('Parameters:', self.__p_theta, self.__p_lambda, self.__p_beta)
            print('Hit rate:', hit_rate, len(test_data['answers']))

        print('Reestimated: ', self.__p_theta, self.__p_lambda, self.__p_beta)

        if test_data:
            return (self.__x, self.__p_theta, self.__p_lambda, self.__p_beta, hit_rate, initial_hit_rate)

        return (self.__x, self.__p_theta, self.__p_lambda, self.__p_beta)

    def estimate(self, missing_indexes=None, test_data=None):
        """
        Estimate parameters by applying autologistic model

        Args:
            missing_indexes (numpy vector):
                ``x``'s indexes of missing values
            test_data (numpy vector):
                Vector for checking the precision (hit rate)
                The precision (hit rate) is calculated
                by comparing estimated x and this (output to standard output)

        """
        sampled_x = numpy.zeros(len(self.__x))

        eta = 0.01  # Learning rate

        likelihood_sub = 10000.0
        fudge_rate = 1.0e-8
        iteration = 0
        initial_x = self.__x.copy()

        # Corresponds to Adam's beta
        tau_1 = 0.9
        tau_2 = 0.999
        m_theta = 0.0
        v_theta = 0.0
        m_lambda = 0.0
        v_lambda = 0.0
        m_beta = 0.0
        v_beta = 0.0

        iter_count = 0
        ITER_NUM = 10
        BURN_IN_NUM = 5
        SAMPLE_NUM = ITER_NUM - BURN_IN_NUM

        while numpy.fabs(likelihood_sub) > 0.01 and iter_count < 100:
            iter_count += 1

            dl_dtheta = 0
            dl_dlambda = 0
            dl_dbeta = numpy.array([0 for i in range(self.__max_value+1)])

            original_p_theta = self.__p_theta
            original_p_lambda = self.__p_lambda
            original_p_beta = self.__p_beta.copy()

            # Sample missing value on vector x
            if missing_indexes is None:
                dl_dtheta += self.__s_original * SAMPLE_NUM
                dl_dlambda += self.__t_original * SAMPLE_NUM
                dl_dbeta += self.__u_original * SAMPLE_NUM
            else:
                sampled_x = initial_x.copy()
                for j in range(ITER_NUM):
                    before_x = sampled_x

                    sampled_x = self.__sample_x(sampled_x,
                                                original_p_theta,
                                                original_p_lambda,
                                                original_p_beta,
                                                missing_indexes)
                    print('CRate',
                          (before_x != sampled_x).sum() / len(missing_indexes),
                          (before_x != sampled_x).sum(), len(missing_indexes),
                          j)
                    if BURN_IN_NUM > j:
                        continue

                    dl_dtheta += self.__neighbor_sum(sampled_x, self.__spatial_graph, missing_indexes)
                    dl_dlambda += self.__neighbor_sum(sampled_x, self.__phylogenetical_graph, missing_indexes)
                    dl_dbeta += [self.__sum_vector(sampled_x, i) for i in range(self.__max_value+1)]
                    sampled_final = sampled_x

            # Sample full elements on vector x
            sampled_x = initial_x.copy()
            for j in range(ITER_NUM):
                before_x = sampled_x

                sampled_x = self.__sample_x(sampled_x, original_p_theta, original_p_lambda, original_p_beta)
                print('CRate',
                      (before_x != sampled_x).sum() / len(self.__x),
                      (before_x != sampled_x).sum(), len(self.__x), j)

                if BURN_IN_NUM > j:
                    continue

                dl_dtheta -= self.__neighbor_sum(sampled_x, self.__spatial_graph)
                dl_dlambda -= self.__neighbor_sum(sampled_x, self.__phylogenetical_graph)
                dl_dbeta -= [self.__sum_vector(sampled_x, i) for i in range(self.__max_value+1)]

            initial_x = sampled_final

            # Update parameters (Using Adam)
            likelihood_before = self.__likelihood(sampled_final)

            dl_dtheta = dl_dtheta/SAMPLE_NUM
            dl_dlambda = dl_dlambda/SAMPLE_NUM
            dl_dbeta = dl_dbeta/SAMPLE_NUM

            m_theta = tau_1 * m_theta + (1 - tau_1) * dl_dtheta
            v_theta = tau_2 * v_theta + (1 - tau_2) * (dl_dtheta**2)
            m_lambda = tau_1 * m_lambda + (1 - tau_1) * dl_dlambda
            v_lambda = tau_2 * v_lambda + (1 - tau_2) * (dl_dlambda**2)
            m_beta = tau_1 * m_beta + (1 - tau_1) * dl_dbeta
            v_beta = tau_2 * v_beta + (1 - tau_2) * (dl_dbeta**2)

            update_targets = [
                (m_theta, v_theta),
                (m_lambda, v_lambda),
                (m_beta, v_beta)
            ]
            deltas = list()
            for m_t, v_t in update_targets:
                m_hat = m_t / (1 - tau_1 ** (iteration+1))
                v_hat = v_t / (1 - tau_2 ** (iteration+1))
                deltas.append((m_hat, v_hat))

            self.__p_theta += eta * deltas[0][0] / (numpy.sqrt(deltas[0][1]) + fudge_rate)
            self.__p_lambda += eta * deltas[1][0] / (numpy.sqrt(deltas[1][1]) + fudge_rate)
            self.__p_beta += eta * deltas[2][0] / (numpy.sqrt(deltas[2][1]) + fudge_rate)

            likelihood_after = self.__likelihood(sampled_final)
            likelihood_sub = likelihood_after - likelihood_before

            iteration += 1
            print(iteration, likelihood_sub, self.__p_theta, self.__p_lambda, self.__p_beta)

            if test_data is None:
                continue
            hit_rate = (sampled_final[test_data['target_indexes']] == test_data['answers']).sum() / len(test_data['answers'])
            print('Hit rate:', hit_rate)

    def __neighbor_sum(self, vector, neighbor_graph, target_indexes=None):
        concordants = 0

        for i, a in enumerate(vector):
            for j in neighbor_graph.neighbors(i):
                if i >= j:
                    continue
                if a == vector[j]:
                    concordants += neighbor_graph[i][j]['weight']

        return concordants

    def __sum_vector(self, vector, value):
        return (vector == value).sum()

    def __sample_x(self, x_init, p_theta, p_lambda, p_beta, missing_indexes=None):
        x_ret = x_init.copy()

        if missing_indexes is None:
            # Estimate full elements
            target_indexes = list(range(len(x_ret)))
        else:
            # If given missing_indexes, estimate missing elements only
            target_indexes = missing_indexes.copy()

        numpy.random.shuffle(target_indexes)
        for i in target_indexes:
            probs = numpy.zeros(self.__max_value+1)
            for j in range(self.__max_value+1):
                s = self.__sum_vector(x_ret[self.__spatial_graph.neighbors(i)], j)
                t = self.__sum_vector(x_ret[self.__phylogenetical_graph.neighbors(i)], j)
                probs[j] = numpy.exp(p_beta[j] + p_theta*s + p_lambda*t)

            denominator_p = numpy.sum(probs)
            probs = probs / denominator_p

            x_ret[i] = numpy.random.choice(self.__value_list, p=probs)
        return x_ret

    def __likelihood(self, x):
        s = self.__neighbor_sum(x, self.__spatial_graph)
        t = self.__neighbor_sum(x, self.__phylogenetical_graph)
        u = numpy.array([self.__sum_vector(x, i) for i in range(self.__max_value+1)])

        u_sum = sum([self.__p_beta[i]*u[i] for i in range(self.__max_value+1)])
        return u_sum + self.__p_theta*s + self.__p_lambda*t

    def __non_missing_probs(self, graph, i, missing_indexes):
        target_indexes = list(set(graph.neighbors(i)) - set(missing_indexes))
        if len(target_indexes) == 0:
            return None

        value_dist = Counter(self.__x[target_indexes])
        values = list(value_dist.keys())
        probs = numpy.array(list(value_dist.values()))
        return (probs / numpy.sum(probs), values)

    def __init_missing_values(self):
        value_dist = Counter()
        for i in numpy.where(self.__x != -1)[0]:
            value_dist[self.__x[i]] += 1
        values = list(value_dist.keys())
        init_probs = numpy.array(list(value_dist.values()))
        init_probs = init_probs / numpy.sum(init_probs)

        missing_indexes = numpy.where(self.__x == -1)[0]

        for i in missing_indexes:
            phylogenetical_probs = self.__non_missing_probs(self.__phylogenetical_graph, i, missing_indexes)
            spatial_probs = self.__non_missing_probs(self.__spatial_graph, i, missing_indexes)

            if phylogenetical_probs is None and spatial_probs is None:
                self.__x[i] = numpy.random.choice(values, p=init_probs)
            elif spatial_probs is None:
                self.__x[i] = numpy.random.choice(phylogenetical_probs[1], p=phylogenetical_probs[0])
            elif phylogenetical_probs is None:
                self.__x[i] = numpy.random.choice(spatial_probs[1], p=spatial_probs[0])
            else:
                phylogenetical_sampled = numpy.random.choice(phylogenetical_probs[1], p=phylogenetical_probs[0])
                spatial_sampled = numpy.random.choice(spatial_probs[1], p=spatial_probs[0])
                sampled_values = [phylogenetical_sampled, spatial_sampled]
                self.__x[i] = numpy.random.choice(sampled_values, p=[0.5, 0.5])
        return missing_indexes
