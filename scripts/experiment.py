import argparse
import os
import pickle
import csv
import json
from collections import defaultdict
from collections import OrderedDict
from itertools import zip_longest

from spatial_graph_generator import SpatialGraphGenerator
from phylogenetic_graph_generator import PhylogeneticGraphGenerator
from autologistic import Autologistic
import numpy


class Experiment(object):
    """Experiment

    Args:
        language_file_path (str): File path for the WALS language csv file
        sp_graph_file_path (str): File path for the spatial graph
        ph_graph_file_path (str): File path for the phylogenetic graph
        output_dir (str): Directory path for output

    Attributes:
        languages (array):
        phylogenetic_graph ():
        spatial_graph ():
        output_dir (str): Directory path for output

    """
    def __init__(self, language_file_path, **kwargs):
        sp_graph_file_path = kwargs['sp_graph_file_path']
        ph_graph_file_path = kwargs['ph_graph_file_path']

        output_dir = kwargs['output_dir']
        output_dir = os.path.join(output_dir, 'results')
        self.output_dir = output_dir

        self.languages = []
        self.__feature_values = defaultdict(set)
        self.__feature_value_map = {}
        self.__int_feature_map = {}

        self.phylogenetic_graph = None
        self.spatial_graph = None

        self.__init_data(
            language_file_path,
            sp_graph_file_path,
            ph_graph_file_path
        )

    def __init_data(self, language_file_path,
                    sp_graph_file_path, ph_graph_file_path):
        self.__load_languages(language_file_path)
        self.__create_feature_value_map()

        # Create graphs
        if sp_graph_file_path:
            with open(sp_graph_file_path, 'rb') as f:
                self.spatial_graph = pickle.load(f)
        else:
            sp_graph_generator = SpatialGraphGenerator()
            self.spatial_graph = sp_graph_generator.generate_graph(
                self.languages, 'spacial_graph.pkl'
            )

        if ph_graph_file_path:
            with open(ph_graph_file_path, 'rb') as f:
                self.phylogenetic_graph = pickle.load(f)
        else:
            ph_graph_generator = PhylogeneticGraphGenerator()
            self.phylogenetic_graph = ph_graph_generator.generate_graph(
                self.languages, 'phylogenetic_graph.pkl'
            )

    def __load_languages(self, language_file_path):
        with open(language_file_path) as fw:
            reader = csv.reader(fw)
            header_line = next(reader)

            for l in reader:
                # exclude Sign languages
                if l[6] == 'Sign Languages':
                    continue

                language = {}

                language['id'] = l[0]
                language['name'] = l[3]
                language['longitude'] = float(l[4])
                language['latitude'] = float(l[5])
                language['ph_group'] = l[6]  # l[6]:subfamily l[7]:family

                language['features'] = {}

                for feature_idx in range(10, len(header_line)):
                    feature_name = header_line[feature_idx]
                    if not l[feature_idx] == '':
                        self.__feature_values[feature_name].add(l[feature_idx])
                    language['features'][feature_name] = 'NA' if l[feature_idx] == '' else l[feature_idx]

                self.languages.append(language)

    def __create_feature_value_map(self):
        """Create integer to feature raw value map
        """
        for feature_name, values in self.__feature_values.items():
            value_map = {}
            int_map = {}
            values = sorted(values)
            for i, value in enumerate(values):
                value_map[value] = i
                int_map[i] = value

            self.__feature_value_map[feature_name] = value_map
            self.__int_feature_map[feature_name] = int_map

    def print_target_features_list(self):
        """Print target features list
        """
        selected_features, xs = self.__select_target_features()
        print("Index\tFeature name")
        print('----------------------------------------------')
        for i, feature in enumerate(selected_features):
            print(str(i)+"\t"+feature)

    def execute(self, feature_idx_min, feature_idx_max,
                experiment_type, log=False):
        """Execute multiclass Autologistic model

        Args:
            feature_idx_min (int): Minimum index of the target features
            feature_idx_max (int): Maximum index of the target features
            experiment_type (str):
                If ``mvi``, it evaluate the performance of
                 missing value imputation by calculate 10-fold accuracies.
                If ``param``, it estimates parameters, :math:`\lambda`
                and :math:`\theta` and missing values
                by using all observed feature values
            log (bool):
                If ``True``, it outputs a log file of the experiment process
        """
        selected_features, xs = self.__select_target_features()

        for i, feature in enumerate(selected_features):
            if feature_idx_min > i:
                continue
            if feature_idx_max < i:
                break

            print('Feature:', feature, '(Index:', i, ')')
            max_feature_value = max(self.__feature_value_map[feature].values())

            if experiment_type == 'mvi':
                self.__hide_existing_values(
                    feature, xs[i], max_feature_value
                )
            elif experiment_type == 'param':
                model = Autologistic(
                    xs[i], self.spatial_graph,
                    self.phylogenetic_graph, max_feature_value
                )
                result = model.estimate_with_missing_value()
                output_dir_param = os.path.join(
                    self.output_dir, 'param'
                )
                if not os.path.exists(output_dir_param):
                    os.makedirs(output_dir_param)
                self.__output_result(
                    feature, xs[i], result,
                    output_dir_param, hidden_indexes=[]
                )
            else:
                raise ValueError('Experiment type must be param or mvi')

    def __select_target_features(self):
        selected_features = []
        features_list = sorted(list(self.__feature_value_map.keys()))
        xs = []

        for feature in features_list:
            # Create vector x
            estimate_targets = []
            x_ = []
            for idx, language in enumerate(self.languages):
                if language['features'][feature] == 'NA':
                    x_.append(-1)
                    estimate_targets.append(idx)
                else:
                    int_feature_value = self.__feature_value_map[feature][language['features'][feature]]
                    x_.append(int_feature_value)
            x = numpy.array(x_)
            filled_rate = len(numpy.where(x != -1)[0]) / len(x)

            if filled_rate > 0.2:
                selected_features.append(feature)
                xs.append(x)

        return (selected_features, xs)

    def __hide_existing_values(self, feature, x, max_value):
        """Execute a 10-fold cross validation experiment to
           evaluate model performance of missing value imputation
        """
        x_original = x.copy()
        not_missing_indexes = numpy.where(x_original != -1)[0]
        numpy.random.shuffle(not_missing_indexes)

        split_unit = len(not_missing_indexes)//10
        print('Split_unit', split_unit)

        cv_iter = zip_longest(*[iter(not_missing_indexes)]*split_unit)
        for cv_count, hidden_indexes in enumerate(cv_iter):
            x = x_original.copy()
            hidden_indexes = [y for y in hidden_indexes if y is not None]

            test_data = {}
            test_data['target_indexes'] = hidden_indexes
            test_data['answers'] = x[hidden_indexes].copy()
            x[hidden_indexes] = -1

            output_dir_mvi = os.path.join(
                self.output_dir,
                'mvi', "{0:03d}".format(cv_count)
            )
            if not os.path.exists(output_dir_mvi):
                os.makedirs(output_dir_mvi)
            model = Autologistic(
                x, self.spatial_graph, self.phylogenetic_graph,
                max_value
            )
            result = model.estimate_with_missing_value(test_data)

            self.__output_result(feature, x_original, result,
                                 output_dir_mvi, hidden_indexes)

    def __output_result(self, feature, x_original, result,
                        output_dir, hidden_indexes=[]):
        """Output experiment results as a JSON format
        """
        accuracy_exp = False
        output_json = OrderedDict()
        if len(result) == 6:
            x_estimated, p_theta, p_lambda, p_beta, \
                accuracy_proposed, accuracy_baseline = result
            accuracy_exp = True
        else:
            x_estimated, p_theta, p_lambda, p_beta = result

        estimate_targets = numpy.where(x_original == -1)[0]

        # Generate output json
        output_json['feature'] = feature
        output_json['lambda'] = p_lambda
        output_json['theta'] = p_theta
        output_json['beta'] = p_beta.tolist()
        if accuracy_exp:
            output_json['accuracy'] = {}
            output_json['accuracy']['proposed'] = accuracy_proposed
            output_json['accuracy']['baseline'] = accuracy_baseline

        output_json['estimate_results'] = []
        for language_idx, int_value in enumerate(x_estimated):
            hidden_estimated = language_idx in hidden_indexes
            estimated = language_idx in estimate_targets

            estimate_result = {}
            estimate_result['language'] = self.languages[language_idx]['name']
            estimate_result['is_hidden_estimated'] = hidden_estimated
            if hidden_estimated:
                estimate_result['value'] = self.__int_feature_map[feature][x_original[language_idx]]
                estimate_result['original'] = self.__int_feature_map[feature][int_value]
            else:
                estimate_result['value'] = self.__int_feature_map[feature][int_value]
                estimate_result['is_original'] = estimated
            output_json['estimate_results'].append(estimate_result)

        output_file_name = '_'.join(feature.split(' ')) + '.json'
        output_file_path = os.path.join(output_dir, output_file_name)
        with open(output_file_path, 'w') as f:
            json.dump(
                output_json,
                f,
                ensure_ascii=False,
                indent=4, separators=(',', ': ')
            )


def _parse_args():
    parser = argparse.ArgumentParser(
        description='arguments for experiment',
        formatter_class=argparse.RawTextHelpFormatter
    )
    experiment_type_description = """\
Experiment type
- mvi   Evaluate the performance of the autologistic model
        to estimate missing value
        (calculate the accuracies of 10-fold closs validation)
- param Estimate lambda and theta
    """
    sub_parsers = parser.add_subparsers(
        help='Commands', title='commands')

    parser_exp = sub_parsers.add_parser('exp', help='Execute experiments')
    parser_pf = sub_parsers.add_parser('print-features',
                                       help='Print features to process'\
                                       'with their indices')

    DEFAULT_LANGUAGE_PATH = './data/language.csv'

    # Execute experiments
    parser_exp.add_argument('min_idx', type=int,
                            help='Minimum index of the target features')
    parser_exp.add_argument('max_idx', type=int,
                            help='Maximum index of the target features')
    parser_exp.add_argument('experiment_type', type=str,
                            choices=['param', 'mvi'],
                            help=experiment_type_description)
    parser_exp.add_argument('-l', dest='language_file_path',
                            help='Path for language file (WALS csv)',
                            default=DEFAULT_LANGUAGE_PATH)
    parser_exp.add_argument('-s', dest='spg_filename',
                            help='Path for apatical graph file', default=None)
    parser_exp.add_argument('-p', dest='phg_filename',
                            help='Path for phylogenetic graph file',
                            default=None)
    parser_exp.add_argument('-o', dest='output_dir',
                            help='Directory for output files', default='')
    parser_exp.set_defaults(func=execute_experiments)

    # Print features
    parser_pf.add_argument('-l', dest='language_file_path',
                           help='Path for language file (WALS csv)',
                           default=DEFAULT_LANGUAGE_PATH)
    parser_pf.set_defaults(func=print_target_features_list)

    args = parser.parse_args()
    return args


def execute_experiments(args):
    experiment = Experiment(args.language_file_path,
                            sp_graph_file_path=args.spg_filename,
                            ph_graph_file_path=args.phg_filename,
                            output_dir=args.output_dir)
    experiment.execute(args.min_idx, args.max_idx,
                       args.experiment_type)


def print_target_features_list(args):
    experiment = Experiment(args.language_file_path,
                            sp_graph_file_path=None,
                            ph_graph_file_path=None,
                            output_dir='')
    experiment.print_target_features_list()


if __name__ == '__main__':
    args = _parse_args()
    args.func(args)
