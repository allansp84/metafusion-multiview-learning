# -*- coding: utf-8 -*-

import argparse

from antispoofing.mcnns.utils import *
from antispoofing.mcnns.datasets import *
from antispoofing.mcnns.controller import *
from antispoofing.mcnns.classification import *
from antispoofing.mcnns.metaclassification import *


class CommandLineParser(object):
    def __init__(self):

        # -- define the arguments available in the command line execution
        self.parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    def parsing(self):
        dataset_options = 'Available dataset interfaces: '
        for k in sorted(registered_datasets.keys()):
            dataset_options += ('%s-%s, ' % (k, registered_datasets[k].__name__))

        ml_algo_options = 'Available Algorithm for Classification: '
        for k in sorted(ml_algo.keys()):
            ml_algo_options += ('%s-%s, ' % (k, ml_algo[k].__name__))

        meta_ml_algo_options = 'Available Algorithm for Meta Classification: '
        for k in sorted(ml_algo.keys()):
            meta_ml_algo_options += ('%s-%s, ' % (k, meta_ml_algo[k].__name__))

        losses_functions_options = 'Available Algorithm for Losses: '
        for k in sorted(losses_functions.keys()):
            losses_functions_options += ('%s-%s, ' % (k, losses_functions[k]))

        optimizer_methods_options = 'Available Optimizers: '
        for k in sorted(optimizer_methods.keys()):
            optimizer_methods_options += ('%s-%s, ' % (k, optimizer_methods[k]))

        # -- arguments related to the dataset and to the output
        group_a = self.parser.add_argument_group('Arguments')

        group_a.add_argument('--protocol', type=str, metavar='', default='intra-test', choices=['intra-dataset', 'inter-dataset'],
                             help='(default=%(default)s).')

        group_a.add_argument('--dataset', type=int, metavar='', default=0, choices=range(len(registered_datasets)),
                             help=dataset_options + '(default=%(default)s).')

        group_a.add_argument('--dataset_b', type=int, metavar='', default=0, choices=range(len(registered_datasets)),
                             help=dataset_options + '(default=%(default)s).')

        group_a.add_argument('--dataset_path', type=str, metavar='', default='',
                             help='Path to the dataset.')

        group_a.add_argument('--dataset_path_b', type=str, metavar='', default='',
                             help='Path to the dataset.')

        group_a.add_argument('--output_path', type=str, metavar='', default='working',
                             help='Path where the results will be saved (default=%(default)s).')

        group_a.add_argument('--ground_truth_path', type=str, metavar='', default='',
                             help='A .csv file containing the ground-truth (default=%(default)s).')

        group_a.add_argument('--ground_truth_path_b', type=str, metavar='', default='',
                             help='A .csv file containing the ground-truth for the second dataset (default=%(default)s).')

        group_a.add_argument('--permutation_path', type=str, metavar='', default='',
                             help='A .csv file containing the data divided in two sets, training and testing sets (default=%(default)s).')

        group_a.add_argument('--iris_location', type=str, metavar='', default='extra/irislocation_osiris.csv',
                             help='A .csv file containing the irises locations (default=%(default)s).')

        group_a.add_argument('--iris_location_b', type=str, metavar='', default='extra/irislocation_osiris.csv',
                             help='A .csv file containing the irises locations for the dataset b (default=%(default)s).')

        # -- arguments related to the Feature extraction module
        group_b = self.parser.add_argument_group('Available Parameters for Feature Extraction')

        group_b.add_argument('--feature_extraction', action='store_true',
                             help='Execute the feature extraction process (default=%(default)s).')

        group_b.add_argument("--descriptor", type=str, default="RawImage", metavar="",
                             choices=['RawImage', 'bsif'],
                             help="(default=%(default)s)")

        group_b.add_argument("--desc_params", type=str, default="", metavar="",
                             help="Additional parameters for feature extraction.")

        # -- arguments related to the Classification module
        group_c = self.parser.add_argument_group('Available Parameters for Classification')

        group_c.add_argument('--classification', action='store_true',
                             help='Execute the classification process (default=%(default)s).')

        group_c.add_argument('--ml_algo', type=int, metavar='', default=0, choices=range(len(ml_algo)),
                             help=ml_algo_options + '(default=%(default)s).')

        group_c.add_argument('--epochs', type=int, metavar='', default=300,
                             help='Number of the epochs considered during the learning stage (default=%(default)s).')

        group_c.add_argument('--bs', type=int, metavar='', default=32,
                             help='The size of the batches (default=%(default)s).')

        group_c.add_argument('--lr', type=float, metavar='', default=0.01,
                             help='The learning rate considered during the learning stage (default=%(default)s).')

        group_c.add_argument('--decay', type=float, metavar='', default=0.0,
                             help='The decay value considered during the learning stage (default=%(default)s).')

        group_c.add_argument('--reg', type=float, metavar='', default=0.1,
                             help='The value of the L2 regularization method (default=%(default)s).')

        group_c.add_argument('--loss_function', type=int, metavar='', default=0, choices=range(len(losses_functions_options)),
                             help=losses_functions_options + '(default=%(default)s).')

        group_c.add_argument('--optimizer', type=int, metavar='', default=0, choices=range(len(optimizer_methods)),
                             help=optimizer_methods_options + '(default=%(default)s).')

        group_c.add_argument('--fold', type=int, metavar='', default=0,
                             help='(default=%(default)s).')

        group_c.add_argument('--n_run', type=int, metavar='', default=0,
                             help='(default=%(default)s).')

        group_c.add_argument('--force_train', action='store_true',
                             help='(default=%(default)s).')

        group_c.add_argument('--meta_ml_algo', type=int, metavar='', default=0, choices=range(len(meta_ml_algo)),
                             help=meta_ml_algo_options + '(default=%(default)s).')

        group_c.add_argument('--meta_classification', action='store_true',
                             help='(default=%(default)s).')

        group_c.add_argument('--compute_feature_importance', action='store_true',
                             help='(default=%(default)s).')

        group_c.add_argument('--meta_classification_from', type=str, metavar='', default='scores', choices=['scores', 'labels'],
                             help='(default=%(default)s).')

        group_c.add_argument('--n_models', type=int, metavar='', default=21,
                             help='(default=%(default)s).')

        group_c.add_argument('--selection_algo', type=int, metavar='', default=0,
                             help='(default=%(default)s).')

        group_d = self.parser.add_argument_group('Other options')

        group_d.add_argument('--operation', type=str, metavar='', default='crop', choices=['none', 'crop', 'segment'],
                             help='(default=%(default)s).')

        group_d.add_argument('--max_axis', type=int, metavar='', default=260,
                             help='(default=%(default)s).')

        group_d.add_argument('--device_number', type=str, metavar='', default='0',
                             help='(default=%(default)s).')

        group_d.add_argument('--show_results', action='store_true',
                             help='(default=%(default)s).')

        self.parser.add_argument('--n_jobs', type=int, metavar='int', default=N_JOBS,
                                 help='Number of jobs to be used during processing (default=%(default)s).')

        deprecated = self.parser.add_argument_group('Deprecated arguments')

        deprecated.add_argument('--last_layer', type=str, metavar='', default='linear', choices=['linear', 'softmax'],
                                help='(default=%(default)s).')

        deprecated.add_argument('--layers_name', nargs='+', type=str, metavar='', default=['conv_1'],
                                help='(default=%(default)s).')

        deprecated.add_argument('--fv', action='store_true',
                                help='(default=%(default)s).')

    def get_args(self):
        return self.parser.parse_args()


# -- main function
def main():

    # -- parsing the command line options
    command_line = CommandLineParser()
    command_line.parsing()
    args = command_line.get_args()
    print('ARGS:', args)
    sys.stdout.flush()

    # -- create and execute a Controller object
    control = Controller(args)
    control.run()


if __name__ == "__main__":
    start = time.time()

    main()

    elapsed = (time.time() - start)
    print('Total time elapsed: {0}!'.format(time.strftime("%d days, and %Hh:%Mm:%Ss", time.gmtime(elapsed))))
    sys.stdout.flush()
