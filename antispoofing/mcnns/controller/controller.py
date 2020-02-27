# -*- coding: utf-8 -*-

import json

from antispoofing.mcnns.utils import *
from antispoofing.mcnns.datasets import *
from antispoofing.mcnns.features import Extraction
from antispoofing.mcnns.classification import *
from antispoofing.mcnns.metaclassification import *


class Controller(object):
    """ This class implements the Controller responsible for managing the main steps of our pipeline and its input and output directories.

    Note:
        1. For now, only the classification module is being used.

    Args:
        args (ArgumentParser): Object that contains the options provided on the command line.

    Attributes:
        args (ArgumentParser): Contains the options provided by the user on the command line.
        data (Dataset): The dataset to be processed.
        data_b (Dataset): A secondary dataset to be used as testing in the cross-dataset protocol.
        n_jobs (int): Number of processor(s) considered for the parallel execution.

    Todo:
        1. Implement and test the Extraction module using BSIF (Andrey's branch).
    """

    def __init__(self, args):

        self.args = args
        self.data = None
        self.data_b = None
        self.n_jobs = self.args.n_jobs

        self.desc_params_path = ""
        self.features_path = ""
        self.classification_path = ""
        self.path_to_features = ""

    def feature_extraction(self):
        """ Method responsible for extracting the feature vectors of each image into the dataset.

        This method is responsible for creating and running the Extraction object, which takes as input the filename of the image
        to be processed and its respective output filename. In case the user opts for processing the images in parallel, this method will
        create a pool of Extraction object, one for each image, which will be processed in parallel using the package multiprocessing.
        """

        start = get_time()

        # -- get the list of images to be processed
        input_fnames = self.data.meta_info['all_fnames']

        # --  creates a output filename for each image in the dataset
        output_fnames = []
        for i in range(len(input_fnames)):
            rel_fname = os.path.relpath(input_fnames[i], self.data.dataset_path)
            # rel_fname = '{0}.npy'.format(rel_fname)
            output_fnames.append(os.path.join(self.path_to_features, rel_fname))
        output_fnames = np.array(output_fnames)

        # -- creates a pool of Extraction object
        tasks = []
        for idx in range(len(input_fnames)):
            tasks += [Extraction(output_fnames[idx], input_fnames[idx],
                                 descriptor=self.args.descriptor,
                                 params=self.args.desc_params,
                                 )]

        # -- start to execute the objects Extraction by running the method Extraction.run()
        if self.n_jobs > 1:
            print("running %d tasks in parallel" % len(tasks))
            RunInParallel(tasks, self.n_jobs).run()
        else:
            print("running %d tasks in sequence" % len(tasks))
            for idx in range(len(input_fnames)):
                tasks[idx].run()
                progressbar('-- RunInSequence', idx, len(input_fnames))

        elapsed = total_time_elapsed(start, get_time())
        print('spent time: {0}!'.format(elapsed))
        sys.stdout.flush()

    def classification(self):
        """ Method responsible for executing the classification step.

        This method is responsible for managing the classification step, creating and running the Classifier according to the input
        arguments provided on the command line by the user, such as classification algorithm and its parameters.
        """

        start = get_time()

        algo = ml_algo[self.args.ml_algo]

        output_fname = "max_axis-{}-epochs-{}-bs-{}-losses-{}-lr-{}-decay-{}-optimizer-{}-reg-{}-" \
                       "fold-{}".format(self.args.max_axis,
                                        self.args.epochs,
                                        self.args.bs,
                                        losses_functions[self.args.loss_function],
                                        self.args.lr,
                                        self.args.decay,
                                        optimizer_methods[self.args.optimizer],
                                        self.args.reg,
                                        self.args.fold,
                                        )

        output_path = os.path.join(self.data.output_path,
                                   self.args.descriptor,
                                   self.desc_params_path,
                                   self.classification_path,
                                   os.path.splitext(os.path.basename(self.args.permutation_path))[0],
                                   output_fname,
                                   )

        input_path = os.path.abspath(self.path_to_features)
        self.data.dataset_path = input_path
        # self.data.file_types = ['.tiff', '.png']

        if 'inter' in self.args.protocol:
            input_path = os.path.abspath(self.path_to_features_b)
            self.data_b.dataset_path = input_path
            # self.data_b.file_types = ['.tiff', '.png']
        else:
            self.data_b = None

        algo(output_path, self.data,
             dataset_b=self.data_b,
             input_shape=self.args.max_axis,
             epochs=self.args.epochs,
             batch_size=self.args.bs,
             loss_function=losses_functions[self.args.loss_function],
             lr=self.args.lr,
             decay=self.args.decay,
             optimizer=optimizer_methods[self.args.optimizer],
             regularization=self.args.reg,
             device_number=self.args.device_number,
             force_train=self.args.force_train,
             filter_vis=self.args.fv,
             layers_name=self.args.layers_name,
             fold=self.args.fold,
             ).run()

        elapsed = total_time_elapsed(start, get_time())
        print('spent time: {0}!'.format(elapsed))
        sys.stdout.flush()

    def show_results(self):
        """ Method responsible for showing the classification results.
        """

        start = get_time()

        output_fname = "max_axis-{}-epochs-{}-bs-{}-losses-{}-lr-{}-decay-{}-optimizer-{}-reg-{}-" \
                       "fold-{}".format(self.args.max_axis,
                                        self.args.epochs,
                                        self.args.bs,
                                        losses_functions[self.args.loss_function],
                                        self.args.lr,
                                        self.args.decay,
                                        optimizer_methods[self.args.optimizer],
                                        self.args.reg,
                                        self.args.fold,
                                        )

        input_path = os.path.join(self.data.output_path,
                                  self.args.descriptor,
                                  self.desc_params_path,
                                  self.classification_path,
                                  os.path.splitext(os.path.basename(self.args.permutation_path))[0],
                                  output_fname,
                                  )

        fnames = retrieve_fnames_by_type(os.path.abspath(input_path), '.json')
        fnames = [fname for fname in fnames if 'results.json' in fname]
        report = {}
        for fname in fnames:
            key = os.path.basename(os.path.dirname(fname))
            json_data = json.load(open(fname, 'r'))
            report[key] = json_data

        classification_results_summary(report)

        elapsed = total_time_elapsed(start, get_time())
        print('spent time: {0}!'.format(elapsed))
        sys.stdout.flush()

    def meta_classification(self):
        """ Method responsible for showing the classification results.
        """

        start = get_time()

        algo = meta_ml_algo[self.args.meta_ml_algo]

        output_path = os.path.join(self.data.output_path,
                                   self.args.round, 
                                   "{}".format(self.args.n_points),
                                   "meta_classification",
                                   "selection_algo-{}".format(self.args.selection_algo),
                                   self.args.meta_classification_from,
                                   str(algo.__name__).lower(),
                                   "n_models-{}".format(self.args.n_models),
                                   "n_run-{}".format(self.args.n_run),
                                   "fold-{}".format(self.args.fold),
                                   )

        input_path = os.path.abspath(os.path.join(self.data.output_path, self.args.round, "{}".format(self.args.n_points)))

        if 'scores' in self.args.meta_classification_from:
            predictions_files = retrieve_fnames_by_type(input_path, '.scores')
            gt_files = retrieve_fnames_by_type(input_path, '.labels')

            predictions_files = [fn for fn in predictions_files if 'svm' not in fn]
            gt_files = [fn for fn in gt_files if 'svm' not in fn]
            assert (np.chararray.replace(gt_files, '.labels', '') == np.chararray.replace(predictions_files, '.scores', '')).all(), 'AssertionError: Missing or non-ordered files'

        else:
            predictions_files = retrieve_fnames_by_type(input_path, '.predictions')
            gt_files = retrieve_fnames_by_type(input_path, '.labels')
            assert (np.chararray.replace(gt_files, '.labels', '') == np.chararray.replace(predictions_files, '.predictions', '')).all(), 'AssertionError: Missing or non-ordered files'

        # gt_files = np.array([fname for fname in gt_files if 'labels.txt' in fname])

        algo(output_path, predictions_files, gt_files,
             meta_classification_from=self.args.meta_classification_from,
             n_models=self.args.n_models,
             step_algo=self.args.step_algo,
             selection_algo=self.args.selection_algo,
             compute_feature_importance=self.args.compute_feature_importance,
             force_train=self.args.force_train,
             fold=self.args.fold,
             n_jobs=self.n_jobs,
             ).run()

        elapsed = total_time_elapsed(start, get_time())
        print('spent time: {0}!'.format(elapsed))
        sys.stdout.flush()

    def run(self):
        """ Method responsible for executing the steps of our pipeline.

        This is the main method of the class Controller responsible for executing the steps of our pipeline, according to the input
        arguments provided on the command line by the user.
        """

        # -- create an object for the main dataset
        dataset = registered_datasets[self.args.dataset]

        self.data = dataset(self.args.dataset_path,
                            ground_truth_path=self.args.ground_truth_path,
                            permutation_path=self.args.permutation_path,
                            iris_location=self.args.iris_location,
                            output_path=self.args.output_path,
                            operation=self.args.operation,
                            max_axis=self.args.max_axis,
                            )

        self.data.output_path = os.path.join(self.args.output_path,
                                             self.args.protocol,
                                             str(self.data.__class__.__name__).lower(),
                                             )

        if self.args.desc_params:
            self.desc_params_path = "{}x{}x{}".format(*eval(self.args.desc_params))
        else:
            self.desc_params_path = ""

        self.features_path = "features"
        self.classification_path = "classification"
        self.path_to_features = os.path.join(self.data.output_path, self.args.descriptor, self.desc_params_path, self.features_path)

        # -- create an object for the second dataset in case the user opts for running the cross-dataset protocol
        self.data_b = None
        if 'inter' in self.args.protocol:
            dataset_b = registered_datasets[self.args.dataset_b]
            self.data_b = dataset_b(self.args.dataset_path_b,
                                    ground_truth_path=self.args.ground_truth_path_b,
                                    permutation_path=self.args.permutation_path,
                                    iris_location=self.args.iris_location_b,
                                    output_path=self.args.output_path,
                                    operation=self.args.operation,
                                    max_axis=self.args.max_axis,
                                    )

            self.data_b.output_path = os.path.join(self.data.output_path,
                                                   str(self.data_b.__class__.__name__).lower(),
                                                   )

            self.features_path = "features"
            self.classification_path = "classification"
            self.path_to_features_b = os.path.join(self.data_b.output_path, self.args.descriptor, self.desc_params_path, self.features_path)

        if self.args.feature_extraction:
            print("-- extracting features ...")
            self.feature_extraction()

        if self.args.classification:
            print("-- classifying ...")
            self.classification()

        if self.args.meta_classification:
            print("-- meta classification ...")
            self.meta_classification()

        if self.args.show_results:
            print("-- showing the results ...")
            self.show_results()
