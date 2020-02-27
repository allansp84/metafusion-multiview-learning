# -*- coding: utf-8 -*-

import json

from abc import ABCMeta
from abc import abstractmethod
from sklearn import metrics
from matplotlib import ticker

from antispoofing.mcnns.utils import *
from antispoofing.mcnns.measure import *


class BaseClassifier(metaclass=ABCMeta):
    """ Abstract Class that will be the super-class of the others classes responsible for implementing the Classification.

    This Class implements the common methods that will be use for all subclasses and also defines two methods, training() and testing(),
    that must me implemented by the subclasses.

    Args:
        output_path (str): A path where the results and intermediate files will be saved.
        dataset (Dataset): A Dataset object containing the metadata of the dataset that will be used.
        fold (int): Tell which fold will be used during the processing.
        dataset_b (Dataset): A secondary dataset that will be used in the cross-dataset protocol.

    """

    def __init__(self, output_path, dataset, fold=0, dataset_b=None):

        self.verbose = True
        self.output_path = os.path.abspath(output_path)
        self.dataset = dataset
        self.dataset_b = dataset_b
        self.fold = fold

    def interesting_samples(self, all_fnames, test_sets, class_report, predictions, threshold_type='EER'):
        """ This method persists a dictionary containing interesting samples for later visual assessments, which contains the filenames of
        the samples that were incorrectly classified.

        Args:
            all_fnames (numpy.ndarray): A list containing the filename for each image in the dataset.
            test_sets (dict): A dictionary containing the data and the labels for all testing sets that compose the dataset.
            class_report (dict): A dictionary containing several evaluation measures for each testing set.
            predictions (dict): A dictionary containing the predicted scores and labels for each testing set.
            threshold_type (str): Defines what threshold will be considered for deciding the false acceptance and false rejections.

        """

        int_samples = {}
        predictions_test = predictions.copy()
        predictions_test.pop('train_set')

        for key in predictions_test:

            gt = predictions_test[key]['gt']
            scores = predictions_test[key]['predicted_scores']
            test_idxs = test_sets[key]['idxs']
            int_samples_idxs = get_interesting_samples(gt, scores, class_report[key][threshold_type]['threshold'], n_samples=5)

            int_samples[key] = {}

            for key_samples in int_samples_idxs.keys():
                int_samples[key][key_samples] = {'input_fnames': []}
                for idx in int_samples_idxs[key_samples]:
                    int_samples[key][key_samples]['input_fnames'] += [all_fnames[test_idxs[idx]]]

        json_fname = os.path.join(self.output_path, 'int_samples.json')
        with open(json_fname, mode='w') as f:
            print("--saving json file:", json_fname)
            sys.stdout.flush()
            f.write(json.dumps(int_samples, indent=4))

    def save_performance_results(self, class_report):
        """ Save the performance results in a .json file.

        Args:
            class_report (dict): A dictionary containing the evaluation results for each testing set.

        """

        print('-- saving the performance results in {0}\n'.format(self.output_path))
        sys.stdout.flush()

        for k in class_report:
            output_dir = os.path.join(self.output_path, k)
            try:
                os.makedirs(output_dir)
            except OSError:
                pass

            json_fname = os.path.join(output_dir, 'results.json')
            with open(json_fname, mode='w') as f:
                print("--saving results.json file:", json_fname)
                sys.stdout.flush()
                f.write(json.dumps(class_report[k], indent=4))

    def plot_score_distributions(self, thresholds, neg_scores, pos_scores, set_name):
        """ Plot the score distribution for a binary classification problem.

        Args:
            thresholds (list): A list of tuples containing the types and the values of the thresholds applied in this work.
            neg_scores (numpy.ndarray): The scores for the negative class.
            pos_scores (numpy.ndarray): The scores for the positive class.
            set_name (str): Name of the set used for computing the scores

        """

        plt.clf()
        plt.figure(figsize=(12, 10), dpi=100)

        plt.title("Score distributions (%s set)" % set_name)
        n, bins, patches = plt.hist(neg_scores, bins=25, normed=True, facecolor='green', alpha=0.5, histtype='bar', label='Negative class')
        na, binsa, patchesa = plt.hist(pos_scores, bins=25, normed=True, facecolor='red', alpha=0.5, histtype='bar', label='Positive class')

        # -- add a line showing the expected distribution
        y = mlab.normpdf(bins, np.mean(neg_scores), np.std(neg_scores))
        _ = plt.plot(bins, y, 'k--', linewidth=1.5)
        y = mlab.normpdf(binsa, np.mean(pos_scores), np.std(pos_scores))
        _ = plt.plot(binsa, y, 'k--', linewidth=1.5)

        for thr_type, thr_value in thresholds:
            plt.axvline(x=thr_value, linewidth=2, color='blue')
            plt.text(thr_value, 0, str(thr_type).upper(), rotation=90)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.xlabel('Scores', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)

        plt.legend()

        filename = os.path.join(self.output_path, '%s.score.distribution.png' % set_name)
        plt.savefig(filename)

    @staticmethod
    def plot_crossover_error_rate(neg_scores, pos_scores, filename, n_points=1000):
        """ TODO: Not ready yet.

        Args:
            neg_scores (numpy.ndarray):
            pos_scores (numpy.ndarray):
            filename (str):
            n_points (int):
        """

        fars, frrs, thrs = farfrr_curve(neg_scores, pos_scores, n_points=n_points)
        x_range = np.arange(0, len(thrs), 1)

        # -- create the general figure
        fig1 = plt.figure(figsize=(12, 8), dpi=300)

        # -- plot the FAR curve
        ax1 = fig1.add_subplot(111)
        ax1.plot(fars[x_range], 'b-')
        plt.ylabel("(BPCER) FAR")

        # -- plot the FRR curve
        ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
        ax2.plot(frrs[x_range], 'r-')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        plt.ylabel("(APCER) FRR")

        plt.xticks()
        ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        plt.xlabel('Threshold')
        plt.ylabel('Error Rate')

        plt.show()
        plt.savefig(filename)

    @staticmethod
    def classification_results_summary(report):
        """ This method is responsible for printing a summary of the classification results.

        Args:
            report (dict): A dictionary containing the measures (e.g., Acc, APCER)  computed for each test set.

        """

        print('-- Classification Results')

        headers = ['Testing set', 'Threshold (value)', 'AUC', 'ACC', 'Balanced ACC', 'BPCER (FAR)', 'APCER (FRR)', 'HTER']
        header_line = "| {:<20s} | {:<20s} | {:<12s} | {:<12s} | {:<12s} | {:<12s} | {:<12s} | {:<8s} |\n".format(*headers)
        sep_line = '-' * (len(header_line)-1) + '\n'

        final_report = sep_line
        final_report += header_line

        for k1 in sorted(report):
            final_report += sep_line
            for k2 in sorted(report[k1]):
                values = [k1,
                          "{} ({:.2f})".format(k2, report[k1][k2]['threshold']),
                          report[k1][k2]['auc'],
                          report[k1][k2]['acc'],
                          report[k1][k2]['bacc'],
                          report[k1][k2]['bpcer'],
                          report[k1][k2]['apcer'],
                          report[k1][k2]['hter'],
                          ]
                line = "| {:<20s} | {:<20s} | {:<12.4f} | {:<12.4f} | {:<12.4f} | {:<12.4f} | {:<12.4f} | {:<8.4f} |\n".format(*values)
                final_report += line
        final_report += sep_line

        print(final_report)
        sys.stdout.flush()

    def performance_evaluation(self, predictions):
        """ Compute the performance of the fitted model for each test set.

        Args:
            predictions (dict): A dictionary with the ground-truth, the predicted scores and the predicted labels for the testing data. For Exemple: {'test': {'gt': y_test, 'predicted_labels': y_pred, 'predicted_scores': y_scores}}

        Returns:
            dict: A dictionary containing the performance results for each testing set. For example: {'test': {'acc': acc, 'apcer': apcer, 'bpcer': bpcer}}

        """

        try:
            os.makedirs(self.output_path)
        except OSError:
            pass

        report = {}

        # -- compute the thresholds using the training set
        gt_train = predictions['train_set']['gt']
        pred_scores_train = predictions['train_set']['predicted_scores']
        genuine_scores_train, attack_scores_train = split_score_distributions(gt_train, pred_scores_train, label_neg=0, label_pos=1)

        far_thr = far_threshold(genuine_scores_train, attack_scores_train, far_at=0.01)
        eer_thr = eer_threshold(genuine_scores_train, attack_scores_train)

        thresholds = [('FAR@0.01', far_thr),
                      ('EER', eer_thr),
                      ('0.5', 0.5),
                      ]

        # -- plotting the score distribution for the training set
        self.plot_score_distributions(thresholds, genuine_scores_train, attack_scores_train, 'train')

        # -- compute the evaluation metrics for the test sets
        for key in predictions:
            report[key] = {}

            ground_truth = predictions[key]['gt']
            pred_scores = predictions[key]['predicted_scores']

            neg_scores, pos_scores = split_score_distributions(ground_truth, pred_scores, label_neg=0, label_pos=1)

            # -- compute the Area Under ROC curve
            roc_auc = metrics.roc_auc_score(ground_truth, pred_scores)

            for thr_type, thr_value in thresholds:

                # -- compute the FAR and FRR
                # -- FAR (BPCER) is the rate of Genuine images classified as Presentation Attacks images
                # -- FRR (APCER) is the rate of Presentation Attack images classified as Genuine images
                # -- Note: Presentation Attacks images represent the Positive Class (label 1) and the
                # --       genuine images represent the Negative Class (0)
                bpcer, apcer = farfrr(neg_scores, pos_scores, thr_value)

                hter = (bpcer + apcer)/2.

                # -- compute the ACC and Balanced ACC
                acc = acc_threshold(ground_truth, pred_scores, thr_value, label_neg=0, label_pos=1)
                bacc = bacc_threshold(ground_truth, pred_scores, thr_value, label_neg=0, label_pos=1)

                # -- save the results in a dictionary
                report[key][thr_type] = {'auc': roc_auc, 'acc': acc, 'bacc': bacc, 'threshold': thr_value,
                                         'apcer': apcer, 'bpcer': bpcer, 'hter': hter,
                                         }

            # -- plotting the score distribution and the Crossover Error Rate (CER) graph
            self.plot_score_distributions(thresholds, neg_scores, pos_scores, key)
            self.plot_crossover_error_rate(neg_scores, pos_scores, filename=os.path.join(self.output_path, '%s.cer.png' % key))

        if self.verbose:
            self.classification_results_summary(report)

        return report

    def _save_raw_scores(self, all_fnames, dataset_protocol, predictions):
        """ Save the scores obtained for the trained model.

        Args:
            all_fnames (numpy.ndarray): List of filenames of all images in the dataset.
            dataset_protocol (dict): A dictionary containing the metadata of the dataset.
            predictions (dict): A dictionary containing the obtained scores.
        """

        # for key in dataset_protocol['test_set']:
        #     indexes = dataset_protocol['test_set'][key]['idxs']
        #     aux = zip(all_fnames[indexes], predictions[key]['predicted_scores'], predictions[key]['predicted_labels'])
        #     dt = dict(names=('fname', 'predicted_score', 'predicted_label'), formats=('U300', np.float32, np.int32))
        #     results = np.array(list(aux), dtype=dt)
        #     np.savetxt(os.path.join(self.output_path, '%s.predictions.txt' % key), results, fmt="%s,%.6f,%d", delimiter=',')

        # json_fname = os.path.join(self.output_path, 'predictions.json')
        # with open(json_fname, mode='w') as f:
        #     print("--saving json file:", json_fname, flush=True)
        #     convert_numpy_dict_items_to_list(predictions)
        #     f.write(json.dumps(predictions, indent=4))

        fname = os.path.join(self.output_path, 'predictions.pkl')
        save_object(predictions, fname)

    def run_evaluation_protocol(self):
        """
        This method implements the whole training and testing process considering the evaluation protocol defined for the dataset.
        """

        try:
            os.makedirs(self.output_path)
        except OSError:
            pass

        predictions = {}

        # -- get the sets of images according to the protocol evaluation defined in each dataset.
        dataset_protocol = self.dataset.protocol_eval(fold=self.fold, output_path=self.output_path)

        # -- start the training process
        self.training(dataset_protocol['train_set']['data'], dataset_protocol['train_set']['labels'],
                      dataset_protocol['test_set']['test']['data'], dataset_protocol['test_set']['test']['labels'])

        # -- compute the predicted scores and labels for the training set
        if dataset_protocol['train_set']['data'].size:
            predictions['train_set'] = self.testing(dataset_protocol['train_set']['data'], dataset_protocol['train_set']['labels'])

        # -- compute the predicted scores and labels for the testing sets
        for key in dataset_protocol['test_set']:
            if dataset_protocol['test_set'][key]['data'].size:
                predictions[key] = self.testing(dataset_protocol['test_set'][key]['data'], dataset_protocol['test_set'][key]['labels'])

        # -- estimating the performance of the classifier
        class_report = self.performance_evaluation(predictions)

        # -- saving the performance results
        self.save_performance_results(class_report)

        # -- saving the raw scores
        self._save_raw_scores(self.dataset.meta_info['all_fnames'], dataset_protocol, predictions)

        # -- saving the interesting samples for further analysis
        self.interesting_samples(self.dataset.meta_info['all_fnames'], dataset_protocol['test_set'], class_report, predictions,
                                 threshold_type='EER')

        # --Cross-Dataset
        if self.dataset_b is not None:
            predictions_b = {}
            dataset_b_protocol = self.dataset_b.protocol_eval(fold=self.fold)

            # -- compute the predicted scores and labels for the training set
            if dataset_b_protocol['train_set']['data'].size:
                predictions_b['train_set'] = self.testing(dataset_b_protocol['train_set']['data'],
                                                          dataset_b_protocol['train_set']['labels'])

            # -- compute the predicted scores and labels for the testing sets
            for key in dataset_b_protocol['test_set']:
                if dataset_b_protocol['test_set'][key]['data'].size:
                    predictions_b[key] = self.testing(dataset_b_protocol['test_set'][key]['data'],
                                                      dataset_b_protocol['test_set'][key]['labels'])

            # -- estimating the performance of the classifier
            class_report = self.performance_evaluation(predictions_b)

            # -- saving the performance results
            self.save_performance_results(class_report)

            # -- saving the raw scores
            self._save_raw_scores(self.dataset_b.meta_info['all_fnames'], dataset_b_protocol, predictions_b)

            # -- saving the interesting samples for further analysis
            self.interesting_samples(self.dataset_b.meta_info['all_fnames'], dataset_b_protocol['test_set'], class_report, predictions_b,
                                     threshold_type='EER')

    def run(self):
        """
        Start the classification step.
        """

        try:
            os.makedirs(self.output_path)
        except OSError:
            pass

        self.run_evaluation_protocol()

    @abstractmethod
    def training(self, x_train, y_train, x_validation=None, y_validation=None):
        """ This method implements the training stage.

        The training stage will be implemented by the subclasses, taking into account the particularities of the classification algorithm to be used.

        Args:
            x_train (numpy.ndarray): A multidimensional array containing the feature vectors (or images) to be used to train a classifier.
            y_train (numpy.ndarray): A multidimensional array containing the labels refers to the feature vectors that will be used during the training stage.
            x_validation (numpy.ndarray, optional): A multidimensional array containing the feature vectors (or images) to be used to test the classifier.
            y_validation (numpy.ndarray, optional): A multidimensional array containing the labels refers to the feature vectors that will be used for testing the classification model.
        """
        return NotImplemented

    @abstractmethod
    def testing(self, x_test, y_test):
        """ This method implements the testing stage.

        The testing stage will be implemented by the subclasses.

        Args:
            x_test (numpy.ndarray): A multidimensional array containing the feature vectors (or images) to be used to test the classifier.
            y_test (numpy.ndarray): A multidimensional array containing the labels refers to the feature vectors that will be used to test the classification model.

        Returns:
            A dictionary with the ground-truth, the predicted scores and the predicted labels for the testing data, such as {'gt': y_test, 'predicted_labels': y_pred, 'predicted_scores': y_scores}
        """
        return NotImplemented
