# -*- coding: utf-8 -*-

import json

from abc import ABCMeta
from abc import abstractmethod

from sklearn import metrics
from matplotlib import ticker
from operator import itemgetter

from antispoofing.mcnns.utils import *
from antispoofing.mcnns.measure import *

from sklearn import preprocessing
from sklearn.datasets import dump_svmlight_file
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import scipy.stats as stats


class BaseMetaClassifier(metaclass=ABCMeta):

    def __init__(self, output_path, predictions_files, gt_files, meta_classification_from, n_models,
                 step_algo='selection', selection_algo=0, compute_feature_importance=False, fold=1, n_jobs=-1):

        self.verbose = True
        self.output_path = os.path.abspath(output_path)
        self.predictions_files = predictions_files
        self.gt_files = gt_files
        self.meta_classification_from = meta_classification_from
        self.n_models = n_models
        self.step_algo = step_algo
        self.selection_algo = selection_algo
        self.compute_feature_importance = compute_feature_importance
        self.fold = fold
        self.n_jobs = n_jobs

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
            int_samples_idxs = get_interesting_samples(gt, scores, class_report[key][threshold_type]['threshold'])

            int_samples[key] = {}

            for key_samples in int_samples_idxs.keys():
                int_samples[key][key_samples] = {'input_fnames': []}
                for idx in int_samples_idxs[key_samples]:
                    int_samples[key][key_samples]['input_fnames'] += [all_fnames[test_idxs[idx]]]

        json_fname = os.path.join(self.output_path, 'int_samples.json')
        with open(json_fname, mode='w') as f:
            print("-- saving json file:", json_fname)
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
                print("-- saving results.json file:", json_fname)
                sys.stdout.flush()
                f.write(json.dumps(class_report[k], indent=4))

    def plot_roc_curve(self, false_positive_rate, true_positive_rate, roc_auc, filename, set_name):
        """

        Args:
            false_positive_rate (numpy.ndarray):
            true_positive_rate (numpy.ndarray):
            n_points (int): Number of points considered to build the curve.
            axis_font_size (str): A string specifying the axis font size.
            **kwargs: Optional arguments for the plot function of the matplotlib.pyplot package.

        """
        plt.clf()
        plt.figure(figsize=(10, 10), dpi=100)

        title_font = {'size': '18', 'color': 'black'}
        plt.title("Receiver operating characteristic Curve (%s set)" % set_name, **title_font)
        plt.plot(false_positive_rate, true_positive_rate,
                 color=(0, 0, 0), marker='o', linestyle='-', linewidth=2, label='AUC = %2.4f)' % roc_auc)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])

        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.legend(loc="lower right", fontsize=16)
        plt.savefig(filename)

    def plot_score_distributions(self, thresholds, neg_scores, pos_scores, filename, set_name):
        """ Plot the score distribution for a binary classification problem.

        Args:
            thresholds (list): A list of tuples containing the types and the values of the thresholds applied in this work.
            neg_scores (numpy.ndarray): The scores for the negative class.
            pos_scores (numpy.ndarray): The scores for the positive class.
            set_name (str): Name of the set used for computing the scores

        """

        plt.clf()
        plt.figure(figsize=(12, 9), dpi=100)

        title_font = {'size': '18', 'color': 'black'}
        plt.title("Score distributions (%s set)" % set_name, **title_font)

        n, bins, patches = plt.hist(neg_scores, bins=30, density=True, facecolor='red', alpha=0.5, histtype='bar', label='Negative class')
        na, binsa, patchesa = plt.hist(pos_scores, bins=30, density=True, facecolor='green', alpha=0.5, histtype='bar', label='Positive class')

        # -- add a line showing the expected distribution
        y = mlab.normpdf(bins, np.mean(neg_scores), np.std(neg_scores))
        _ = plt.plot(bins, y, 'k--', linewidth=1.5)
        y = mlab.normpdf(binsa, np.mean(pos_scores), np.std(pos_scores))
        _ = plt.plot(binsa, y, 'k--', linewidth=1.5)

        for thr_type, thr_value in thresholds:
            plt.axvline(x=thr_value, linewidth=1, color='blue')
            plt.text(thr_value, -3, str(thr_type).upper(), fontsize=14, rotation=90)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # plt.xlabel('Scores', fontsize=16)
        plt.ylabel('Frequency', fontsize=16)

        plt.legend(fontsize=16)
        plt.subplots_adjust(left=0.125, bottom=0.2, right=0.9, top=0.9, wspace=0, hspace=0)

        plt.savefig(filename)

    @staticmethod
    def plot_det_curve(neg_scores, pos_scores, filename, key):

        # -- bottom vertical alignment for more space
        title_font = {'size': '18', 'color': 'black', 'weight': 'normal', 'verticalalignment': 'bottom'}
        axis_font = {'size': '14'}
        font_size_axis = 12

        fig1 = plt.figure(figsize=(8, 6), dpi=100)
        plt.clf()
        n_points = 300
        det(neg_scores, pos_scores, n_points, color=(0, 0, 0), marker='o', linestyle='-', linewidth=2, label=key)
        det_axis([0.01, 40, 0.01, 40])
        plt.xlabel('FRR (%)', **axis_font)
        plt.ylabel('FAR (%)', **axis_font)

        plt.xticks(size=font_size_axis)
        plt.yticks(size=font_size_axis)

        plt.legend()
        title = 'DET Curve'
        plt.title(title, **title_font)
        plt.grid(True)

        fig1.savefig(filename)

    @staticmethod
    def plot_crossover_error_rate(neg_scores, pos_scores, filename, n_points=100):
        """ TODO: Not ready yet.

        Args:
            neg_scores (numpy.ndarray):
            pos_scores (numpy.ndarray):
            filename (str):
            n_points (int):
        """

        fars, frrs, thrs = farfrr_curve(neg_scores, pos_scores, n_points=n_points)

        # -- create the general figure
        fig1 = plt.figure(figsize=(12, 8), dpi=300)

        # -- plot the FAR curve
        ax1 = fig1.add_subplot(111)
        ax1.plot(fars, thrs, 'b-')
        plt.ylabel("(BPCER) FAR")

        # -- plot the FRR curve
        ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
        ax2.plot(frrs, thrs, 'r-')
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
        if self.verbose: print('-- performance_evaluation', flush=True)

        try:
            os.makedirs(self.output_path)
        except OSError:
            pass

        report = {}

        # -- compute the thresholds using the training set
        gt_train = predictions['train_set']['gt']
        pred_scores_train = predictions['train_set']['predicted_scores']

        scaler = preprocessing.MinMaxScaler().fit(pred_scores_train.reshape(-1, 1))
        pred_scores_train = scaler.transform(pred_scores_train.reshape(-1, 1)).flatten()

        genuine_scores_train, attack_scores_train = split_score_distributions(gt_train, pred_scores_train, label_neg=0, label_pos=1)

        far_thr = far_threshold(genuine_scores_train, attack_scores_train, far_at=0.01)
        eer_thr = eer_threshold(genuine_scores_train, attack_scores_train)
        min_eer_thr = min_eer_threshold(genuine_scores_train, attack_scores_train)
        hter_thr = min_hter_threshold(genuine_scores_train, attack_scores_train)

        thresholds = [('0.5', 0.5),
                      ]

        # # -- plotting the score distribution for the training set
        # self.plot_score_distributions(thresholds, genuine_scores_train, attack_scores_train, 'train')

        # -- compute the evaluation metrics for the test sets
        for key in predictions:
            report[key] = {}

            ground_truth = predictions[key]['gt']
            pred_scores = predictions[key]['predicted_scores']
            pred_scores = scaler.transform(pred_scores.reshape(-1, 1)).flatten()

            neg_scores, pos_scores = split_score_distributions(ground_truth, pred_scores, label_neg=0, label_pos=1)
            # import pdb; pdb.set_trace()
            # -- compute ROC curve and AUC value
            fpr, tpr, _ = metrics.roc_curve(ground_truth, pred_scores)
            roc_auc = metrics.roc_auc_score(ground_truth, pred_scores)

            for thr_type, thr_value in thresholds:

                # -- compute the FAR and FRR
                # -- FAR (BPCER) is the rate of Genuine images classified as Presentation Attacks images
                # -- FRR (APCER) is the rate of Presentation Attack images classified as Genuine images
                # -- Note: Presentation Attacks images represent the Positive Class (label 1) are the
                # --       genuine images represent the Negative Class (0)

                if '0.5' in thr_type:
                    # neg_scores = neg_scores + (thr_value - hter_thr)
                    # pos_scores = pos_scores + (thr_value - hter_thr)
                    # pred_scores = pred_scores + (thr_value - hter_thr)

                    bpcer, apcer = farfrr(neg_scores, pos_scores, thr_value)

                    # -- compute the ACC and Balanced ACC
                    acc = acc_threshold(ground_truth, pred_scores, thr_value, label_neg=0, label_pos=1)
                    bacc = bacc_threshold(ground_truth, pred_scores, thr_value, label_neg=0, label_pos=1)

                else:
                    bpcer, apcer = farfrr(neg_scores, pos_scores, thr_value)
                    # -- compute the ACC and Balanced ACC
                    acc = acc_threshold(ground_truth, pred_scores, thr_value, label_neg=0, label_pos=1)
                    bacc = bacc_threshold(ground_truth, pred_scores, thr_value, label_neg=0, label_pos=1)

                # import pdb; pdb.set_trace()
                acc = accuracy_score(ground_truth, predictions[key]['predicted_labels'])
                bacc = balanced_accuracy_score(ground_truth, predictions[key]['predicted_labels'])
                hter = (bpcer + apcer)/2.

                # -- save the results in a dictionary
                report[key][thr_type] = {'auc': roc_auc, 'acc': acc, 'bacc': bacc, 'threshold': thr_value,
                                         'apcer': apcer, 'bpcer': bpcer, 'hter': hter,
                                         }

            # -- plotting the score distribution and the Crossover Error Rate (CER) graph
            self.plot_roc_curve(fpr, tpr, roc_auc, filename=os.path.join(self.output_path, '%s.roc_curve.png' % key), set_name=key)
            self.plot_score_distributions(thresholds, neg_scores, pos_scores, filename=os.path.join(self.output_path, '%s.score.distribution.png' % key), set_name=key)
            self.plot_crossover_error_rate(neg_scores, pos_scores, filename=os.path.join(self.output_path, '%s.cer.png' % key))
            self.plot_det_curve(neg_scores, pos_scores, filename=os.path.join(self.output_path, '%s.det_curve.png' % key), key=key)

        if self.verbose:
            self.classification_results_summary(report)

        return report

    def complementary_by_cohen_kappa_score(self, selected_models_by_importance_idxs, lpredictors):

        lpredictors = np.array(lpredictors)

        all_agreement_coefs = []
        for a, pred_a in zip(selected_models_by_importance_idxs, lpredictors[selected_models_by_importance_idxs]):
            kappa_coefs = []
            for b, pred_b in enumerate(lpredictors):
                try:
                    kc = metrics.cohen_kappa_score(pred_a, pred_b)
                except ValueError:
                    raise Exception('Please, use the predicted labels, instead of predicted scores')

                if np.isnan(kc):
                    kc = 0.

                kappa_coefs += [[kc, a, b]]

            if len(kappa_coefs):
                kappa_coefs = sorted(kappa_coefs, key=itemgetter(0), reverse=True)
            all_agreement_coefs += [kappa_coefs]

        return all_agreement_coefs

    def complementary_by_kendalltau(self, selected_models_by_importance_idxs, lpredictors):

        lpredictors = np.array(lpredictors)

        all_agreement_coefs = []
        for a, pred_a in zip(selected_models_by_importance_idxs, lpredictors[selected_models_by_importance_idxs]):
            kappa_coefs = []
            for b, pred_b in enumerate(lpredictors):

                kc, pv = stats.kendalltau(pred_a, pred_b)
                if np.isnan(kc):
                    kc = 0.

                kappa_coefs += [[kc, a, b]]

            if len(kappa_coefs):
                kappa_coefs = sorted(kappa_coefs, key=itemgetter(0), reverse=True)
            all_agreement_coefs += [kappa_coefs]

        return all_agreement_coefs

    def complementary_by_q_stat(self, selected_models_by_importance_idxs, lpredictors):

        lpredictors = np.array(lpredictors)

        all_agreement_coefs = []
        for a, pred_a in zip(selected_models_by_importance_idxs, lpredictors[selected_models_by_importance_idxs]):

            kappa_coefs = []

            for b, pred_b in enumerate(lpredictors):

                cm_a, cm_b, cm_c, cm_d = 0, 0, 0, 0

                pred_a = np.array(pred_a)
                pred_b = np.array(pred_b)
                n_total = len(pred_a)

                for pa, pb in zip(pred_a, pred_b):

                    if pa == pb:
                        if pa == 1:
                            cm_a += 1
                        if pa == 0:
                            cm_d += 1

                    if pa != pb:
                        if pa == 1 and pb == 0:
                            cm_b += 1
                        if pa == 0 and pb == 1:
                            cm_c += 1

                cm_a /= n_total
                cm_b /= n_total
                cm_c /= n_total
                cm_d /= n_total

                numerator = (cm_a * cm_d - cm_b * cm_c)
                denominator = (cm_a * cm_d + cm_b * cm_c)

                kc = 0.
                if denominator:
                    kc = numerator/denominator

                kappa_coefs += [[kc, a, b]]

            if len(kappa_coefs):
                kappa_coefs = sorted(kappa_coefs, key=itemgetter(0), reverse=True)
            all_agreement_coefs += [kappa_coefs]

        return all_agreement_coefs

    def find_complementary_models(self, x_train, n_models=-1, method='kappa'):

        min_value, max_value = .85, .98

        selected_models_by_importance_idxs = self.select_classifiers_by_feature_importance(n_models=self.n_models)

        n_selected_models = len(selected_models_by_importance_idxs)

        # import pdb; pdb.set_trace()

        if 'kappa' in method:
            all_agreement_coefs = self.complementary_by_cohen_kappa_score(selected_models_by_importance_idxs, x_train.T)
        elif 'q_stat' in method:
            all_agreement_coefs = self.complementary_by_q_stat(selected_models_by_importance_idxs, x_train.T)
        elif 'kendalltau' in method:
            all_agreement_coefs = self.complementary_by_kendalltau(selected_models_by_importance_idxs, x_train.T)
        else:
            raise('Method not implemented')

        all_selected_models_idxs = []
        for kappa_coefs in all_agreement_coefs:
            selected_models_idxs = []
            print('-- kappa_coefs:', kappa_coefs)
            for k, a, b in kappa_coefs:
                if k >= min_value and k <= max_value:
                # if k >= min_value:
                    selected_models_idxs += [a, b]
                # if n_models == 0:
                #     # if k > min_value and k < max_value:
                #     if k > min_value:
                #         selected_models_idxs += [a, b]
                # else:
                #     selected_models_idxs += [a, b]

            selected_models_idxs = list(unique_everseen(selected_models_idxs))

            all_selected_models_idxs += [selected_models_idxs]

        # all_selected_models_idxs = list(unique_everseen(np.concatenate(all_selected_models_idxs)))

        final_list = []
        for list_a in all_selected_models_idxs:
            for list_b in all_selected_models_idxs:
                final_list += [np.intersect1d(list_a, list_b)]

        final_list = np.concatenate(final_list)

        counter = np.zeros((final_list.max()+1,),dtype=np.int)
        for elem in final_list:
            counter[elem] += 1

        all_selected_models_idxs = []
        for c, idx  in zip(counter[counter.argsort()[::-1]], counter.argsort()[::-1]):
            if c > 1:
                all_selected_models_idxs += [idx]

        # all_selected_models_idxs = np.concatenate((selected_models_by_importance_idxs, all_selected_models_idxs))
        # all_selected_models_idxs = list(unique_everseen(all_selected_models_idxs))

        # return all_selected_models_idxs[:n_models*4]
        return all_selected_models_idxs[:n_models]

    def select_classifiers_by_feature_importance(self, n_models=-1):
        # import pdb; pdb.set_trace()
        json_fname = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(self.output_path)))),
                                  'predictor_importances-fold-{}.json'.format(self.fold))

        # # -- open json file
        # json_fname = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(self.output_path)))), 'predictor_importances.json')
        
        print("-- reading the predictors importance:", json_fname, flush=True)
        # data = json.load(open(json_fname, 'r'))
        dt = dict(names=('idxs', 'importance'), formats=(np.int, np.float32))
        data = np.loadtxt(json_fname, dtype=dt, delimiter=',')

        data = sorted(data, key=itemgetter(1), reverse=True)
        data = data[:n_models]
        print('-- data', data)

        selected_models_idxs = []
        for idx, importance in data:

            # -- remove the features that has no importance according to random forest
            if importance > 0.:
                selected_models_idxs += [int(idx)]

        print('-- selected_models_idxs', sorted(selected_models_idxs), flush=True)
        selected_models_idxs = np.array(selected_models_idxs)

        return selected_models_idxs[:n_models]

    @staticmethod
    def get_method_name(fname):
        return "{}_{}".format(os.path.basename(os.path.dirname(fname)), os.path.basename(fname).split('_fold')[0])

    @staticmethod
    def concatenate_files(fnames, dtype=np.int32):

        fnames = fnames.flatten()

        preds = []
        for fname in fnames:
            preds += [np.loadtxt(fname,dtype=dtype,delimiter=',')]
        preds = np.array(preds).flatten()

        return preds

    def load_predictions_v1(self):

        # -- load the ground-truth
        assert self.gt_files.size == 2, "GT files not found"

        gt_train_idx = [i for i, v in enumerate(self.gt_files) if 'split_80' in v][0]
        gt_test_idx = [i for i, v in enumerate(self.gt_files) if 'split_20' in v][0]

        gt_train = np.loadtxt(self.gt_files[gt_train_idx], dtype=np.uint32, delimiter=',')
        gt_test = np.loadtxt(self.gt_files[gt_test_idx], dtype=np.uint32,delimiter=',')

        # -- get the name of the methods
        methods = []
        for fpred in self.predictions_files:
            # if 'logistic' in fpred or 'mlp' in fpred or 'div' in fpred: 
            # if 'densenet121' in fpred or 'xception' in fpred or 'vgg19' in fpred: 
                methods += [self.get_method_name(fpred)]
        methods = dict.fromkeys(np.unique(methods), {'train_set':[], 'test':[]})

        print("Methods", list(methods.keys()))

        for key in methods:
            filter_by_method = [fpred for fpred in self.predictions_files if key == self.get_method_name(fpred)]

            train_set = [fm for fm in filter_by_method if 'split_80' in fm]
            test = [fm for fm in filter_by_method if 'split_20' in fm]

            methods[key]['train_set'] = np.reshape(train_set, (-1, 5))
            methods[key]['test'] = np.reshape(test, (-1, 5))

        preds = []
        for key in methods:

            preds_train = self.concatenate_files(methods[key]['train_set'][:, (self.fold-1)])
            preds_test = self.concatenate_files(methods[key]['test'][:, (self.fold-1)])

            rdict = {'train_set': {'gt': gt_train,
                                   'predicted_labels': preds_train,
                                   'predicted_scores': preds_train,
                                   },
                     'test': {'gt': gt_test,
                              'predicted_labels': preds_test,
                              'predicted_scores': preds_test,
                              },
                    }
            preds += [rdict]

        return preds, list(methods.keys())

    def load_predictions_v2(self):

        # -- get the name of the methods
        methods_id, empty_sets = [], []
        for fpred in self.predictions_files:
            # if 'logistic' in fpred or 'mlp' in fpred or 'div' in fpred: 
            # if 'densenet121' in fpred or 'xception' in fpred or 'vgg19' in fpred: 
            methods_id += [self.get_method_name(fpred)]
            empty_sets += [{'train_set':[], 'test':[]}]

        # methods = dict.fromkeys(np.unique(methods), {'train_set':[], 'test':[]})
        zip_methods = zip(methods_id, empty_sets)
        methods = dict(zip_methods)
        print("Methods", len(list(methods.keys())), list(methods.keys()))

        for key in methods:
            print('-- Key', key)
            # filter_by_method = []
            # for fpred in self.predictions_files:
            #     if key == self.get_method_name(fpred):
            #         filter_by_method += [fpred]
            filter_by_method = [fpred for fpred in self.predictions_files if key == self.get_method_name(fpred)]

            train_set = np.array([fm for fm in filter_by_method if '/valid/' in fm])
            test = np.array([fm for fm in filter_by_method if '/test/' in fm])

            # print(filter_by_method)
            # print('train_set', train_set)
            # print(methods[key]['train_set'])
            methods[key]['train_set'] = np.reshape(train_set, (-1, 5))
            methods[key]['test'] = np.reshape(test, (-1, 5))
            # import pdb; pdb.set_trace()

        # self.filter_by_method(self.predictions_files, methods)
        preds = []
        for key in methods:
            # print('-- loading', methods[key]['train_set'])

            preds_train = self.concatenate_files(methods[key]['train_set'][:, (self.fold-1)])
            preds_test = self.concatenate_files(methods[key]['test'][:, (self.fold-1)])

            gt_train_files = np.chararray.replace(methods[key]['train_set'], '.predictions', '.labels')
            gt_test_files = np.chararray.replace(methods[key]['test'], '.predictions', '.labels')

            gt_train = self.concatenate_files(gt_train_files[:, (self.fold-1)])
            gt_test = self.concatenate_files(gt_test_files[:, (self.fold-1)])

            rdict = {'train_set': {'gt': gt_train,
                                   'predicted_labels': preds_train,
                                   'predicted_scores': preds_train,
                                   },
                     'test': {'gt': gt_test,
                              'predicted_labels': preds_test,
                              'predicted_scores': preds_test,
                              },
                    }
            preds += [rdict]

        return preds, list(methods.keys())

    def load_predictions_v3(self):

        # -- get the name of the methods
        methods_id, empty_sets = [], []
        for fpred in self.predictions_files:
            # if 'logistic' in fpred or 'mlp' in fpred or 'div' in fpred: 
            # if 'densenet121' in fpred or 'xception' in fpred or 'vgg19' in fpred: 
            methods_id += [self.get_method_name(fpred)]
            empty_sets += [{'train_set':[], 'test':[]}]

        # methods = dict.fromkeys(np.unique(methods), {'train_set':[], 'test':[]})
        zip_methods = zip(methods_id, empty_sets)
        methods = dict(zip_methods)
        print("Methods", len(list(methods.keys())), list(methods.keys()))

        for key in methods:
            print('-- Key', key)
            # filter_by_method = []
            # for fpred in self.predictions_files:
            #     if key == self.get_method_name(fpred):
            #         filter_by_method += [fpred]
            filter_by_method = [fpred for fpred in self.predictions_files if key == self.get_method_name(fpred)]

            train_set = np.array([fm for fm in filter_by_method if '/train/' in fm])
            test = np.array([fm for fm in filter_by_method if '/test/' in fm])

            # print(filter_by_method)
            # print('train_set', train_set)
            # print(methods[key]['train_set'])
            methods[key]['train_set'] = np.reshape(train_set, (-1, 5))
            methods[key]['test'] = np.reshape(test, (-1, 5))
            # import pdb; pdb.set_trace()

        if 'scores' in self.meta_classification_from:
            prediction_type = np.float32
        else:
            prediction_type = np.int32

        # self.filter_by_method(self.predictions_files, methods)
        preds = []
        for key in methods:
            # print('-- loading', methods[key]['train_set'])
            # import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()
            train_idxs = (np.core.chararray.find(methods[key]['train_set'], 'train{}'.format(self.fold)) != -1)
            validation_idxs = (np.core.chararray.find(methods[key]['train_set'], 'valid{}'.format(self.fold)) != -1)

            preds_train = self.concatenate_files(methods[key]['train_set'][train_idxs], dtype=prediction_type)
            preds_valid = self.concatenate_files(methods[key]['train_set'][validation_idxs], dtype=prediction_type)
            preds_test = self.concatenate_files(methods[key]['test'][:, (self.fold-1)], dtype=prediction_type)

            gt_train_files = np.chararray.replace(methods[key]['train_set'], '.predictions', '.labels')
            gt_test_files = np.chararray.replace(methods[key]['test'], '.predictions', '.labels')

            gt_train = self.concatenate_files(gt_train_files[train_idxs], dtype=np.int32)
            gt_valid = self.concatenate_files(gt_train_files[validation_idxs], dtype=np.int32)
            gt_test = self.concatenate_files(gt_test_files[:, (self.fold-1)], dtype=np.int32)

            rdict = {'train_set': {'gt': gt_train,
                                   'predicted_labels': preds_train,
                                   'predicted_scores': preds_train,
                                   },
                     # 'test': {'gt': gt_test,
                     #          'predicted_labels': preds_test,
                     #          'predicted_scores': preds_test,
                     #          },
                     'test': {'gt': gt_valid,
                              'predicted_labels': preds_valid,
                              'predicted_scores': preds_valid,
                              },
                    }
            preds += [rdict]

        return preds, list(methods.keys())

    def filter_by_method(self, files, methods):
        pass

    def run_training_test_protocol(self, x_train, y_train, x_test, y_test, methods_name):
        print('-- x_train', x_train.shape)
        print('-- y_train', y_train.shape)
        print('-- x_test', x_test.shape)
        print('-- y_test', y_test.shape)

        start = get_time()

        meta_predictions = {}
        dataset_protocol = {'train_set': {'data': x_train, 'labels': y_train},
                            'test_set': {'test': {'data': x_test, 'labels': y_test},
                                         }
                            }

        # -- start the training process
        self.training(dataset_protocol['train_set']['data'], dataset_protocol['train_set']['labels'],
                      dataset_protocol['test_set']['test']['data'], dataset_protocol['test_set']['test']['labels'],
                      methods_name=methods_name)

        elapsed = total_time_elapsed(start, get_time())
        print('-- training process (time consumption): {0}!'.format(elapsed), flush=True)

        start = get_time()

        # -- compute the predicted scores and labels for the training set
        if dataset_protocol['train_set']['data'].size:
            meta_predictions['train_set'] = self.testing(dataset_protocol['train_set']['data'], dataset_protocol['train_set']['labels'])

        elapsed = total_time_elapsed(start, get_time())
        print('-- predictions for train_set (time consumption): {0}!'.format(elapsed), flush=True)

        # -- compute the predicted scores and labels for the testing sets
        for key in dataset_protocol['test_set']:
            if dataset_protocol['test_set'][key]['data'].size:
                start = get_time()
                meta_predictions[key] = self.testing(dataset_protocol['test_set'][key]['data'], dataset_protocol['test_set'][key]['labels'])
                elapsed = total_time_elapsed(start, get_time())
                print('-- predictions for {0} (time consumption): {1}!'.format(key, elapsed), flush=True)

        # -- estimating the performance of the classifier
        class_report = self.performance_evaluation(meta_predictions)

    def problem_reading(self, all_predictions):
        x_train_pred, x_train_pred, y_train = [], [], []
        x_test_pred, x_test_pred, y_test = [], [], []

        key = 'predicted_{0}'.format(self.meta_classification_from)

        for preds in all_predictions:

            x_train_pred += [preds['train_set'][key]]
            x_test_pred += [preds['test'][key]]

            # -- the ground-truth for all models are iquals, so it's necessary to get only the last one
            y_train = [preds['train_set']['gt']]
            y_test = [preds['test']['gt']]

        x_train = np.array(x_train_pred).transpose()
        x_test = np.array(x_test_pred).transpose()
        y_train = np.array(y_train).flatten()
        y_test = np.array(y_test).flatten()

        if 'scores' in self.meta_classification_from:
            x_train_scale = preprocessing.MinMaxScaler().fit(x_train)
            x_train = x_train_scale.transform(x_train)
            x_test = x_train_scale.transform(x_test)

        return x_train, y_train, x_test, y_test

    def save_feature_importance(self, importances, methods_name=None):

        idxs = np.arange(len(importances))
        sorted_idxs = np.argsort(importances)[::-1]

        aux = zip(idxs, importances)

        dt = dict(names=('idxs', 'importances'), formats=(np.int, np.float32))
        predictor_importances = np.array(list(aux), dtype=dt)

        output_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(self.output_path))))
        json_fname = os.path.join(output_dir, 'predictor_importances-fold-{}.json'.format(self.fold))
        print("-- saving predictors importances:", json_fname, flush=True)
        np.savetxt(json_fname, predictor_importances, fmt="%d,%.5f", delimiter=',')

        # Plot the feature importances of the forest
        importances = np.array(importances)
        methods_name = np.array(methods_name)
        top_k=40
        plt.clf()
        plt.figure(figsize=(15, 6), dpi=300)
        # import pdb; pdb.set_trace()
        title_font = {'size': '18', 'color': 'black'}
        plt.title("Feature importances", **title_font)
        plt.bar(range(top_k), importances[sorted_idxs[:top_k]], color="b", align="center")
        plt.xticks(range(top_k), methods_name[sorted_idxs[:top_k]], rotation='vertical')
        plt.xlim([-1, top_k])
        
        plt.xlabel('Features', fontsize=16)
        plt.ylabel('Importance', fontsize=16)

        plot_fname = os.path.join(output_dir, 'predictor_importances-fold-{}.png'.format(self.fold))
        plt.savefig(plot_fname, bbox_inches='tight')


    def compute_accuracies_for_original_classifiers(self, x_train, y_train, x_test, y_test, methods_name):

        baccs = []
        for col in range(x_train.shape[1]):
            baccs += [balanced_accuracy_score(y_train, x_train[:, col])]

        self.save_feature_importance(baccs, methods_name=methods_name)

    def run(self):
        """
        Start the classification step.
        """

        try:
            os.makedirs(self.output_path)
        except OSError:
            pass

        start = get_time()

        all_predictions, methods_name = self.load_predictions_v3()

        x_train, y_train, x_test, y_test = self.problem_reading(all_predictions)

        elapsed = total_time_elapsed(start, get_time())

        print('-- load predictions (time consumption): {0}!'.format(elapsed), flush=True)

        if self.step_algo == 'selection':
            print('-- Selection Step')

            if self.selection_algo == 1:
                self.run_training_test_protocol(x_train, y_train, x_test, y_test, methods_name)
            elif self.selection_algo == 2:
                self.compute_accuracies_for_original_classifiers(x_train, y_train, x_test, y_test, methods_name)

        elif self.step_algo == 'fusion':
            print('-- Fusion Step')
            start = get_time()

            # -- selecting the most promissing models
            print('-- selection_algo', self.selection_algo, flush=True)

            # if self.selection_algo == 0: # SVM classifier
            #     selected_models_idxs = self.select_classifiers_by_feature_importance(n_models=self.n_models)
            # elif self.selection_algo == 1: # RF classifier

            if 'scores' in self.meta_classification_from:
                complementary_method = 'kendalltau'
            else:
                complementary_method = 'kappa'

            selected_models_idxs = self.find_complementary_models(x_train,  n_models=self.n_models, method=complementary_method)
            # else:
            #     raise Exception("selection_algo not implemented!")

            x_train = x_train[:, selected_models_idxs]
            x_test = x_test[:, selected_models_idxs]

            selected_models = {}
            for a, fpred in enumerate(np.array(self.predictions_files)[selected_models_idxs]):
                selected_models.update({'%d'% a: fpred})
                print(a, fpred, flush=True)

            json_fname = os.path.join(self.output_path, 'selected_models_for_fusion.json')
            with open(json_fname, mode='w') as f:
                print("-- saving json file:", json_fname)
                sys.stdout.flush()
                f.write(json.dumps(selected_models, indent=4))

            elapsed = total_time_elapsed(start, get_time())
            print('-- select the most promissing models (time consumption): {0}!'.format(elapsed), flush=True)

            self.run_training_test_protocol(x_train, y_train, x_test, y_test, methods_name)

    @abstractmethod
    def training(self, x_train, y_train, x_validation=None, y_validation=None, methods_name=None):
        return NotImplemented

    @abstractmethod
    def testing(self, x_test, y_test):
        return NotImplemented
