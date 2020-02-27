# -*- coding: utf-8 -*-

import json

from antispoofing.mcnns.utils import *
from antispoofing.mcnns.metaclassification.basemetaclassifier import BaseMetaClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def build_grid_search(x_train, lcat, n_jobs):

    # -- build set the parameters for grid search
    search_space = {"n_estimators": [10], # [3, 5, 7, 10, 15, 20, 25]
                    "criterion": ["gini", "entropy"],
                    "max_depth": [3, 5, 7, 10, 15, 20, None],
                    "min_samples_split": [2, 3, 5, 7, 10, 0.5, 1.0],
                    "min_samples_leaf": [1, 3, 5, 7, 10],
                    "max_features": ['auto'],
                    "max_leaf_nodes": [None],
                    "bootstrap": [True],
                    }

    k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), search_space,
                               cv=k_fold, scoring='balanced_accuracy', n_jobs=n_jobs)

    grid_search.fit(x_train, lcat)

    return grid_search


def one_ova(x_train, y_train, cat, n_jobs):
    lcat = np.zeros(y_train.size)

    lcat[y_train != cat] = -1
    lcat[y_train == cat] = +1

    clf = build_grid_search(x_train, lcat, n_jobs)
    best_params = clf.best_params_

    print('-- best_params:', best_params, flush=True)

    clf = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                 criterion=best_params['criterion'],
                                 max_depth=best_params['max_depth'],
                                 min_samples_split=best_params['min_samples_split'],
                                 min_samples_leaf=best_params['min_samples_split'],
                                 max_features=best_params['max_features'],
                                 max_leaf_nodes=best_params['max_leaf_nodes'],
                                 bootstrap=best_params['bootstrap'],
                                 n_jobs=n_jobs,
                                 random_state=42,
                                 class_weight='balanced',
                                 )
    clf.fit(x_train, lcat)

    return clf


class MetaRandomForest(BaseMetaClassifier):
    """ This class implements a detector for iris-based spoofing using a shallow Convolutional Neural Network (CNN).

    In this class, we define the architecture of our model and we implement the learning stage and the testing stage. Due to the huge
    amount of options available to implement the learning stage, we decide to parametizer the main components of this stage in order to
    find the best learning parameters for achieving a good classification result. In this way, for instance, it's possible to choose the
    loss function, the optimizer, so on, using the command line interface.

    Args:
        output_path (str):
        dataset (Dataset):
        dataset_b (Dataset):
        input_shape (int): Defaults to 200.
        epochs (int): Defaults to 50.
        batch_size (int): Defaults to 8.
        loss_function (str): Defaults to 'categorical_crossentropy'.
        lr (float): Defaults to 0.01.
        decay (float): Defaults to 0.0005.
        optimizer (str): Defaults to 'SGD'.
        regularization (float): Defaults to 0.1.
        device_number (int): Defaults to 0.
        force_train (bool): Defaults to False.
        filter_vis (bool): Defaults to False.
        layers_name (tuple): Defaults to ('conv_1',).
        fold (int):(default: 0)
    """

    def __init__(self, output_path, predictions_files, gt_files, meta_classification_from, n_models,
                 step_algo='selection', selection_algo=0, compute_feature_importance=False, force_train=False, fold=1, n_jobs=-1):

        super(MetaRandomForest, self).__init__(output_path, predictions_files, gt_files, meta_classification_from, n_models,
                                               step_algo=step_algo,
                                               selection_algo=selection_algo,
                                               compute_feature_importance=compute_feature_importance,
                                               fold=fold,
                                               n_jobs=n_jobs,
                                               )

        self.verbose = True
        self.output_path = output_path
        self.compute_feature_importance = compute_feature_importance
        self.force_train = force_train

        self.output_model = os.path.join(self.output_path, "meta_rf_classifier_model.pkl")
        self.output_weights = os.path.join(self.output_path, "weights.hdf5")
        self.model = None
        self.fold = fold

    def fit_model(self, x_train, y_train, x_validation=None, y_validation=None, class_weights=None):
        """ Fit a model classification.

        Args:
            x_train (numpy.ndarray): A multidimensional array containing the feature vectors (or images) to be used to train a classifier.
            y_train (numpy.ndarray): A multidimensional array containing the labels refers to the feature vectors that will be used during the training stage.
            x_validation (numpy.ndarray, optional): A multidimensional array containing the feature vectors (or images) to be used to test the classifier.
            y_validation (numpy.ndarray, optional): A multidimensional array containing the labels refers to the feature vectors that will be used for testing the classification model.
            class_weights (dict): A dictionary containig class weights for unbalanced datasets.
        """

        # -- fit a model
        self.model = one_ova(x_train, y_train, 1, self.n_jobs)

    def compute_and_save_feature_importance(self, n_features=0, methods_name=None):

        importances = self.model.feature_importances_.tolist()
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
        top_k=30
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


    def training(self, x_train, y_train, x_validation=None, y_validation=None, methods_name=None):
        """ This method implements the training process of our CNN.

        Args:
            x_train (numpy.ndarray): Training data
            y_train (numpy.ndarray): Labels of the training data
            x_validation (numpy.ndarray, optional): Testing data. Defaults to None.
            y_validation (numpy.ndarray, optional): Labels of the testing data. Defaults to None.

        """

        if self.force_train or not os.path.exists(self.output_model):
            print('-- training ...')
            sys.stdout.flush()

            # -- fit the model
            self.fit_model(x_train, y_train)

            if self.compute_feature_importance:
                print('-- computing the feature importance ...')
                self.compute_and_save_feature_importance(n_features=x_train.shape[1], methods_name=methods_name)

            # -- save the fitted model
            print("-- saving model", self.output_model)
            sys.stdout.flush()

            save_object(self.model, self.output_model)

        else:
            print('-- model already exists in', self.output_model)
            sys.stdout.flush()

    def testing(self, x_test, y_test):
        """ This method is responsible for testing the fitted model.

        Args:
            x_test (numpy.ndarray): Testing data
            y_test (numpy.ndarray): Labels of the Testing data

        Returns:
            dict: A dictionary with the ground-truth, the predicted scores and the predicted labels for the testing data, such as {'gt': y_test, 'predicted_labels': y_pred, 'predicted_scores': y_scores}

        """

        self.model = load_object(self.output_model)

        # -- get the predicted scores and labels for the testing data
        if self.model:

            predicted_scores = self.model.predict_proba(x_test)[:, 1]

            predicted_labels = self.model.predict(x_test)
            predicted_labels[predicted_labels != 1] = 0

        else:
            sys.exit('-- model not found! Please, execute the training again!')

        # -- define the output dictionary
        r_dict = {'gt': y_test,
                  'predicted_labels': predicted_labels,
                  'predicted_scores': predicted_scores,
                  }

        return r_dict
