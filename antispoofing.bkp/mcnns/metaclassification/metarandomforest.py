# -*- coding: utf-8 -*-

import json

from antispoofing.mcnns.utils import *
from antispoofing.mcnns.metaclassification.basemetaclassifier import BaseMetaClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def grid_search(x_train, lcat):

    # -- build set the parameters for grid search
    search_space = {"n_estimators": [10],  # [1, 3, 5, 10, 50],
                    "criterion": ["gini", "entropy"],
                    "max_depth": [3, 5, None],
                    "min_samples_split": [2, 3, 5, 10, 0.5, 1.0],
                    "min_samples_leaf": [1, 3, 5, 10],
                    "max_features": ['auto'],
                    "max_leaf_nodes": [None],
                    "bootstrap": [True],
                    }

    k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), search_space,
                               cv=k_fold, scoring='roc_auc', n_jobs=-1)

    grid_search.fit(x_train, lcat)

    return grid_search


def one_ova(x_train, y_train, cat):
    lcat = np.zeros(y_train.size)

    lcat[y_train != cat] = -1
    lcat[y_train == cat] = +1

    clf = grid_search(x_train, lcat)
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
                                 n_jobs=-1,
                                 random_state=42,
                                 # class_weight='balanced',
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

    def __init__(self, output_path, predictions_files, meta_classification_from, n_models,
                 selection_algo=0, compute_feature_importance=False, force_train=False):

        super(MetaRandomForest, self).__init__(output_path, predictions_files, meta_classification_from, n_models,
                                               selection_algo=selection_algo,
                                               compute_feature_importance=compute_feature_importance,
                                               )

        self.verbose = True
        self.output_path = output_path
        self.compute_feature_importance = compute_feature_importance
        self.force_train = force_train

        self.output_model = os.path.join(self.output_path, "meta_rf_classifier_model.pkl")
        self.output_weights = os.path.join(self.output_path, "weights.hdf5")
        self.model = None

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
        self.model = one_ova(x_train, y_train, 1)

    def compute_and_save_feature_importance(self):

        importance = self.model.feature_importances_.tolist()
        idxs = np.arange(len(importance))

        aux = zip(idxs, importance)

        dt = dict(names=('idxs', 'importance'), formats=(np.int, np.float32))
        predictor_importances = np.array(list(aux), dtype=dt)

        json_fname = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(self.output_path))), 'predictor_importances.json')
        print("-- saving predictors importance:", json_fname, flush=True)
        np.savetxt(json_fname, predictor_importances, fmt="%d,%.5f", delimiter=',')


    def training(self, x_train, y_train, x_validation=None, y_validation=None):
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
                self.compute_and_save_feature_importance()

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
