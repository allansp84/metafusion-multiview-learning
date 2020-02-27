# -*- coding: utf-8 -*-

import json

from antispoofing.mcnns.utils import *
from antispoofing.mcnns.metaclassification.basemetaclassifier import BaseMetaClassifier

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def build_grid_search(n_jobs):

    # -- build set the parameters for grid search
    log2c = np.logspace(-5,  20, 20, base=2).tolist()
    log2g = np.logspace(-15, 5, 20, base=2).tolist()

    search_space = [{'kernel': ['rbf'], 'gamma':log2g, 'C':log2c, 'class_weight':['balanced']}]
    # search_space += [{'kernel': ['linear'], 'C':log2c, 'class_weight':['balanced']}]
    # search_space = [{'kernel': ['rbf'], 'C': log2c, 'class_weight': ['balanced']}]

    k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(SVC(random_state=42), search_space, 
                               cv=2, scoring='balanced_accuracy', n_jobs=n_jobs)

    return grid_search


def one_ova(x_train, y_train, cat, n_jobs):
    lcat = np.zeros(y_train.size)

    lcat[y_train != cat] = -1
    lcat[y_train == cat] = +1

    # clf = SVC(kernel='linear', C=1e5, class_weight='balanced', random_state=42)
    clf = build_grid_search(n_jobs)
    clf.fit(x_train, lcat)

    return clf


class MetaSVM(BaseMetaClassifier):
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

        super(MetaSVM, self).__init__(output_path, predictions_files, gt_files, meta_classification_from, n_models,
                                      step_algo=step_algo,
                                      selection_algo=selection_algo,
                                      compute_feature_importance=compute_feature_importance,
                                      fold=fold,
                                      n_jobs=n_jobs,
                                      )

        self.verbose = True
        self.output_path = output_path
        self.force_train = force_train

        self.output_model = os.path.join(self.output_path, "meta_svm_classifier_model.pkl")
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
        self.model = one_ova(x_train, y_train, 1, self.n_jobs)

    def training(self, x_train, y_train, x_validation=None, y_validation=None, methods_name=None):
        """ This method implements the training process of our CNN.

        Args:
            x_train (numpy.ndarray): Training data
            y_train (numpy.ndarray): Labels of the training data
            x_validation (numpy.ndarray, optional): Testing data. Defaults to None.
            y_validation (numpy.ndarray, optional): Labels of the testing data. Defaults to None.

        """

        if self.force_train or not os.path.exists(self.output_model):
            print('-- training ...', flush=True)

            # -- fit the model
            self.fit_model(x_train, y_train)

            # -- save the fitted model
            print("-- saving model", self.output_model, flush=True)
            save_object(self.model, self.output_model)

            svm_params_fname = os.path.join(self.output_path, 'svm_params.json')
            with open(svm_params_fname, mode='w') as f:
                print("-- saving json file:", svm_params_fname, flush=True)
                f.write(json.dumps(self.model.best_params_, indent=4))


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
            predicted_scores = self.model.decision_function(x_test)
            # predicted_scores = self.model.predict_proba(x_test)[:, 1]

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
