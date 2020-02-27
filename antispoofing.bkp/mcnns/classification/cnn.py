# -*- coding: utf-8 -*-

import json

from antispoofing.mcnns.utils import *
from antispoofing.mcnns.classification.baseclassifier import BaseClassifier
from sklearn.utils import class_weight
# from vis.utils import utils
# from vis.visualization import visualize_activation
# from vis.visualization import visualize_saliency


class CNN(BaseClassifier):
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

    def __init__(self, output_path, dataset, dataset_b=None, input_shape=200, epochs=50,
                 batch_size=8, loss_function='categorical_crossentropy', lr=0.01, decay=0.0005, optimizer='SGD', regularization=0.1,
                 device_number=0, force_train=False, filter_vis=False, layers_name=('conv_1',),
                 fold=0):

        super(CNN, self).__init__(output_path, dataset, dataset_b=dataset_b, fold=fold)

        self.verbose = True

        self.dataset = dataset
        self.output_path = output_path
        self.output_model = os.path.join(self.output_path, "full_model.hdf5")
        self.output_weights = os.path.join(self.output_path, "weights.hdf5")
        self.model = None

        self.input_shape = (input_shape, input_shape, 1)
        self.num_classes = 2
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.lr = lr
        self.decay = decay
        self.optimizer = optimizer
        self.regularization = regularization
        self.device_number = device_number
        self.force_train = force_train
        self.filter_vis = filter_vis
        self.layers_name = list(layers_name)

    def set_gpu_configuration(self):
        """ This function is responsible for setting up which GPU will be used during the processing and some configurations related
        to GPU memory usage when the TensorFlow is used as backend.
        """

        if 'tensorflow' in keras.backend.backend():
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device_number

            K.clear_session()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9,
                                        # allow_growth=True,
                                        allocator_type='BFC',
                                        )
            K.set_session(K.tf.Session(config=K.tf.ConfigProto(gpu_options=gpu_options,
                                                               allow_soft_placement=True,
                                                               log_device_placement=True)))

    def architecture_definition(self):
        """
        In this method we define the architecture of our CNN.
        """

        img_input = Input(shape=self.input_shape, name='input_1')

        # -- first layer
        conv2d_1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu', name='conv_1')(img_input)
        max_pooling_1 = MaxPooling2D(pool_size=(9, 9), strides=(2, 2), name='pool_1')(conv2d_1)
        batch_norm_1 = BatchNormalization(momentum=0.9, epsilon=1e-3, name='batch_norm_1')(max_pooling_1)

        # -- second layer
        conv2d_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', name='conv_2')(batch_norm_1)
        max_pooling_2 = MaxPooling2D(pool_size=(9, 9), strides=(8, 8), name='pool_2')(conv2d_2)
        batch_norm_2 = BatchNormalization(momentum=0.9, epsilon=1e-3, name='batch_norm_2')(max_pooling_2)

        # -- fully-connected layer
        flatten_1 = Flatten(name='flatten')(batch_norm_2)
        dense_1 = Dense(units=1024, activation='relu', name='fc1')(flatten_1)

        # -- classification block
        output = Dense(self.num_classes, activation='softmax', name='predictions',
                       kernel_regularizer=keras.regularizers.l2(self.regularization))(dense_1)

        self.model = keras.models.Model(img_input, output, name='mcnn')

        if self.verbose:
            print(self.model.summary())

        # -- saving the CNN architecture definition in a .json file
        model_json = json.loads(self.model.to_json())
        json_fname = os.path.join(self.output_path, 'model.json')
        with open(json_fname, mode='w') as f:
            print("--saving json file:", json_fname)
            sys.stdout.flush()
            f.write(json.dumps(model_json, indent=4))

    def saving_training_history(self, history):
        """ Saving the plot containg the training history.

        Args:
            history (dict): A dictionary containing the values of accuracy and losses obtainied in each epoch of the learning stage.

        """

        # -- save the results obtained during the training process
        json_fname = os.path.join(self.output_path, 'training.history.json')
        with open(json_fname, mode='w') as f:
            print("--saving json file:", json_fname)
            sys.stdout.flush()
            f.write(json.dumps(history.history, indent=4))

        output_history = os.path.join(self.output_path, 'training.history.png')
        fig1 = plt.figure(figsize=(8, 6), dpi=100)
        title_font = {'size': '18', 'color': 'black', 'weight': 'normal', 'verticalalignment': 'bottom'}
        axis_font = {'size': '14'}
        font_size_axis = 12
        title = "Training History"

        plt.clf()
        plt.plot(range(1, len(history.history['acc']) + 1), history.history['acc'], color=(0, 0, 0), marker='o', linestyle='-', linewidth=2,
                 label='train')
        plt.plot(range(1, len(history.history['val_acc']) + 1), history.history['val_acc'], color=(0, 1, 0), marker='s', linestyle='-',
                 linewidth=2, label='test')

        plt.xlabel('Epochs', **axis_font)
        plt.ylabel('Accuracy', **axis_font)

        plt.xticks(size=font_size_axis)
        plt.yticks(size=font_size_axis)

        plt.legend(loc='upper left')
        plt.title(title, **title_font)
        plt.grid(True)

        fig1.savefig(output_history)

    def fit_model(self, x_train, y_train, x_validation=None, y_validation=None, class_weights=None):
        """ Fit a model classification.

        Args:
            x_train (numpy.ndarray): A multidimensional array containing the feature vectors (or images) to be used to train a classifier.
            y_train (numpy.ndarray): A multidimensional array containing the labels refers to the feature vectors that will be used during the training stage.
            x_validation (numpy.ndarray, optional): A multidimensional array containing the feature vectors (or images) to be used to test the classifier.
            y_validation (numpy.ndarray, optional): A multidimensional array containing the labels refers to the feature vectors that will be used for testing the classification model.
            class_weights (dict): A dictionary containig class weights for unbalanced datasets.
        """

        # -- configure the GPU that will be used
        self.set_gpu_configuration()

        # -- define the architecture
        self.architecture_definition()

        # -- choose the optimizer that will be used during the training process
        optimizer_methods = {'SGD': keras.optimizers.SGD,
                             'Adam': keras.optimizers.Adam,
                             'Adagrad': keras.optimizers.Adagrad,
                             'Adadelta': keras.optimizers.Adadelta,
                             }

        try:
            opt = optimizer_methods[self.optimizer]
        except KeyError:
            raise Exception('The optimizer %s is not being considered in this work yet:' % self.optimizer)

        # --  configure the learning process
        self.model.compile(loss=self.loss_function, optimizer=opt(lr=self.lr, decay=self.decay), metrics=['accuracy'])

        # -- define the callbacks
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=20, verbose=0, mode='auto'),
                     ]

        # -- fit a model
        history = self.model.fit(x_train, y_train,
                                 batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                                 callbacks=callbacks,
                                 validation_data=(x_validation, y_validation),
                                 shuffle=True,
                                 class_weight=class_weights,
                                 )

        # -- save the training history
        self.saving_training_history(history)

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

            # -- compute the class weights for unbalanced datasets
            class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

            # -- convert class vectors to binary class matrices.
            y_train = keras.utils.to_categorical(y_train, self.num_classes)
            if y_validation is not None:
                y_validation = keras.utils.to_categorical(y_validation, self.num_classes)

            # -- fit the model
            self.fit_model(x_train, y_train, x_validation=x_validation, y_validation=y_validation, class_weights=class_weights)

            # -- save the fitted model
            print("-- saving model", self.output_model)
            sys.stdout.flush()

            self.model.save(self.output_model)
            self.model.save_weights(self.output_weights)
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

        # -- configure the GPU that will be used
        self.set_gpu_configuration()

        # -- load the fitted model
        self.model = keras.models.load_model(self.output_model)

        # -- generates output predictions for the testing data.
        scores = self.model.predict(x_test, batch_size=self.batch_size, verbose=0)

        # -- get the predicted scores and labels for the testing data
        y_pred = np.argmax(scores, axis=1)
        y_scores = scores[:, 1]

        # -- define the output dictionary
        r_dict = {'gt': y_test,
                  'predicted_labels': y_pred,
                  'predicted_scores': y_scores,
                  }

        return r_dict
