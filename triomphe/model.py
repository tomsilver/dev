from dt_utils import extract_lpp_from_dt
from sklearn.tree import DecisionTreeClassifier, plot_tree
# from dt import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np



class OutputPixelColorClassifierModel:
    """This model predicts full output grids from input grids.

    Parameters
    ----------
    classifier : OutputPixelColorClassifier
    output_shape_predictor : OutputShapePredictor
    num_predictions : int
        The number of predictions to make per test input matrix.
        For kaggle, this should be 3. For development, it's often
        easier to just do 1.
    seed : int
        Random seed.
    """
    _colors = list(range(10))

    def __init__(self, classifier, output_shape_predictor, num_predictions, seed=0):
        self._classifier = classifier
        self._output_shape_predictor = output_shape_predictor
        self.num_predictions = num_predictions
        self._rng = np.random.RandomState(seed)

    def __str__(self):
        return str(self._classifier)

    def fit(self, train_task):
        """
        Parameters
        ----------
        train_task : dict
            An ARC task, as loaded from the json.
        """
        # Keep track of input-output grid shapes to fit output_shape_predictor
        input_grid_shapes, output_grid_shapes = [], []

        # Keep track of training data to fit _classifier
        # X will consist of (grid, r, c, color, cache)
        # Y is binary
        X, Y = [], []

        # Go through each of the input-output training matrices
        for train_task_i in train_task:
            
            # Convert to numpy arrays to make some DSL functions more efficient
            grid = np.array(train_task_i['input'])
            output_matrix = np.array(train_task_i['output'])

            # Keep a cache for this training task. Example: we only want to
            # do one "inside" calculation for the whole input grid; we don't
            # want to redo it for every cell. The DSL functions are allowed
            # to access and update this cache.
            grid_cache = {}
            for r in range(output_matrix.shape[0]):
                for c in range(output_matrix.shape[1]):
                    valid_color = output_matrix[r, c]
                    for color in self._colors:
                        # Keep a cache for this cell. Same motivation as grid
                        # cache, but sometimes we do need to compute per-cell.
                        cache = {'cell' : {}, 'grid' : grid_cache}
                        raw_x = (grid, r, c, color, cache)
                        if color == valid_color:
                            y = 1
                        else:
                            y = 0
                        X.append(raw_x)
                        Y.append(y)
            input_grid_shapes.append(grid.shape)
            output_grid_shapes.append(output_matrix.shape)

        # Fit the main classifier
        score = self._classifier.fit(X, Y)

        # Fit the output shape predictor
        self._output_shape_predictor.fit(input_grid_shapes, output_grid_shapes)

        return score

    def predict(self, grid):
        """Predict top-k (self.num_predictions) output grids given the input grid.

        Must call fit before predict.

        Parameters
        ----------
        grid : [[ int ]]
            A test input grid.

        Returns
        -------
        output_grids : [ [[ int ]] ]
            Top-k output grid predictions.
        """
        grid = np.array(grid)

        # First predict the output shape and initialize black grids
        output_shape = self._output_shape_predictor.predict(grid.shape)
        output_grids = [np.zeros(output_shape, dtype=grid.dtype) for _ in range(self.num_predictions)]

        # See self.fit for cache comment
        grid_cache = {}
        for r in range(output_shape[0]):
            for c in range(output_shape[1]):
                valid_colors = [[] for _ in range(self.num_predictions)]
                for color in self._colors:
                    # See self.fit for cache comment
                    cache = {'cell' : {}, 'grid' : grid_cache}
                    raw_x = (grid, r, c, color, cache)
                    # The classifier returns top-k predictions
                    predictions = self._classifier.predict(raw_x)
                    for i, prediction in enumerate(predictions):
                        if prediction:
                            valid_colors[i].append(color)

                for i, valid_colors_i in enumerate(valid_colors):
                    if len(valid_colors_i) > 0:
                        selected_color = self._rng.choice(valid_colors_i)
                        output_grids[i][r, c] = selected_color

        return output_grids


class OutputPixelColorClassifier:
    """A binary classifier of (input_grid, row, column, color).

    Parameters
    ----------
    features : [ Feature ]
    log_priors : [ float ]
    hyperparameters : [ dict ]
        One set of hyperparameters for each top-k prediction.
    init_seed : int
    """
    def __init__(self, features, log_priors, hyperparameters, init_seed=0):
        max_num_features = max([h['num_programs'] for h in hyperparameters])
        features = features[:max_num_features]
        log_priors = log_priors[:max_num_features]

        self._features = features
        self._log_priors = log_priors
        self._init_seed = init_seed
        self._hyperparameters = hyperparameters
        print("Initialized OPPC with {} features".format(len(features)))

    def __str__(self):
        return str(self._lpps)

    def fit(self, raw_X, Y):
        """
        Parameters
        ----------
        raw_X : [ (input_grid, row, col, color, cache) ]
        Y : [ 0 or 1 ]
        """
        # This is the most expensive part!
        all_features_X, kept_feature_idxs = self._preprocess_inputs_train(raw_X)
        self._features = [self._features[idx] for idx in kept_feature_idxs]
        self._log_priors = [self._log_priors[idx] for idx in kept_feature_idxs]

        # Top-k classifiers
        self._clfs = []
        # Top-k logical program policies
        self._lpps = []

        max_score = 0.

        # Make top-k predictions (one per hyperparameter set)
        for hyperparameter_set in self._hyperparameters:
            max_depth = hyperparameter_set['max_depth']
            num_seeds = hyperparameter_set['num_seeds']
            num_programs = hyperparameter_set['num_programs']

            X = [x[:num_programs] for x in all_features_X]

            clf = DecisionTreeClassifier(random_state=self._init_seed, max_depth=max_depth)
            clf.fit(X, Y)

            score = clf.score(X, Y)
            # print("score:", score)
            max_score = max(score, max_score)
            # if score == 1.0:
                # print("Found a fitting tree with {} features".format(num_programs))
                # break

            ## TODO check whether this actually makes any difference...
            # print("Trying to find tree with better prior...")
            best_log_prior = -np.inf
            best_clf = None
            best_lpp = None
            for seed_offset in range(num_seeds):
                if seed_offset > 0:
                    random_state = self._init_seed+seed_offset
                    clf = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)
                    clf.fit(X, Y)
                lpp, log_prior = extract_lpp_from_dt(clf, self._features, self._log_priors)
                # print("log_prior:", log_prior)
                if log_prior > best_log_prior:
                    best_log_prior = log_prior
                    best_clf = clf
                    best_lpp = lpp

            # best_clf = clf
            # best_lpp = 0.

            self._clfs.append(best_clf)
            self._lpps.append(best_lpp)

        # plt.figure()
        # plot_tree(self._clf)
        # plt.savefig('/tmp/dt.png')
        return max_score
    
    def _preprocess_inputs(self, raw_X):
        X = [[] for _ in raw_X]
        for i, raw_x in enumerate(raw_X):
            for j, feature in enumerate(self._features):
                # print("running feature {}".format(i))
                x_ij = feature(raw_x)
                X[i].append(x_ij)
        return X

    def _preprocess_inputs_train(self, raw_X):
        XT = []
        unique_patterns = set()
        tries = 0
        kept_feature_idxs = []
        for idx, feature in enumerate(self._features):
            tries += 1
            outs = []
            for raw_x in raw_X:
                out = feature(raw_x)
                outs.append(out)
            tup_outs = tuple(outs)
            if True: #tup_outs not in unique_patterns:
                unique_patterns.add(tup_outs)
                XT.append(outs)
                kept_feature_idxs.append(idx)
        # print("KEPT {}/{} FEATURES AFTER COLLAPSING".format(len(kept_feature_idxs), tries))
        return np.transpose(XT), kept_feature_idxs

    def predict(self, raw_x):
        all_features_X = self._preprocess_inputs([raw_x])

        yhats = []

        for i, hyperparameter_set in enumerate(self._hyperparameters):
            num_programs = hyperparameter_set['num_programs']
            X = [x[:num_programs] for x in all_features_X]
            yhat = self._clfs[i].predict(X)[0]
            yhats.append(yhat)

        return yhats



class OutputShapePredictor:

    def fit(self, input_shapes, output_shapes):
        input_shape, output_shape = input_shapes[0], output_shapes[0]
        self._height_scale = output_shape[0] / input_shape[0]
        self._width_scale = output_shape[1] / input_shape[1]

    def predict(self, input_shape):
        height, width = input_shape
        pred_height = max(1, int(height * self._height_scale))
        pred_width = max(1, int(width * self._width_scale))
        return (pred_height, pred_width)

