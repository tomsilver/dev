from features import load_features
from model import OutputPixelColorClassifier, OutputShapePredictor, OutputPixelColorClassifierModel
from structs import Feature
from util import plot_task

import numpy as np

import os
import json
import time



def run(model, task_files, prediction_dir='predictions', load_predictions=False,
        visualize='none', visualize_outdir='out'):
    """Fit and make predictions on a set of tasks.

    Parameters
    ----------
    model : OutputPixelColorClassifierModel
    task_files : [ str ]
    load_predictions : bool
        If True, don't run the model, just load previous predictions.
        Useful for visualizing after the fact.
    prediction_dir : str
        Where the predictions will get saved.
    visualize : str
        One of "all", "none", "correct_only", or "incorrect_only".
    visualize_outdir : str
        Where to save visualizations.

    Returns
    -------
    prediction_files : [ str ]
        File paths for saved predictions.
    """
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    num_correct = 0
    num_total = 0
    prediction_files = []
    all_correct_jsons = []
    overfit_jsons = []

    for task_file in task_files:
        with open(task_file, 'r') as f:
            task = json.load(f)

        # Number of test input/output pairs (typically only 1)
        num_outputs = len(task['test'])

        task_json_name = os.path.split(task_file)[-1]
        prediction_outfile = "{}/{}".format(prediction_dir, task_json_name)

        # Load or make predictions
        if load_predictions:
            task_predictions = [task['test'][i] for i in range(num_outputs)]

        else:

            # Fit the model
            score = model.fit(task['train'])
            print("score:", score)

            # print(model)

            # Make predictions
            task_predictions = []
            for i in range(num_outputs):
                
                # Make prediction for one test input
                matrix_predictions = model.predict(task['test'][i]['input'])
                
                # Convert to lists for pickling
                matrix_predictions = [prediction.tolist() for prediction in matrix_predictions]

                task_predictions.append(matrix_predictions)

                # Add predictions to json for easy future visualizing
                task['test'][i]['predictions'] = matrix_predictions

            # Save task predictions
            with open(prediction_outfile, 'w') as f:
                json.dump(task, f)
            prediction_files.append(prediction_outfile)

        # Score predictions for task

        task_correct = True
        # For each test input matrix (typically there's only 1)
        for i in range(num_outputs):

            # A model outputs top k predictions. The task counts as correct if
            # any of the top k are correct.
            predictions_correct = [True for _ in range(model.num_predictions)]

            # Check if each prediction is correct
            for j, prediction in enumerate(task_predictions[i]):
                if np.any(prediction != task['test'][i]['output']):
                    predictions_correct[j] = False

            if not any(predictions_correct):
                task_correct = False

        num_correct += task_correct
        num_total += 1
        print("Num correct: {} / {}".format(num_correct, num_total)) #, end='\r')

        if (score == 1.0) and (not task_correct):
            overfit_jsons.append(task_json_name)

        if task_correct:
            print("\n\n** Correct task: {}".format(task_json_name))
            print("\nModel: {}".format(str(model)))
            all_correct_jsons.append(task_json_name)

        if visualize == 'all' or (visualize == 'correct_only' and task_correct) or \
            (visualize == 'incorrect_only' and not task_correct):
            visualize_outprefix = "{}/{}".format(visualize_outdir, task_json_name.split('.')[0])
            plot_task(task, visualize_outprefix)

    print("Num correct: {} / {}".format(num_correct, num_total))
    print("All correct jsons:", all_correct_jsons)
    return prediction_files


def build_model(features, log_priors, hyperparameters):
    osp = OutputShapePredictor()
    oppc = OutputPixelColorClassifier(features, log_priors, hyperparameters)
    return OutputPixelColorClassifierModel(oppc, osp, num_predictions=len(hyperparameters))


def main(visualize='none'):
    start_time = time.time()

    max_num_programs = 2500

    hyperparameters = [
        # {'num_programs' : 100, 'max_depth' : 10, 'num_seeds' : 10},
        # {'num_programs' : 1000, 'max_depth' : 10, 'num_seeds' : 10},
        {'num_programs' : max_num_programs, 'max_depth' : None, 'num_seeds' : 10}
    ]

    data_path = 'data/kaggle'
    # data_path = 'data/debug'
    training_path = os.path.join(data_path, 'training')
    # evaluation_path = os.path.join(data_path, 'evaluation')

    training_tasks = sorted(os.listdir(training_path))
    # evaluation_tasks = sorted(os.listdir(evaluation_path))
    task_files = [os.path.join(training_path, training_tasks[i]) for i in range(400)][:10]
    # task_files = [os.path.join(evaluation_path, evaluation_tasks[i]) for i in range(400)]
    # select_task_jsons = ['00d62c1b.json', '25ff71a9.json', '3618c87e.json', '4347f46a.json', 'e9afcf9a.json',
        # '4258a5f9.json', '6f8cd79b.json', '72322fa7.json', 'bb43febb.json']
    # select_task_jsons = ['00d62c1b.json'] #['0d3d703e.json'] #, '4258a5f9.json', '0ca9ddb6.json']
    # task_files = [os.path.join(training_path, p) for p in select_task_jsons]

    # main stuff
    features, log_priors = load_features(num_programs=max_num_programs)
    model = build_model(features, log_priors, hyperparameters)
    prediction_files = run(model, task_files, visualize=visualize)

    print("Finished running. Total time: {}s".format(time.time() - start_time))


if __name__ == "__main__":
    main(visualize='none')

