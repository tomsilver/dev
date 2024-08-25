from settings import Settings
from features import load_features

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import time


def get_task_files():
    data_path = 'data/kaggle'
    training_path = os.path.join(data_path, 'training')
    evaluation_path = os.path.join(data_path, 'evaluation')
    training_tasks = sorted(os.listdir(training_path))
    evaluation_tasks = sorted(os.listdir(evaluation_path))
    train_task_files = [os.path.join(training_path, training_tasks[i]) for i in range(400)]
    evaluation_task_files = [os.path.join(evaluation_path, evaluation_tasks[i]) for i in range(400)]
    return train_task_files, evaluation_task_files

def load_all_grids(task_files):
    all_grids = []
    for task_file in task_files:
        with open(task_file, 'r') as f:
            task = json.load(f)
        num_inputs = len(task['train'])
        num_outputs = len(task['test'])
        for i in range(num_inputs):
            grid = task['train'][i]['input']
            all_grids.append(grid)
        for i in range(num_outputs):
            grid = task['test'][i]['input']
            all_grids.append(grid)
    return all_grids

def number_of_cells_in_task(task_file):
    grids = load_all_grids([task_file])
    num_cells = 0
    for grid in grids:
        num_cells += len(grid) * len(grid[0])
    return num_cells

def number_of_cells_distribution(task_files, outfile):
    xs = [number_of_cells_in_task(f) for f in task_files]
    assert max(xs) < 5000
    plt.figure()
    plt.hist(xs, bins=np.arange(0, 5001, 100))
    plt.ylabel("Number of tasks")
    plt.xlabel("Number of cells")
    plt.tight_layout()
    plt.savefig(outfile)
    print("Wrote out to {}.".format(outfile))

def number_of_cells_distribution_experiment():
    train_task_files, evaluation_task_files = get_task_files()

    number_of_cells_distribution(
        train_task_files[:10],
        "analysis/cell_distribution_train10.png"
    )

    number_of_cells_distribution(
        train_task_files,
        "analysis/cell_distribution_train400.png"
    )

    number_of_cells_distribution(
        evaluation_task_files,
        "analysis/cell_distribution_eval400.png"
    )

def running_programs_timing_experiment():
    Settings.use_cache = False
    Settings.do_collapse = False
    outfile_prefix = "no_cache_no_collapse"
    running_programs_timing_experiment_helper(outfile_prefix)

    Settings.use_cache = True
    Settings.do_collapse = False
    outfile_prefix = "yes_cache_no_collapse"
    running_programs_timing_experiment_helper(outfile_prefix)

    # Settings.use_cache = True
    # Settings.do_collapse = True
    # outfile_prefix = "yes_cache_yes_collapse"
    # running_programs_timing_experiment_helper(outfile_prefix)

def running_programs_timing_experiment_helper(outfile_prefix):
    train_task_files, _ = get_task_files()
    grids = load_all_grids(train_task_files[:10])
    cell_color_inputs = create_all_inputs(grids)
    programs, _ = load_features(num_programs=2500)

    timings_mat_outfile = "analysis/{}_timings_mat.npy".format(outfile_prefix)

    if os.path.exists(timings_mat_outfile):
        timings = np.load(timings_mat_outfile)
    else:
        # Build matrix where rows are programs, cols are (cell, color),
        # and entries are times
        timings = get_timing_matrix(programs, cell_color_inputs)
        np.save(timings_mat_outfile, timings)
        print("Wrote out to {}.".format(timings_mat_outfile))

    # convert from s to ms
    timings = 1000*timings

    dsl_primitives, dsl_timings = get_dsl_primitive_timings(programs, timings)

    print("Total time for running programs ({}):".format(outfile_prefix))
    print("{} ms".format(timings.sum()))

    print("Average time for running all programs on 1 input ({}):".format(outfile_prefix))
    print("{} ms".format(timings.mean(axis=1).sum()))

    print("Average time for running 1 program on {} input ({}):".format(len(cell_color_inputs), outfile_prefix))
    print("{} ms".format(timings.mean(axis=0).sum()))

    print("Average time for running 1 program on 1 input ({}):".format(outfile_prefix))
    print("{} ms".format(timings.mean().mean()))

    outfile = "analysis/{}_dsl_prims.png".format(outfile_prefix)
    plt.figure()
    plt.bar(np.arange(len(dsl_primitives)), dsl_timings.mean(axis=1))
    plt.xticks(np.arange(len(dsl_primitives)), dsl_primitives, rotation='vertical')
    plt.ylabel("Average runtime per input (ms)")
    plt.tight_layout()
    plt.savefig(outfile)
    print("Wrote out to {}.".format(outfile))

    outfile = "analysis/{}_all_programs.png".format(outfile_prefix)
    plt.figure()
    plt.bar(np.arange(len(programs)), timings.mean(axis=1))
    plt.xlabel("Program")
    plt.ylabel("Average runtime per input (ms)")
    plt.tight_layout()
    plt.savefig(outfile)
    print("Wrote out to {}.".format(outfile))

def create_all_inputs(grids):
    all_inputs = []
    for grid in grids:
        grid = np.array(grid)
        grid_cache = {}
        for r in range(len(grid)):
            for c in range(len(grid[r])):
                for color in range(10):
                    cache = {'cell' : {}, 'grid' : grid_cache}
                    raw_x = (grid, r, c, color, cache)
                    all_inputs.append(raw_x)
    return all_inputs

def get_timing_matrix(programs, cell_color_inputs):
    timings = np.zeros((len(programs), len(cell_color_inputs)))
    for i, program in enumerate(programs):
        for j, raw_x in enumerate(cell_color_inputs):
            start_time = time.time()
            program(raw_x)
            duration = time.time() - start_time
            timings[i, j] = duration
    return timings

def get_dsl_primitive_timings(programs, timings):
    dsl_primitives = get_dsl_primitives()

    dsl_total_timings = np.zeros((len(dsl_primitives), timings.shape[1]))
    dsl_counts = np.zeros((len(dsl_primitives), timings.shape[1]))

    for i, dsl_primitive in enumerate(dsl_primitives):
        for j, program in enumerate(programs):
            if dsl_primitive in str(program):
                dsl_total_timings[i] += timings[j]
                dsl_counts[i] += 1

    dsl_timings = dsl_total_timings / dsl_counts

    return dsl_primitives, dsl_timings

def get_dsl_primitives():
    return ['compare_colors', 'search_for_color', 'inside', 'row_is_divisible_by', 
        'col_is_divisible_by', 'get_color_from_action', 'get_color_from_cell',
        'most_common_color', 'least_common_color', 'candidate_cell', 
        'neighbor_cell', 'mod_candidate_cell', 'mod_neighbor_cell']

if __name__ == "__main__":
    # number_of_cells_distribution_experiment()
    running_programs_timing_experiment()

