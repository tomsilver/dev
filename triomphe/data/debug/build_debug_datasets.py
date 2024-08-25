import os
import json
import numpy as np

rng = np.random.RandomState(0)

def build_task(train, test, task_name, subdir='training'):
    task = {'train' : train, 'test' : test}

    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(dir_path, subdir, task_name + '.json')

    with open(filename, 'w') as fp:
        json.dump(task, fp)

    print("Dumped task to {}.".format(filename))

def build_identity_task():
    num_train_input = 4
    num_test_output = 1

    train = []
    for _ in range(num_train_input):
        tinput = rng.randint(9, size=(5, 5))
        toutput = tinput.copy()
        train.append({'input' : tinput.tolist(), 'output' : toutput.tolist()})

    test = []
    for _ in range(num_test_output):
        tinput = rng.randint(9, size=(5, 5))
        toutput = tinput.copy()
        test.append({'input' : tinput.tolist(), 'output' : toutput.tolist()})

    return build_task(train, test, 'identity')

if __name__ == "__main__":
    build_identity_task()
