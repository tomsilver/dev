from dsl import *
from grammar_utils import enumerate_from_grammar
from structs import Feature

import pickle


### Grammatical Prior
START, START2, GET_CONST_COLOR, GET_VAR_COLOR, CELL, DIRECTION, POSITIVE_NUM, NEGATIVE_NUM, VALUE = range(9)

def create_grammar():
    object_types = [str(i) for i in range(10)] + ['None']
    grammar = {
        START : ([['compare_colors(', GET_VAR_COLOR, ', ', GET_CONST_COLOR, ', x)'],
                  ['compare_colors(', GET_VAR_COLOR, ', ', GET_VAR_COLOR, ', x)'],
                  ['search_for_color(', DIRECTION, ', ', GET_CONST_COLOR, ', x)'],
                  ['inside(', GET_CONST_COLOR, ', x)'],
                  ['row_is_divisible_by(1+', POSITIVE_NUM, ',', CELL,', x)'],
                  ['col_is_divisible_by(1+', POSITIVE_NUM, ',', CELL,', x)'],
                  ],
                  [1./6, 1./6, 1./6, 1./6, 1./6, 1./6]),
        GET_CONST_COLOR :([['lambda x :', VALUE]],
                          [1.0]), # a constant color, independent of x
        GET_VAR_COLOR : ([['lambda x : get_color_from_action(x)'], # a color depending on the action
                          ['lambda x : get_color_from_cell(', CELL, ', x)'], # a color depending on the grid
                          ['lambda x : most_common_color(x)'], # most common color in grid
                          ['lambda x : least_common_color(x)'], # least common color in grid
                         ],
                         [1./4, 1./4, 1./4, 1./4]),
        CELL : ([['lambda x : candidate_cell(x)'], 
                ['lambda x : neighbor_cell(', DIRECTION, ', x)'],
                ['lambda x : mod_candidate_cell(x)'],
                ['lambda x : mod_neighbor_cell(', DIRECTION, ', x)'],
                ],
                [1./4, 1./4, 1./4, 1./4]),
        DIRECTION : ([['(', POSITIVE_NUM, ', 0)'], ['(0,', POSITIVE_NUM, ')'],
                      ['(', NEGATIVE_NUM, ', 0)'], ['(0,', NEGATIVE_NUM, ')'],
                      ['(', POSITIVE_NUM, ',', POSITIVE_NUM, ')'], ['(', NEGATIVE_NUM, ',', POSITIVE_NUM, ')'],
                      ['(', POSITIVE_NUM, ',', NEGATIVE_NUM, ')'], ['(', NEGATIVE_NUM, ',', NEGATIVE_NUM, ')']],
                     [1./8] * 8),
        POSITIVE_NUM : ([
                         ['1'], [POSITIVE_NUM, '+1']
                         # ['1'], ['2'],
                         ], 
                         [0.99, 0.01]),
        NEGATIVE_NUM : ([
                         ['-1'], [NEGATIVE_NUM, '-1']
                         # ['-1'], ['-2'],
                         ], # 
                         [0.99, 0.01]),
        VALUE : (object_types, 
                 [1./len(object_types) for _ in object_types])
    }
    return grammar

def generate_features(num_programs=2500, outfile='features.pkl'):
    grammar = create_grammar()

    program_generator = enumerate_from_grammar(grammar)
    programs = []
    program_prior_log_probs = []

    print("Generating {} programs".format(num_programs))
    for _ in range(num_programs):
        try:
            program, lp = next(program_generator)
        except StopIteration:
            print("Only {} programs have been enumerated. Continue?".format(len(programs)))
            import ipdb; ipdb.set_trace()
            break
        programs.append(program)
        program_prior_log_probs.append(lp)
        print("program:", program)
    print("\nDone.")

    with open(outfile, 'wb') as f:
        pickle.dump((programs, program_prior_log_probs), f)

    print("Dumped features to {}.".format(outfile))




def load_features(num_programs=2500, outfile='features.pkl'):
    generate_features(num_programs=num_programs)
    with open(outfile, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    generate_features()
