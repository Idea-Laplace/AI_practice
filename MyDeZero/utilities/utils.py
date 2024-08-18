import os
import subprocess
import numpy as np
from MyDeZero import Variable

def get_dot_graph(output: Variable, verbose: bool=True):
    txt = ''
    functions = []
    seen_set = set()

    def add_function(f: callable):
        if f not in seen_set:
            functions.append(f)
            # Contrary to the backward method in the class Variable,
            # The graph which would be generated isn't concerned with the order of drawing arrows,
            # as all that the graph show is complete, abstract stream of backpropagtion.
            # In the backward, this is critical: functions.sort(key=lambda x: x.generation)
            seen_set.add(f)
    
    # Initialization before iteration
    add_function(output.creator)
    txt += _dot_var(output, verbose)

    while functions:
        func = functions.pop()
        # Arrowing inputs and outputs
        txt += _dot_function(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_function(x.creator)
    
    return 'digraph g {\n' + txt + '}'


def plot_dot_graph(output: Variable,\
                    verbose: bool=True,\
                    to_file: str='graph.png'):

    dot_graph = get_dot_graph(output, verbose)
    tmp_dir = os.path.join(os.path.expanduser('~'), '.mydezero')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as tmp_file:
        tmp_file.write(dot_graph)
    
    extension = os.path.splitext(to_file)[1][1:] #  'graph.png' -> '.png' -> 'png'
    command = f'dot {graph_path} -T {extension} -o {to_file}'
    subprocess.run(command, shell=True)



def _dot_var(v: Variable, verbose: bool=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'
    name = '' if v.name is None else v.name

    if verbose and v.data is not None:
        # When the name of v is defined.
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)

def _dot_function(f: callable, verbose: bool=False):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)
    dot_arrows = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_arrows.format(id(x), id(f))
    for y in f.outputs:
        # Remember that f.outputs makes use of weakref
        txt += dot_arrows.format(id(f), id(y()))
    
    return txt
