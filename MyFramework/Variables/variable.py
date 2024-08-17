import numpy as np
import sys, os
sys.path.append(os.pardir)

'''
Variable:
    self.data   : Type>np.ndarray
                Main information the created instance would contain.

    self.creator: Type> callable
                Default value is None. If a given instance is from a return value of a function,
                the self.creator variable should store the function instance.
                The self.creator functions like self.prev as in linklist ADT

    self.grad   : Type> np.ndarray
                It stores the information of gradient for the recent called function
    
    self.generation:  Type> int
                In a chain of 'Variable's and 'Function's, this instance variable indicates
                the relative heirarchy of its instance
'''
                    

class Variable:
    def __init__(self, data: np.ndarray):
        if (data is not None) and (not isinstance(data, np.ndarray)):
            raise TypeError(f'{type(data)} is not supported.')

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func: callable):
        self.creator = func
        # Since a given 'Variable' is generated by its creator.
        self.generation = func.generation + 1

    def clear_grad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        #-------------------------------------------------
        functions = []
        seen_functions = set()

        # Rearrange the catalog of functions
        def add_function(f: callable):
            if f not in seen_functions:
                functions.append(f)
                seen_functions.add(f)
                functions.sort(key=lambda f: f.generation)
        
        add_function(self.creator)
        
        while functions:
            func = functions.pop()
            # Call the most recent called Variable instances
            # Warning: y may not be the instance that would call this method.

            # func.outputs is guaranteed that it is of type tuple
            grad_ys = [output.grad for output in func.outputs]
            grad_xs = func.backward(*grad_ys)

            if not isinstance(grad_xs, tuple):
                grad_xs = (grad_xs,)
                
            for x, grad_x in zip(func.inputs, grad_xs):
                if x.grad is None:
                    x.grad = grad_x
                else:
                    x.grad += grad_x

                # If there are many multi-variable functions in the
                # chain of a neuron system heirarchy, the 'functions'
                # list would be of an abstract 'tree' structure.
                if x.creator is not None:
                    # The functions are arranged according to generation
                    add_function(x.creator)

#--------------------------------------------------------------------------------
