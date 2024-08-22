import numpy as np
import weakref
import MyDeZero
from MyDeZero.core.configuration import *


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
    self.name: Type> str
                The name of a created Variable instance                    
'''

class Variable:
    def __init__(self, data: np.ndarray, name = None):
        if data is not None and not isinstance(data, np.ndarray):
            self.data = np.array(data)
            if not np.issubdtype(self.data.dtype, np.number):
                raise TypeError(f'{type(data)} is not supported.')
        else:
            self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0
    
    # For user usability------------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype
    
    @property
    def ndim(self):
        return self.data.ndim
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'Variable(None)'
        else:
            repr = str(self.data).replace('\n', '\n' + ' ' * len('Variable('))
            return 'Variable(' + repr + ')'
    
    def reshape(self, *shape):
        # When an user enter a shape as a form of tuple or list.
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return MyDeZero.core.core_functions.reshape(self, shape)
    
    def transpose(self, *axes):
        return MyDeZero.core.core_functions.transpose(self, tuple(axes))
    
    @property
    def T(self):
        return MyDeZero.core.core_functions.transpose(self)
    
    def sum(self, axis=None, keepdims=False):
        return MyDeZero.core.core_functions.sum(self, axis, keepdims)


    # /For user usability-----------------------------

    # Pertaining to differentiation-------------------
    def set_creator(self, func: callable):
        self.creator = func
        # Since a given 'Variable' is generated by its creator.
        self.generation = func.generation + 1

    def clear_grad(self):
        self.grad = None

    # retain_grad is for memory.
    def backward(self, retain_grad=Configuration.retain_grad, create_graph=True):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        #-------------------------------------------------
        # Since functions are actually callable classes,
        # although two 'Function's seem to do same subroutine,
        # they are actually 'different' instances, which would not happen
        # when they are merely of a type of function, not a type of callable classes.
        # They are same with each other only when different variable shares same creator.
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
            # func.outputs is guaranteed that it is of type tuple
            # func.outputs consists of weakref.ref element
            grad_ys = [output().grad for output in func.outputs]

            with using_config('enable_backpropagation', create_graph):
                grad_xs = func.backward(*grad_ys)

                if not isinstance(grad_xs, tuple):
                    grad_xs = (grad_xs,)
                    
                for x, grad_x in zip(func.inputs, grad_xs):
                    if x.grad is None:
                        x.grad = grad_x
                    else:
                        # You should not write as x.grad += grad_x, as it could cause
                        # influence on other Variable.grad, especially when x.grad is
                        # initialized directly on another Variable.grad.
                        x.grad = x.grad + grad_x

                    # If there are many multi-variable functions in the
                    # chain of a neuron system heirarchy, the 'functions'
                    # list would be of an abstract 'tree' structure.
                    if x.creator is not None:
                        # The functions are arranged according to generation
                        # This process guarantees that no variables of lower generation
                        # would not get their gradient until those of higher generation get over.
                        add_function(x.creator)
            
            # After all the gradients of input variables are gained,
            # the used up output gradients, which would only just waste memories in many cases,
            # would be erased.
            if not retain_grad:
                for output_weakref in func.outputs:
                    output_weakref().grad = None
    # /Pertaining to differentiation-------------------

# Class variables
def setup_variable_operations_overload():
    from .core_functions import add, mul, neg, sub, rsub, div, rdiv, pow
    # Operations between np.ndarray and Variable, The Variable operator
    # would be choosed over that of numpy array
    Variable.__array_priority__ = 1000

    Variable.__add__ = add
    Variable.__radd__ = add

    Variable.__mul__ = mul
    Variable.__rmul__ = mul

    Variable.__neg__ = neg

    Variable.__sub__ = sub
    Variable.__rsub__ = rsub

    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv

    Variable.__pow__ = pow

#--------------------------------------------------------------------------------------------------
# class Function
# The class 'Function' is a super class
# for specific function subclasses.

# The Function class refers to its input
# The Function class refers weakly to its output (no reference count for output increases.)
class Function:
    def __call__(self, *inputs) -> Variable:
        # Extracting data to apply to the method 'forward'
        # The 'inputs' is matained after call only when backpropagation mode is true.
        # Wraps np.ndarrays with a Variable class.
        inputs = [as_variable(input) for input in inputs] # inputs: list of Variables
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)

        # To be consistent with tuple type, which is iterable
        if not isinstance(ys, tuple):
            ys = (ys,)

        # Refer to as_array function in the Supportive function section.
        outputs = [Variable(as_array(y)) for y in ys] 

        # Only when backpropagtion is needed.---------------------------------------
        if Configuration.enable_backpropagation:
            # The generation of a 'Function' is determined by its input generation
            self.generation = max([input.generation for input in inputs])
            # A new variable instance
            # When a subroutine returns multiple variables
            # The list below would lose its reference count once the call ends,
            # since outputs will no more exist after the call

            for output in outputs:
                output.set_creator(self)
            # Stores the recent call information
            self.inputs = inputs
            # What is actually remains a list of weakrefs
            self.outputs = [weakref.ref(output) for output in outputs]

        # tuple of Variables or a single Variable
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *input: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
#--------------------------------------------------------------------------------

def as_variable(x: np.ndarray) -> Variable:
    if isinstance(x, Variable):
        return x
    else:
        return Variable(x)

def as_array(x) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    else:
        return x

#--------------------------------------------------------------------------------
'''
The backward method in detail

When a particular 'Variable' instance calls its 'backward' method,
first it calls its creator 'Function', second, the creator function
returns gradient(s) for inputs which would have derived the resulting variable.
Each gradient would be culminated to its proper 'Variable'
If it were not for sort method, until one chain of backpropagtion on all the way a branch 
ends, the other branches could not initiate their backpropagation, which would result in
creating imperfect gradient info and culminating it.
By sorting with reference to the instance variable 'generation', a series of neuron system
coulf backpropagate horizontally not vertically
'''

'''
Chatgpt improved my crude english write.

The Backward Method in Detail

When the backward method is called on a particular Variable instance,
it first invokes the Function that created it.
The creator function then returns the gradient(s) for the inputs that 
contributed to producing the resulting variable. Each gradient is then 
accumulated in its corresponding Variable.
Without a sorting method, one branch of the backpropagation chain would need to fully complete
before other branches could begin their backpropagation. 
This could result in incomplete gradient information and improper accumulation.
By sorting based on the generation instance variable, the neural network can backpropagate
in a horizontal sequence rather than vertically.
'''
