from MyDeZero.core.configuration import *
from MyDeZero.core.core_classes import Variable, Parameter, Layer, Optimizer
from MyDeZero.core.core_classes import as_array, as_variable
from MyDeZero.core.core_classes import setup_variable_operations_overload
from MyDeZero.core.core_functions import reshape, numerical_gradient
from MyDeZero.core.common_classes import Linear
from MyDeZero.core.common_functions import *
from MyDeZero.utilities.utils import get_dot_graph, plot_dot_graph
from MyDeZero.cuda import *

setup_variable_operations_overload()
