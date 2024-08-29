import MyDeZero.core.common_functions as cmf
import MyDeZero.core.core_functions as crf
from MyDeZero import Variable
import numpy as np

x = Variable(np.random.randn(2, 2, 2))
print(x.data)

dropout = cmf.DropOut(0.5)
func = lambda x: dropout(x).sum()
y = dropout(x)
z = y.sum()
z.backward()
print(x.grad.data)