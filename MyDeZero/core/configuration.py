import contextlib

class Configuration:
    retain_grad = True
    enable_backpropagation = True
'''
For an instance to be capable of being with 'with',
it must have two magic method, __enter__ and __exit__.
What the __enter__ does is preprocessing before initiation of the indented code block within 'with'
What the __exit__ does is postprocessing after the code block is finished.
the contextlib.contextmanger helps an instance to be such instance, with try-finally paragraphs,
each of which parallel to __enter__ and __exit__ respectively.

Revised version by the chatgpt

For an instance to be usable with the with statement, it must implement two magic methods:
 __enter__ and __exit__. The __enter__ method handles any preprocessing before the indented
 code block within the with statement is executed, while the __exit__ method takes care of
 postprocessing after the code block is finished. 
 However, some instances such as functions cannot implement __enter__ or __exit__ method
 directly. The contextlib.contextmanager decorator wraps such instances with a class and
 simplifies creating such an instance by using try and finally blocks, which correspond to
 the functionality of __enter__ and __exit__, respectively.

'''
@contextlib.contextmanager
def using_config(name: str, value):
    # Functions like the __enter__ magic method-----
    restore_value = getattr(Configuration, name)
    setattr(Configuration, name, value)
    try:
        yield
    #-----------------------------------------------
    # Functions like the __exit__ magic method------
    finally:
        setattr(Configuration, name, restore_value)
    #-----------------------------------------------

def no_backpropagation():
    return using_config('enable_backpropagation', False)

def no_intermediate_grad():
    return using_config('retain_grad', False)