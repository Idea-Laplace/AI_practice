# python documentations
## Basic information
link: https://docs.python.org/3/library/tkinter.html#architecture


Tcl: Tcl is a dynamic interpreted programming language, just like python   

Tk:

Tk is a Tcl package implemented in C that adds custom commands   
to create and manipulate GUI widgets. Each Tk object embeds its   

own Tcl interpreter instance with TK loaded into it.

Ttk(themed tk):
A newer family of Tk widgets that provide a much better appearance   
on different platforms than many of the classic widgets.

Python bindings are provided in a separate module,
tkinter.ttk/

internally, Tk and Ttk use facilites of the underlying OS

tk
The Tk application object created by instatiating Tk. This
provides access to the Tcl interpreter.
Each widget that is attached the same instance of Tk has the
same value for it tk attribute
