import tkinter as tk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Initialze Tkinter
root = tk.Tk()

# Basic configuration
frame = tk.Frame(root)
label = tk.Label(text="Matplotlib wallpaper for MNIST")
label.pack()
frame.pack()

# Matplotlib canvas
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

root.mainloop()
