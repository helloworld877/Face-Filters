import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from CLM import CLM

def on_button_click():
    messagebox.showinfo("Hello", "Button Clicked!")

def select_filter(filter_name):
    global selected_filter
    selected_filter.set(filter_name)
    
    if filter_name == "Sunglasses":
        CLM(0)
    elif filter_name == "Clown":
        CLM(1)
    elif filter_name == "Cats":
        CLM(2)
    elif filter_name == "Mustache":
        CLM(3)
    else:
        print("Invalid filter name")

root = tk.Tk()

# Create filter dropdown
selected_filter = tk.StringVar()

# Create a canvas for displaying the webcam feed
canvas = tk.Canvas(root, width=400, height=100)
canvas.grid(row=0, column=0, padx=10, pady=10, rowspan=5)  # Use rowspan to make the canvas span multiple rows

# Center the canvas in the window
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# Create buttons and center them in the canvas
button1 = tk.Button(root, text="Sunglasses", command=lambda: select_filter("Sunglasses"))
button1.grid(row=1, column=0, padx=10, pady=5, sticky='nsew')

button2 = tk.Button(root, text="Clown", command=lambda: select_filter("Clown"))
button2.grid(row=2, column=0, padx=10, pady=5, sticky='nsew')

button3 = tk.Button(root, text="Cats", command=lambda: select_filter("Cats"))
button3.grid(row=3, column=0, padx=10, pady=5, sticky='nsew')

button4 = tk.Button(root, text="Mustache", command=lambda: select_filter("Mustache"))
button4.grid(row=4, column=0, padx=10, pady=5, sticky='nsew')

# Run the GUI event loop
root.mainloop()