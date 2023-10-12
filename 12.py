import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import pandas as pd

# Create a Tkinter window
root = tk.Tk()
root.title("CSV File Viewer")

# Function to open and display CSV file
def open_csv_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

    if file_path:
        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(file_path)

        # Create a new window to display the data
        data_window = tk.Toplevel(root)
        data_window.title("CSV Data")

        # Create a Treeview widget to display the data
        data_table = ttk.Treeview(data_window)

        # Add a horizontal scrollbar to the Treeview
        xscrollbar = ttk.Scrollbar(data_window, orient="horizontal", command=data_table.xview)
        data_table.configure(xscrollcommand=xscrollbar.set)
        xscrollbar.pack(side="bottom", fill="x")

        # Create columns in the Treeview based on the DataFrame columns
        data_table["columns"] = list(df.columns)
        for col in df.columns:
            data_table.heading(col, text=col)
            data_table.column(col, width=100)

        # Insert the data into the Treeview
        for i, row in df.iterrows():
            values = [row[col] for col in df.columns]
            data_table.insert("", "end", values=values)

        data_table.pack()

# Create a button to open the CSV file
open_button = tk.Button(root, text="Open CSV File", command=open_csv_file)
open_button.pack()

# Start the Tkinter main loop
root.mainloop()
