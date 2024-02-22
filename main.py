import tkinter as tk
import os
import subprocess

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

def run_script(script_path):
    try:
        subprocess.Popen(["python", script_path])
        status_label.config(text="Exercise started successfully.", fg="green")
    except Exception as e:
        status_label.config(text=f"Error: {e}", fg="red")

# Create main window
root = tk.Tk()
root.title("Exercise Selector")
root.resizable(False, False)

# Create labels and buttons
label = tk.Label(root, text="Select an exercise:", font=("Arial", 14, "bold"))
label.pack()

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

def create_button(text, script_path, color):
    button = tk.Button(button_frame, text=text, width=15, height=2, command=lambda: run_script(script_path), bg = color)
    button.pack(side=tk.LEFT, padx=10)

create_button("Deadlift", os.path.join(current_dir, "deadlift", "deadlift.py"), "lightblue")
create_button("Shoulder Press", os.path.join(current_dir, "shoulderPress", "shoulderPress.py"), "lightgreen")
create_button("Bicep Curl", os.path.join(current_dir, "bicepCurl", "bicepCurl.py"), "red")

status_label = tk.Label(root, text="", fg="black")
status_label.pack(pady=5)

# Run the application
root.mainloop()