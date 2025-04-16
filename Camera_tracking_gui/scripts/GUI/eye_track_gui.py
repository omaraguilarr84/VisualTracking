import customtkinter as ctk
from tkinter import filedialog
import threading
import os
import sys
import importlib.util
import time
import subprocess
from eye_segment import segment_image

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Eye Segmentation Tool")
app.geometry("700x500")

# Global vars
input_mode = ctk.StringVar(value="image")
camera_source = ctk.StringVar(value="laptop")
recording_active = False

# Create a frame for instructions that will be shown/hidden
instructions_frame = ctk.CTkFrame(app)
instructions_frame.pack(pady=10, fill="x", padx=20)

instructions_label = ctk.CTkLabel(instructions_frame, text="Video Recording Instructions:", font=("Arial", 14, "bold"))
instructions_label.pack(pady=5)

record_instruction = ctk.CTkLabel(instructions_frame, text="Hit 'r' to record a 45 sec video")
record_instruction.pack(pady=2)

quit_instruction = ctk.CTkLabel(instructions_frame, text="Hit 'q' to stop real time video")
quit_instruction.pack(pady=2)

# Initially hide the instructions
instructions_frame.pack_forget()

def show_instructions():
    global recording_active
    try:
        instructions_frame.pack(pady=10, fill="x", padx=20)
        recording_active = True
    except Exception as e:
        print(f"Error showing instructions: {e}")

def hide_instructions():
    global recording_active
    try:
        instructions_frame.pack_forget()
        recording_active = False
    except Exception as e:
        print(f"Error hiding instructions: {e}")

def browse_file():
    file_path = filedialog.askopenfilename()
    file_entry.delete(0, ctk.END)
    file_entry.insert(0, file_path)

def browse_directory():
    dir_path = filedialog.askdirectory()
    file_entry.delete(0, ctk.END)
    file_entry.insert(0, dir_path)

def run_segmentation():
    mode = input_mode.get()
    source = camera_source.get()
    file_path = file_entry.get()

    def task():
        if mode == "image":
            print(f"[Image Mode] Running model on: {file_path}")
            # Call your image segmentation function here
            segment_image(file_path)
        elif mode == "realtime":  # Changed from "video" to "realtime"
            cam_index = 0 if source == "laptop" else 1
            print(f"[Real-time Mode] Running model on camera {cam_index}")
            
            # Show instructions before starting the video
            app.after(100, show_instructions)
            
            # Import and run the real_time_segment.py script
            try:
                # Get the directory of the current script
                current_dir = os.path.dirname(os.path.abspath(__file__))
                real_time_segment_path = os.path.join(current_dir, "real_time_segment.py")
                
                # Run the script as a subprocess with the camera index argument
                cmd = [sys.executable, real_time_segment_path, "--camera", str(cam_index)]
                process = subprocess.Popen(cmd)
                
                # Wait for the process to complete
                process.wait()
                
                # Hide instructions after the video is closed
                app.after(100, hide_instructions)
            except Exception as e:
                print(f"Error running real-time segmentation: {e}")
                app.after(100, hide_instructions)
    
    threading.Thread(target=task).start()

# Widgets
mode_label = ctk.CTkLabel(app, text="Choose Input Type:")
mode_label.pack(pady=5)

mode_frame = ctk.CTkFrame(app)
mode_frame.pack()

image_radio = ctk.CTkRadioButton(mode_frame, text="Import Image", variable=input_mode, value="image")
realtime_radio = ctk.CTkRadioButton(mode_frame, text="Real-time Video", variable=input_mode, value="realtime")
image_radio.grid(row=0, column=0, padx=10)
realtime_radio.grid(row=0, column=1, padx=10)

# File selection frame
file_frame = ctk.CTkFrame(app)
file_frame.pack(pady=10)

file_entry = ctk.CTkEntry(file_frame, width=300)
file_entry.pack(side="left", padx=5)

# Buttons for browsing file or directory
browse_file_button = ctk.CTkButton(file_frame, text="Browse File", command=browse_file)
browse_file_button.pack(side="left", padx=5)

browse_directory_button = ctk.CTkButton(file_frame, text="Browse Directory", command=browse_directory)
browse_directory_button.pack(side="left", padx=5)

# Camera selection frame - only visible when "Real-time Video" is selected
camera_frame = ctk.CTkFrame(app)
camera_frame.pack(pady=10)

camera_label = ctk.CTkLabel(camera_frame, text="Camera Source:")
camera_label.pack(side="left", padx=10)

laptop_radio = ctk.CTkRadioButton(camera_frame, text="Laptop", variable=camera_source, value="laptop")
usb_radio = ctk.CTkRadioButton(camera_frame, text="USB", variable=camera_source, value="usb")
laptop_radio.pack(side="left")
usb_radio.pack(side="left")

# Function to show/hide frames based on selected mode
def update_ui():
    if input_mode.get() == "image":
        file_frame.pack(pady=10)
        camera_frame.pack_forget()
    else:  # realtime
        file_frame.pack_forget()
        camera_frame.pack(pady=10)

# Set initial UI state
update_ui()

# Add trace to update UI when mode changes
input_mode.trace_add("write", lambda *args: update_ui())

run_button = ctk.CTkButton(app, text="Run Segmentation", command=run_segmentation)
run_button.pack(pady=20)

app.mainloop()
