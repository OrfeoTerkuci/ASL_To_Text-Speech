import tkinter as tk 
import tkinter.messagebox
import customtkinter as ctk

import sys
import os

# Get the path of the 'src' directory
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the 'src' directory to the Python path
sys.path.append(src_dir)

from image_rendering.image_recognition import WebcamHandtracking
from models.cnn.cnn import CNN

ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # configure window
        self.title("ASL Recognition System - GUI")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Models", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.current_model = None
        
         # Change to CNN model
        self.CNN_model = ctk.CTkButton(self.sidebar_frame, command=lambda: self.sidebar_button_event("CNN Model"))
        self.CNN_model.configure(text="CNN Model", corner_radius=4)
        self.CNN_model.grid(row=1, column=0, padx=20, pady=10)
        
        # Change to VIT model
        self.VIT_model = ctk.CTkButton(self.sidebar_frame, command=lambda: self.sidebar_button_event("VIT Model"))
        self.VIT_model.configure(text="VIT Model", corner_radius=4)
        self.VIT_model.grid(row=2, column=0, padx=20, pady=10)
        
        
        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["System", "Light", "Dark"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = ctk.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))
        
        self.webcam_button = ctk.CTkButton(master=self, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), command=lambda: self.start_tracking())
        self.webcam_button.configure(text="Start Tracking", corner_radius=4)
        self.webcam_button.grid(row=0, rowspan=4, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")

    def open_input_dialog_event(self):
        dialog = ctk.CTkInputDialog(text="Type in a number:", title="CTkInputDialog")
        print("CTkInputDialog:", dialog.get_input())

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        ctk.set_widget_scaling(new_scaling_float)

    def sidebar_button_event(self, Button_name: str):
        if Button_name == "CNN Model":
            self.CNN_model.configure(text="✔ Loaded CNN Model")
            self.VIT_model.configure(text="VIT Model")
            # TODO: Implement Logic for changing to CNN model
            self.current_model = CNN("","","", False)
            self.current_model.load("./models/best_models/cnn_1e-07_32_1000_0.004_2_6.pth")
        elif Button_name == "VIT Model":
            # self.VIT_model.configure(text="✔ Loaded VIT Model")
            self.CNN_model.configure(text="CNN Model")
            # TODO: Implement Logic for changing to VIT model
            self.current_model = None
            tkinter.messagebox.showinfo("VIT Model", "VIT Model coming soon")
        
    def start_tracking(self):
        webcam = WebcamHandtracking()
        if self.current_model is not None:
            webcam.load_model(self.current_model)
        webcam.multi_threading_hand_tracking()

