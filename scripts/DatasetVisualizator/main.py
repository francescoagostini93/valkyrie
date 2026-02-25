import os
import csv
import json
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import argparse

class DataVerificationTool:
    def __init__(self, master, dataset_path):
        self.master = master
        master.title("Data Verification Tool")
        self.dataset_path = dataset_path
        self.rejected_images = []
        self.create_widgets()
        self.bind_keys()

    def create_widgets(self):
        # Main frame
        self.main_frame = ttk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Left frame for existing content
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Database selection
        self.db_frame = ttk.Frame(self.left_frame)
        self.db_frame.pack(pady=10)
        ttk.Label(self.db_frame, text="Select Database:").pack(side=tk.LEFT)
        self.db_combo = ttk.Combobox(self.db_frame, values=self.get_database_folders())
        self.db_combo.pack(side=tk.LEFT)
        self.db_combo.bind("<<ComboboxSelected>>", self.on_database_selected)

        # Progress bar
        self.progress_frame = ttk.Frame(self.left_frame)
        self.progress_frame.pack(pady=5)
        self.progress_bar = ttk.Progressbar(self.progress_frame, length=300, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT)
        self.progress_label = ttk.Label(self.progress_frame, text="0/0")
        self.progress_label.pack(side=tk.LEFT, padx=5)

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

        # Buttons
        self.button_frame = ttk.Frame(self.left_frame)
        self.button_frame.pack(pady=10)
        self.accept_button = ttk.Button(self.button_frame, text="Accept (A)", command=self.accept_image)
        self.accept_button.pack(side=tk.LEFT, padx=5)
        self.reject_button = ttk.Button(self.button_frame, text="Reject (S)", command=self.reject_image)
        self.reject_button.pack(side=tk.LEFT, padx=5)
        self.back_button = ttk.Button(self.button_frame, text="Back", command=self.go_back)
        self.back_button.pack(side=tk.LEFT, padx=5)
        self.finish_button = ttk.Button(self.button_frame, text="Finish and Apply", command=self.finish_and_apply)
        self.finish_button.pack(side=tk.LEFT, padx=5)

        # Right frame for rejected images list
        self.right_frame = ttk.Frame(self.main_frame, width=200)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        self.right_frame.pack_propagate(False)

        ttk.Label(self.right_frame, text="Rejected Images:").pack()
        self.rejected_listbox = tk.Listbox(self.right_frame)
        self.rejected_listbox.pack(fill=tk.BOTH, expand=True)

    def bind_keys(self):
        self.master.bind('<Left>', lambda e: self.go_back())
        self.master.bind('<Right>', lambda e: self.accept_image())
        self.master.bind('a', lambda e: self.accept_image())
        self.master.bind('s', lambda e: self.reject_image())

    def get_database_folders(self):
        return [f for f in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, f)) and f.startswith('dataset_')]

    def on_database_selected(self, event):
        self.target_folder = os.path.join(self.dataset_path, self.db_combo.get())
        self.images_folder = os.path.join(self.target_folder, 'images')
        self.masks_folder = os.path.join(self.target_folder, 'masks')
        self.quarantine_folder = os.path.join(self.target_folder, 'quarantine')
        self.blacklist_csv = os.path.join(self.target_folder, 'blacklist.csv')
        self.progress_file = os.path.join(self.target_folder, 'progress.json')

        # Create quarantine folder if it doesn't exist
        if not os.path.exists(self.quarantine_folder):
            os.makedirs(os.path.join(self.quarantine_folder, 'images'))
            os.makedirs(os.path.join(self.quarantine_folder, 'masks'))

        # Load image list
        self.image_files = [f for f in os.listdir(self.images_folder) if f.endswith('.png')]
        self.load_progress()
        self.update_progress()
        self.load_image()

    def load_progress(self):
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                progress_data = json.load(f)
            self.current_index = self.image_files.index(progress_data['last_image'])
            self.rejected_images = progress_data['rejected_images']
        else:
            self.current_index = 0
            self.rejected_images = []
        self.update_rejected_listbox()

    def save_progress(self):
        progress_data = {
            'last_image': self.image_files[self.current_index],
            'rejected_images': self.rejected_images
        }
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f)

    def load_image(self):
        if 0 <= self.current_index < len(self.image_files):
            image_file = self.image_files[self.current_index]
            image_path = os.path.join(self.images_folder, image_file)
            mask_path = os.path.join(self.masks_folder, image_file)

            image = np.array(Image.open(image_path))
            mask = np.array(Image.open(mask_path))

            self.ax.clear()
            self.ax.imshow(image)
            self.ax.imshow(mask, alpha=0.3, cmap='jet')  # Reduced overlay opacity
            self.ax.set_title(f"Image: {image_file}")
            self.ax.axis('off')

            self.canvas.draw()
            self.update_progress()
            self.save_progress()
        elif self.current_index >= len(self.image_files):
            self.finish_and_apply()

    def update_progress(self):
        total = len(self.image_files)
        self.progress_bar['value'] = (self.current_index + 1) / total * 100
        self.progress_label['text'] = f"{self.current_index + 1}/{total}"

    def update_rejected_listbox(self):
        self.rejected_listbox.delete(0, tk.END)
        for img in self.rejected_images:
            self.rejected_listbox.insert(tk.END, img)

    def accept_image(self):
        self.current_index += 1
        self.load_image()

    def reject_image(self):
        if self.current_index < len(self.image_files):
            rejected_image = self.image_files[self.current_index]
            if rejected_image not in self.rejected_images:
                self.rejected_images.append(rejected_image)
                self.update_rejected_listbox()
        self.current_index += 1
        self.load_image()

    def go_back(self):
        if self.current_index > 0:
            self.current_index -= 1
            current_image = self.image_files[self.current_index]
            if current_image in self.rejected_images:
                self.rejected_images.remove(current_image)
                self.update_rejected_listbox()
            self.load_image()

    def finish_and_apply(self):
        self.process_rejected_images()
        self.master.quit()

    def process_rejected_images(self):
        for image_file in self.rejected_images:
            image_path = os.path.join(self.images_folder, image_file)
            mask_path = os.path.join(self.masks_folder, image_file)

            # Move image and mask to quarantine
            os.rename(image_path, os.path.join(self.quarantine_folder, 'images', image_file))
            os.rename(mask_path, os.path.join(self.quarantine_folder, 'masks', image_file))

            # Add to blacklist CSV
            with open(self.blacklist_csv, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([image_file])

        messagebox.showinfo("Finished", f"All images have been processed. {len(self.rejected_images)} images were rejected.")
        self.rejected_images = []
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)

def main():
    parser = argparse.ArgumentParser(description="Data Verification Tool")
    parser.add_argument("--path", type=str, help="Path to the dataset folder")
    args = parser.parse_args()

    if args.path and os.path.exists(args.path):
        dataset_path = args.path
    else:
        dataset_path = os.environ.get('DATASET_VALKIRIE', '')
        if not dataset_path:
            print("Error: No valid path provided and DATASET_VALKIRIE environment variable is not set.")
            return

    root = tk.Tk()
    _ = DataVerificationTool(root, dataset_path)
    root.mainloop()

if __name__ == "__main__":
    main()