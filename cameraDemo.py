import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tensorflow.keras.models import load_model
import numpy as np

class cameraDemo:
    def __init__(self,model):
        self.root = tk.Tk()
        self.root.title("Camera and Image Processing")
        self.is_capturing = True
        self.model = model
        self.cap = cv2.VideoCapture(0)

        # Initialize labels
        self.label_hr = tk.Label(self.root)
        self.label_lr = tk.Label(self.root)
        self.label_sr = tk.Label(self.root)
        self.label_hr.pack(side="left")
        self.label_lr.pack(side="left")
        self.label_sr.pack(side="left")

        # Initialize button
        self.btn_toggle_capture = tk.Button(self.root, text="Capture", command=self.toggle_capture)
        self.btn_toggle_capture.pack()

        # Start capturing and processing
        self.capture_and_process()

        self.root.mainloop()

        # Release the camera when done
        self.cap.release()
        cv2.destroyAllWindows()

    def toggle_capture(self):
        self.is_capturing = not self.is_capturing  # Toggle the flag
        if self.is_capturing:
            self.capture_and_process()

    def capture_and_process(self):
        ret, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hr, lr = self.crop_and_resize(frame)
        sr = self.super_resolution(lr)
        self.display_images(hr, lr, sr)

        # Call this function again after 100ms (0.1s) if is_capturing is True
        if self.is_capturing:
            self.root.after(100, self.capture_and_process)

    def crop_and_resize(self, frame):
        # Calculate the center of the frame
        h, w, _ = frame.shape
        center_y, center_x = h // 2, w // 2
        start_x, start_y = center_x - 89, center_y - 109
        end_x, end_y = center_x + 89, center_y + 109

        # Crop the frame to 218X178 around the center
        hr = frame[start_y:end_y, start_x:end_x]

        hr = cv2.resize(hr, (96, 96))
        lr = cv2.resize(hr, (24, 24))

        return hr, lr

    def super_resolution(self, lr):
        # Prepare the LR image for the model and get the SR image
        lr_for_model = np.expand_dims(lr, axis=0)  # Add batch dimension
        sr = self.model.predict(lr_for_model)[0]  # Remove batch dimension
        sr = np.clip(sr, 0, 255).astype(np.uint8)  # Clip values to 0-255 and convert to uint8
        return sr

    def display_images(self, hr, lr, sr):
        # Convert to PIL Images
        frame_im = Image.fromarray(hr)
        lr_im = Image.fromarray(lr)
        sr_im = Image.fromarray(sr)

        # Convert to ImageTk
        frame_photo = ImageTk.PhotoImage(frame_im)
        lr_photo = ImageTk.PhotoImage(lr_im)
        sr_photo = ImageTk.PhotoImage(sr_im)

        # Display images
        self.label_hr.config(image=frame_photo)
        self.label_hr.image = frame_photo
        self.label_lr.config(image=lr_photo)
        self.label_lr.image = lr_photo
        self.label_sr.config(image=sr_photo)
        self.label_sr.image = sr_photo

