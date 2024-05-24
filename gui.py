import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import shutil
import os
import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import cv2

# Variable to store the path of the previously uploaded image and video
previous_image_path = None
previous_video_path = None

# Define the input height and width expected by the model
input_height = 224
input_width = 224

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the images directory and create it if it doesn't exist
images_directory = os.path.join(script_dir, 'downloadedimages')
os.makedirs(images_directory, exist_ok=True)

# Function to load the model
def load_model(model_dir):
    model = tf.saved_model.load(model_dir)
    return model

# Function to detect objects in an image
def detect_objects(image_path, model):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, (input_height, input_width))
    img = tf.cast(img, tf.uint8)
    
    input_tensor = tf.expand_dims(img, 0)
    detections = model(input_tensor)
    
    return img, detections

# Function to draw bounding boxes on the image
def draw_boxes(image, boxes, scores, classes, model_type, threshold=0.5, max_boxes=10):
    image_pil = Image.fromarray(np.uint8(image.numpy()))
    draw = ImageDraw.Draw(image_pil)
    
    if model_type == "Multiclass":
        color_map = {1: "green", 2: "red"}
        class_map = {1: "Misplaced Item", 2: "Out of stock"}
    elif model_type == "Oneclass - Out of Stock":
        color_map = {1: "red"}
        class_map = {1: "Out of stock"}
    elif model_type == "Oneclass - Misplaced":
        color_map = {1: "green"}
        class_map = {1: "Misplaced Item"}

    sorted_indices = np.argsort(scores)[::-1][:max_boxes]
    
    for i in sorted_indices:
        if scores[i] > threshold:
            box = boxes[i]
            class_id = int(classes[i])
            if class_id in class_map and class_id in color_map:
                class_name = class_map[class_id]
                color = color_map[class_id]
                
                draw.rectangle([(box[1] * image_pil.width, box[0] * image_pil.height), 
                                (box[3] * image_pil.width, box[2] * image_pil.height)], outline=color, width=2)
                draw.text((box[1] * image_pil.width, box[0] * image_pil.height), f"{class_name}: {scores[i]:.2f}", fill=color)
    
    return image_pil

# Function to display the result
def display_result(image):
    window = tk.Toplevel(root)
    window.title("Detected Image")
    
    resized_image = image.resize((500, 500))
    img = ImageTk.PhotoImage(resized_image)
    
    label = tk.Label(window, image=img)
    label.image = img
    label.pack()

# Function to upload an image
def upload_image():
    global previous_image_path
    
    file_path = filedialog.askopenfilename(title="Select an image file", 
                                           filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
    
    if file_path:
        root_folder = script_dir
        images_folder = os.path.join(root_folder, 'guiimages')
        os.makedirs(images_folder, exist_ok=True)
        
        file_name = os.path.basename(file_path)
        destination_path = os.path.join(images_folder, file_name)
        
        if previous_image_path and os.path.exists(previous_image_path):
            os.remove(previous_image_path)
        
        shutil.copy(file_path, destination_path)
        previous_image_path = destination_path
        
        selected_model = model_var.get()
        if selected_model == "Multiclass":
            model_path = os.path.join(script_dir, 'gui_model', 'Multiclass')
        elif selected_model == "Oneclass - Out of Stock":
            model_path = os.path.join(script_dir, 'gui_model', 'Outofstock')
        elif selected_model == "Oneclass - Misplaced":
            model_path = os.path.join(script_dir, 'gui_model', 'Misplaced')
        else:
            messagebox.showerror("Error", "Invalid model selected.")
            return
        
        model = load_model(model_path)
        image, detections = detect_objects(destination_path, model)
        
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy()
        
        threshold = 0.5
        
        image_with_boxes = draw_boxes(image, boxes, scores, classes, selected_model, threshold=threshold, max_boxes=10)
        
        display_result(image_with_boxes)
        
        messagebox.showinfo("Success", f"Image has been saved to {destination_path}")
    else:
        messagebox.showwarning("No file selected", "Please select an image file.")

# Function to upload a video
def upload_video():
    global previous_video_path
    
    file_path = filedialog.askopenfilename(title="Select a video file", 
                                           filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
    
    if file_path:
        previous_video_path = file_path
        process_video(file_path)
    else:
        messagebox.showwarning("No file selected", "Please select a video file.")

# Function to process a video
def process_video(video_path):
    selected_model = model_var.get()
    if selected_model == "Multiclass":
        model_path = os.path.join(script_dir, 'gui_model', 'Multiclass')
    elif selected_model == "Oneclass - Out of Stock":
        model_path = os.path.join(script_dir, 'gui_model', 'Outofstock')
    elif selected_model == "Oneclass - Misplaced":
        model_path = os.path.join(script_dir, 'gui_model', 'Misplaced')
    else:
        messagebox.showerror("Error", "Invalid model selected.")
        return
        
    model = load_model(model_path)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video file.")
        return
    
    if selected_model == "Multiclass":
        colors = {1: (0, 255, 0), 2: (0, 0, 255)}
        class_names = {1: "Misplaced Item", 2: "Out of stock"}
    elif selected_model == "Oneclass - Out of Stock":
        colors = {1: (0, 0, 255)}
        class_names = {1: "Out of stock"}
    elif selected_model == "Oneclass - Misplaced":
        colors = {1: (0, 255, 0)}
        class_names = {1: "Misplaced Item"}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = model(input_tensor)
        
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(int)
        
        for i in range(len(scores)):
            if scores[i] > 0.5:
                h, w, _ = frame.shape
                ymin, xmin, ymax, xmax = boxes[i]
                xmin = int(xmin * w)
                xmax = int(xmax * w)
                ymin = int(ymin * h)
                ymax = int(ymax * h)
                
                class_id = classes[i]
                if class_id in colors and class_id in class_names:
                    color = colors[class_id]
                    class_name = class_names[class_id]
                    
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(frame, f"{class_name}: {scores[i]:.2f}", (xmin, ymin - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow('Object Detection', frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Create the main window
root = tk.Tk()
root.title("Image Uploader")

# Create a button to upload a video
upload_video_button = tk.Button(root, text="Upload Video", command=upload_video, font=("Helvetica", 12))
upload_video_button.pack(pady=20)

# Set the window size
root.geometry("400x300")

# Create a label
label = tk.Label(root, text="Upload an image", font=("Helvetica", 14))
label.pack(pady=20)

# Create a dropdown menu for model selection
model_var = tk.StringVar()
model_options = ["Multiclass", "Oneclass - Out of Stock", "Oneclass - Misplaced"]
model_var.set(model_options[0])
model_menu = ttk.Combobox(root, textvariable=model_var, values=model_options, font=("Helvetica", 12))
model_menu.pack(pady=20)

# Create a button to upload an image
upload_button = tk.Button(root, text="Upload Image", command=upload_image, font=("Helvetica", 12))
upload_button.pack(pady=20)

# Run the application
root.mainloop()
