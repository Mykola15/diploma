import numpy as np
from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import clear_output
import imageio
import time

# Load the pretrained CNN model
model = load_model('/Users/mykola/Desktop/model.h5')

# Define the labels corresponding to your classes
class_labels = ['airplane', 'fighter jet', 'drone', 'missile', 'helicopter']

# Open an MP4 video file
video_file = '/Users/mykola/Desktop/helicopter.mp4'  # Specify the path to your MP4 video file

# Read the video using imageio
video_reader = imageio.get_reader(video_file)

# Initialize variables for FPS calculation
start_time = time.time()
frame_count = 0

for frame in video_reader:
    # Preprocess the frame
    frame_pil = Image.fromarray(frame)  # Convert to PIL Image
    frame_pil = frame_pil.resize((150, 150))  # Resize to match the model input size
    frame_np = np.array(frame_pil)  # Convert back to NumPy array
    frame_normalized = frame_np / 255.0  # Normalize pixel values (assuming your model was trained with normalized data)

    # Predict using the model
    input_frame = np.expand_dims(frame_normalized, axis=0)  # Add batch dimension
    predictions = model.predict(input_frame)
    predicted_class = np.argmax(predictions)
    predicted_label = class_labels[predicted_class]
    clear_output(wait=True)
    # Display the result using matplotlib
    plt.figure(figsize=(10, 6))
    plt.imshow(frame_normalized)
    plt.axis('off')

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time

    plt.title('Object Recognition: ' + predicted_label)
    plt.show()

    # clear_output(wait=True)  # Clear the previous frame



# Close the video reader
video_reader.close()
