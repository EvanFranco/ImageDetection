# Import necessary libraries
import cv2  # OpenCV library for video processing
import cvlib as cv  # Computer vision library for object detection
from cvlib.object_detection import draw_bbox  # Function to draw bounding boxes around detected objects

# Initialize the video capture object to use the default camera (0 is typically the built-in webcam)
video = cv2.VideoCapture(0)  
labels = []  # List to store detected objects without duplication

# Start loop to read frames from the camera
while True:
    ret, frame = video.read() 
    # Perform object detection on the frame
    bbox, label, conf = cv.detect_common_objects(frame)  
    # Draw bounding boxes around detected objects with labels
    output_image = draw_bbox(frame, bbox, label, conf)
    # Display the processed frame with bounding boxes
    cv2.imshow("Object Detection", output_image)
    # Check if detected items are already in the labels list; add only new items
    for item in label:
        if item not in labels:  
            labels.append(item)  # Add new object to the labels list
    # Break the loop when the key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):  
        break

# After the loop, create a sentence with the detected objects
i = 0
new_sentence = []  # List to store parts of the final sentence
# Loop through the detected labels to construct the sentence
for label in labels:
    if i == 0:
        # Add the first object with an introductory phrase
        new_sentence.append(f"I found a {label}, and,")  
    else:
        # Add subsequent objects
        new_sentence.append(f"a {label}")  
    i += 1  # Increment the counter
# Print the constructed sentence by joining list elements with spaces
print(" ".join(new_sentence))

