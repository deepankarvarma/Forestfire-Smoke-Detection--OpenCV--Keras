import cv2
import numpy as np
from keras.models import load_model

# load the smoke detection model
model = load_model('smoke_detection_model.h5')

# define a function to apply the smoke detection model to a single frame
def detect_smoke(frame):
    # resize the frame to match the input size of the model
    resized_frame = cv2.resize(frame, (224, 224))
    
    # convert the frame to a batch of size 1 and normalize the pixel values
    batch = np.expand_dims(resized_frame, axis=0) / 255.0
    
    # apply the model to the batch and get the predicted class
    prediction = model.predict(batch)[0][0]
    predicted_class = int(np.round(prediction))
    
    # if the predicted class is 1, then smoke is present in the frame
    if predicted_class == 1:
        return True
    else:
        return False

# prompt the user to enter the path of the video file
video_path = input("Enter the path of the video file: ")

# open the video file using OpenCV
cap = cv2.VideoCapture(video_path)

# loop over the frames of the video
while cap.isOpened():
    # read a single frame from the video
    ret, frame = cap.read()
    
    # if the frame was read successfully, then apply the smoke detection model to it
    if ret:
        # apply the smoke detection model to the frame
        is_smoke = detect_smoke(frame)
        
        # if smoke is present, then draw a red bounding box around the detected smoke
        if is_smoke:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 2)
        
        # display the frame
        cv2.imshow('frame', frame)
        
        # wait for a key press and exit the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# release the video file and close all windows
cap.release()
cv2.destroyAllWindows()