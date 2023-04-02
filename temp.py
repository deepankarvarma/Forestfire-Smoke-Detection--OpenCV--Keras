import cv2
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('smoke_detection_model.h5')

# Load the image
img = cv2.imread('1.jpg')

# Define a function to preprocess the image
def preprocess_image(image):
    # Resize the image to the input size of the model (224x224)
    resized_image = cv2.resize(image, (224, 224))
    # Convert the image to a format that can be used by the model (float32 array)
    input_image = resized_image.astype('float32') / 255.0
    # Add an extra dimension to the input to match the input shape of the model (batch size of 1)
    input_image = tf.expand_dims(input_image, axis=0)
    return input_image

# Preprocess the image
input_image = preprocess_image(img)

# Run the model on the input image
prediction = model.predict(input_image)[0][0]

# If the prediction is greater than a threshold value (e.g. 0.1), print "Smoke detected", otherwise print "No smoke detected"
if prediction > 0.1:
    print("Smoke detected")
else:
    print("No smoke detected")
