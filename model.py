import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the CNN model architecture
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Define the data generators for the training, validation, and testing sets
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    './firesmoke/train/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    classes=['smoke']
)

valid_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
valid_generator = valid_datagen.flow_from_directory(
    './firesmoke/valid/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    classes=['smoke']
)

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    './firesmoke/test/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    classes=['smoke']
)

# Train the model
model.fit(train_generator, validation_data=valid_generator, epochs=10)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)

# save the model to a file
model.save('smoke_detection_model.h5')