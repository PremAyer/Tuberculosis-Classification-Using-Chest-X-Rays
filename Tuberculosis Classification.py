!mkdir -p ~/.kaggle
!cp /content/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!pip install tensorflow numpy matplotlib kaggle -q

!kaggle datasets download -d tawsifurrahman/tuberculosis-tb-chest-xray-dataset -q
!unzip -q tuberculosis-tb-chest-xray-dataset.zip


#Importing the dependencies

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing import image


# --- Constants and Configuration ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32 # Number of images to process in a batch
DATA_DIR = "/content/TB_Chest_Radiography_Database"


# --- Create Data Generators ---

# 1. Training Data Generator with Augmentation
# Augmentation creates modified versions of the images to help the model generalize better.
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,          
    validation_split=0.2,   
    rotation_range=20,       
    width_shift_range=0.2,
    height_shift_range=0.2,  
    shear_range=0.2,         
    zoom_range=0.2,          
    horizontal_flip=True,    
    fill_mode='nearest'
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2)

# Training Set
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',  # For two classes (Normal, Tuberculosis)
    subset='training'     # Specify this is the training set
)

# Validation Set
validation_generator = validation_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'   # Specify this is the validation set
)

# Check the class indices
print("Class Indices:", train_generator.class_indices)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Flatten the feature maps into a 1D vector
    Flatten(),

    # Add a Dense (fully connected) layer
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    # Add a Dropout layer to prevent overfitting
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])

# Display the model's architecture
model.summary()


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


EPOCHS = 30 

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE


#Data Visualization

# Plotting training & validation accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plotting training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

def predict_image(image_path, model):
    """
    Loads an image, preprocesses it, and predicts its class using the trained model.
    """
    # Load the image
    img = image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))

    # Convert image to array
    img_array = image.img_to_array(img)

    # Expand dimensions to create a batch of 1
    img_batch = np.expand_dims(img_array, axis=0)

    # Preprocess the image (rescale)
    img_preprocessed = img_batch / 255.0

    # Make prediction
    prediction = model.predict(img_preprocessed)
    print(f"Prediction: {prediction}")

    # Get the class indices
    class_labels = {v: k for k, v in train_generator.class_indices.items()}

    # Display the image
    plt.imshow(img)
    plt.axis('off')

    # Interpret the prediction
    if prediction[0][0] > 0.5:
        predicted_class = class_labels[1]
        confidence = prediction[0][0] * 100
    else:
        predicted_class = class_labels[0]
        confidence = (1 - prediction[0][0]) * 100

    plt.title(f"Prediction: {predicted_class} ({confidence:.2f}%)")
    plt.show()

predict_image('/content/TB_Chest_Radiography_Database/Normal/Normal-1011.png',model)
predict_image('/content/TB_Chest_Radiography_Database/Tuberculosis/Tuberculosis-158.png',model)
predict_image('/content/TB_Chest_Radiography_Database/Tuberculosis/Tuberculosis-218.png',model)

model.save('tb_model.keras')

print("Model saved as tb_model.keras")
