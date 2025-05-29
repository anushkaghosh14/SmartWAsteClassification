import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# Define constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = "waste_classifier.h5"
TRAIN_DIR = "C:/Users/91914/Downloads/Dataset/train"
VAL_DIR = "C:/Users/91914/Downloads/Dataset/val"

def create_model():
    """Builds and compiles the MobileNetV2 model."""
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    
    # Freeze first 100 layers, fine-tune last 20
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output_layer = Dense(1, activation="sigmoid")(x)

    # Compile the model
    model = Model(inputs=base_model.input, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])
    
    return model

def train_model():
    """Loads data, trains the model, and saves it."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary")
    val_generator = val_datagen.flow_from_directory(VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary")

    # Compute class weights
    class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=np.concatenate([
        np.zeros(12634), np.ones(1467) 
    ]))
    class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

    # Train and save the model
    model = create_model()
    model.fit(train_generator, epochs=10, validation_data=val_generator, class_weight=class_weights_dict)
    model.save(MODEL_PATH)
    print(f"Model saved as {MODEL_PATH}")

def preprocess_frame(frame):
    """Preprocesses an image frame for model prediction."""
    frame = cv2.resize(frame, IMG_SIZE)  # Resize to model input size
    frame = frame.astype(np.float32) / 255.0  # Normalize pixel values
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

def classify_realtime():
    """Loads model and classifies waste in real-time using webcam."""
    if not os.path.exists(MODEL_PATH):
        print("No trained model found. Training a new model...")
        train_model()
    
    # Load the trained model
    model = load_model(MODEL_PATH)

    # Open webcam
    cap = cv2.VideoCapture(0)
    print("Press 's' to capture and classify, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error accessing camera.")
            break

        cv2.imshow("Press 's' to classify waste", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Capture and classify
            processed_frame = preprocess_frame(frame)
            prediction = model.predict(processed_frame)[0][0]

            label = "Non-Biodegradable 🚫" if prediction >= 0.5 else "Biodegradable ✅"
            print(f"Predicted: {label}")

            # Display result on frame
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Classification Result", frame)
            cv2.waitKey(2000)  # Show result for 2 seconds

        elif key == ord('q'):  # Quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    classify_realtime()
