# Script for evaluating the model

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate_model(model_path, data_dir):
    """Evaluate the model on the test dataset."""
    model = tf.keras.models.load_model(model_path)

    datagen = ImageDataGenerator()
    test_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )

    loss, accuracy = model.evaluate(test_gen)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

if __name__ == "__main__":
    evaluate_model("models/crop_disease_model.h5", "Data/Processed")