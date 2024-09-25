import os
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers, models

# Define folder paths for training and testing
train_folder = 'dataset/train'
test_folder = 'dataset/test'

# Define classes and their severity levels
classes = ['acne', 'darkspots', 'wrinkles']
severity_levels = ['mild', 'moderate', 'severe']

# Create data generators with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Define parameters
batch_size = 8
img_height = 128
img_width = 128
num_classes = len(classes)

# Load and preprocess images from train and test folders using the data generators
train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    test_folder,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Define a function to create a model
def create_model():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    for layer in base_model.layers[:100]:
        layer.trainable = False
    flatten_layer = layers.Flatten()(base_model.output)
    dense_layer_1 = layers.Dense(512, activation='relu')(flatten_layer)
    output_layer = layers.Dense(num_classes, activation='softmax')(dense_layer_1)
    model = models.Model(inputs=base_model.input, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Calculate steps_per_epoch and validation_steps
steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

# Create and train models for each class
models_dict = {}
for class_name in classes:
    print(f"Training model for class: {class_name}")
    model = create_model()
    early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=2)
    model_checkpoint = ModelCheckpoint(f"{class_name}_best_model.keras", save_best_only=True)
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=[early_stopping, reduce_lr, model_checkpoint]
    )

    models_dict[class_name] = model

# Evaluate each model on the test set
for class_name, model in models_dict.items():
    loss, accuracy = model.evaluate(validation_generator)
    print(f'{class_name} - Test accuracy: {accuracy}')

# Save each model
for class_name, model in models_dict.items():
    model.save(f'{class_name}_final_model.keras')