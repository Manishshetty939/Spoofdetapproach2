import os
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_preprocessing import get_data_generators

#path
train_dir = "datasets/train"
val_dir = "datasets/val"
test_dir = "datasets/test"

# from preprocessing
train_generator, val_generator, test_generator = get_data_generators(train_dir, val_dir, test_dir)

# Loading MobileNetV2 model pre-trained on ImageNet without to fully connected layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

base_model.trainable = False

# model structure
model = Sequential([
    base_model, 
    layers.GlobalAveragePooling2D(),  
    layers.Dense(512, activation='relu'),  # Fully connected layer
    layers.Dropout(0.5),  
    layers.Dense(1, activation='sigmoid')  # Output layer
])

# Compile 
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint("models/iris_spoof_model.h5", save_best_only=True)


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[early_stopping, model_checkpoint]
)

test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f"Test Accuracy: {test_acc * 100:.2f}%")