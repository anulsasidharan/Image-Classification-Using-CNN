import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (128,128)
BATCH_SIZE = 16

# train and val data rescaling 
train_datagen = ImageDataGenerator(rescale=1./255)    
val_datagen = ImageDataGenerator(rescale = 1./255)

# Read the datagen from the data directory
train_data = train_datagen.flow_from_directory(
    'data/train',
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
)

val_data = train_datagen.flow_from_directory(
    'data/val',
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
)

# This is required because we need to apply softmax at the end where we need to find the 
# probability of each category. 
num_class = len(train_data.class_indices)

# Create the model

model = models.Sequential(
    [
        layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (128,128,3), padding = 'same'),
        # 32, (3,3) -- selecting 32 Kernel with size of the kernel metrix is '3x3', 
        # input_shape -- (128,128,3), select 128x128 size with all 
        # three channels(red, green and blue)
        layers.MaxPooling2D(2,2),

        layers.Conv2D(32, (3,3), activation = 'relu', padding = 'same'),# Only in the first layer, we need to pass the input_shape
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same'),
        layers.MaxPooling2D(2,2),

        # Once we have all the information, do a Flatter layer
        layers.Flatten(),
        layers.Dense(128, activation = 'relu'),
        layers.Dropout(0.3),
        layers.Dense(num_class, activation = 'softmax'),
    ]
)

# Define the loss function and Optimizer

model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)

model.fit(
    train_data,
    epochs = 10, # How many times we need to send our data
    validation_data = val_data
)

# Save the model
model.save('cnn_classifier.h5')

print(model.summary())