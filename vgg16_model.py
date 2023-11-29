import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam

# Load pre-trained VGG16 model
model = VGG16(include_top=False, input_shape=(300, 300, 3))

# Freeze the layers of the pre-trained model
for layer in model.layers:
    layer.trainable = False

# Define data augmentation parameters
data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255.,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% of data for validation
)

# Load training and validation data
path = './data/images_original'
train_datagen = data_gen.flow_from_directory(
    path, target_size=(300, 300), batch_size=32, class_mode='categorical', subset='training'
)
val_datagen = data_gen.flow_from_directory(
    path, target_size=(300, 300), batch_size=32, class_mode='categorical', subset='validation'
)

# Define model architecture
output = model.layers[-1].output
model_final = tf.keras.layers.Flatten()(output)
model_final = tf.keras.layers.Dense(512, activation='relu')(model_final)
model_final = tf.keras.layers.Dense(64, activation='relu')(model_final)
model_final = tf.keras.layers.Dense(10, activation='softmax')(model_final)
model = tf.keras.models.Model(model.input, model_final)

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['acc'])

# ModelCheckpoint callback to save the best weights
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# LearningRateScheduler callback for dynamic learning rate adjustment
def lr_schedule(epoch):
    if epoch < 5:
        return 0.001
    elif 5 <= epoch < 8:
        return 0.0005
    else:
        return 0.0001

lr_scheduler = LearningRateScheduler(lr_schedule)

# Train the model with training and validation data
model.fit(
    train_datagen,
    epochs=10,
    validation_data=val_datagen,
    callbacks=[checkpoint, lr_scheduler],
    verbose=2
)
