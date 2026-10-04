import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1️⃣ Load CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# One-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2️⃣ Data augmentation và preprocessing
IMG_SIZE = 224
BATCH_SIZE = 64

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(
    x_train, y_train,
    batch_size=BATCH_SIZE
)

test_generator = test_datagen.flow(
    x_test, y_test,
    batch_size=BATCH_SIZE
)

# 3️⃣ Build model with transfer learning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Freeze base model

model = models.Sequential([
    layers.Resizing(IMG_SIZE, IMG_SIZE),  # Resize theo batch
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 4️⃣ Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5️⃣ Train model
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10)

# 6 Evaluate
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.4f}")
