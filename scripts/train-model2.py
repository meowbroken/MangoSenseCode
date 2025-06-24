import tensorflow as tf
import os

# Dataset paths
base_dir = 'datasets/split-mango-leaves'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Count images in each class in the training directory
for cls in os.listdir(train_dir):
    count = len(os.listdir(os.path.join(train_dir, cls)))
    print(f"{cls}: {count} images")

# Load datasets (categorical labels)
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical',
    shuffle=True
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

# Number of classes (you have 8 classes in your example)
num_classes = len(train_dataset.class_names)
print(train_dataset.class_names)

# Load MobileNetV2 base model with pretrained ImageNet weights, exclude the top layer
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')

# Freeze the base model to keep its weights fixed initially
base_model.trainable = False

# Build your model on top of MobileNetV2 base
inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)  # preprocess input as MobileNet expects
x = base_model(x, training=False)  # pass inputs through the base model
x = tf.keras.layers.GlobalAveragePooling2D()(x)  # global pooling to flatten features
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)  # final classification layer

model = tf.keras.Model(inputs, outputs)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks for early stopping and learning rate reduction
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train the top layers first (base frozen)
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[early_stopping, reduce_lr]
)

# Now unfreeze some layers of the base model for fine-tuning
base_model.trainable = True

# Optional: freeze the first few layers, only train last layers for fine-tuning
fine_tune_at = 100  # freeze layers before this index
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile the model with a lower learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Continue training (fine-tuning)
fine_tune_epochs = 10
total_epochs = 10 + fine_tune_epochs

history_fine = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.2f}")

# Save the fine-tuned model
model.save('models/mobilenetv2_finetuned_mango.keras')
