import tensorflow as tf
import determined as det
from determined.experimental import Determined
from tensorflow.keras import layers
import determined.experimental as exp

# Define the model architecture
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.max_pool1 = layers.MaxPooling2D((2, 2))
        self.dropout1 = layers.Dropout(0.25)
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout2 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.max_pool1(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout2(x)
        return self.dense2(x)

# Define the data loader
def data_loader(config):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(y_test)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(config["batch_size"])
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(config["batch_size"])
    return train_ds, test_ds

# Define the training function
def train(config):
    strategy = det.keras.DistributedStrategy()
    with strategy.scope():
        model = MyModel()
        model.compile(optimizer=tf.keras.optimizers.Adam(config["learning_rate"]),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])
    train_ds, val_ds = data_loader(config)
    model.fit(train_ds,
              epochs=config["epochs"],
              validation_data=val_ds,
              callbacks=[det.keras.TFKerasCallback(evaluation_interval=1)])
              
# Create a Determined experiment
experiment_config = {
    "hyperparameters": {
        "learning_rate": {
  "type": "log",
  "base": 10,
  "minval": -4,
  "maxval": -2,
},
    },
    "searcher": {
        "name": "single",
        "metric": "val_categorical_accuracy",
        "max_steps": 10
    },
    "resources": {
        "slots_per_trial": 8,
        "native_parallel": False
    },
    "data": {
        "train_data": {
            "type": "numpy",
            "train_images": "/path/to/train/images.npy",
            "train_labels": "/path/to/train/labels.npy"
        },
        "val_data": {
            "type": "numpy",
            "val_images": "/path/to/val/images.npy",
            "val_labels": "/path/to/val/labels.npy"
        }
    }
}
determined = Determined()
determined.create_experiment(experiment_config, train_fn=train, data_loader=data_loader)


