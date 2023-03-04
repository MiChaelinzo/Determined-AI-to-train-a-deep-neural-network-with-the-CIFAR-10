import determined as det
from determined.experimental import Determined

# Define the model architecture
class SimpleModel(det.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = det.keras.layers.Flatten()
        self.dense1 = det.keras.layers.Dense(128, activation='relu')
        self.dense2 = det.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

# Define the data loader
def data_loader(config):
    (x_train, y_train), (x_test, y_test) = det.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return det.keras.utils.to_categorical(y_train), det.keras.utils.to_categorical(y_test)

# Define the training function
def train(config):
    model = SimpleModel()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    train_data, val_data = data_loader(config)
    model.fit(train_data[0], train_data[1],
              validation_data=(val_data[0], val_data[1]),
              epochs=config['num_epochs'])

# Create a Determined experiment
experiment_config = {
    'hyperparameters': {
        'num_epochs': det.Constant(value=10),
        'learning_rate': det.Double(value=0.001, min=0.0001, max=0.01, scale='log')
    },
    'searcher': {
        'name': 'single',
        'metric': 'val_accuracy',
        'max_steps': 10
    },
    'data': {
        'train_data': 'train_data',
        'val_data': 'val_data'
    }
}
determined = Determined()
determined.create_experiment(experiment_config, train_fn=train, data_loader=data_loader)

