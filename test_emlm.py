import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from dscribe.descriptors import MBTR
from ase.io.trajectory import Trajectory

"""
This parameter is used to specify the training set.
After DFT/MD calculation is finished, this part shouldn't be modified.
"""
FILENAME_PREFIX = "test_1533618-amminoborane"

"""
These parameters are used to produce MBTR descriptors.
After MBTR descriptors are generated, this part shouldn't be modified.
"""
class MBTR_PARAMETERS:
    MIN = 0
    MAX = 0.5
    SIGMA = 0.01
    N = 50
    THRESHOLD = 1e-3
    DECAY = 0.5

"""
This parameter is used to define the model hyperparameters.
After the models are defined, this part shouldn't be modified.
"""
L2_REGULARIZER = 0.1

"""
These parameters are for guiding the learning process, and
can be modified at any time.
"""
class TRAIN_PARAMETERS:
    LEARNING_RATE = 4 * 1e-7
    BATCH_SIZE = 1
    INITIAL_LEARNING_RATE = 1 * 1e-8
    EPOCHS = 3000

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def kernel_generator(data_set):
    """
    Given the input dataset, generate the corresponding function
    for calculating Euclid distance vector with all data points
    as reference points.
    """
    data_set = [item for item in data_set]
    base_x = tf.math.reduce_sum(tf.math.square(data_set), axis=1)
    def kernel(y):
        base_y = tf.broadcast_to(tf.math.reduce_sum(tf.math.square(y)), base_x.shape)
        disquare = base_x + base_y - 2 * tf.math.reduce_sum(tf.matmul(data_set, tf.transpose([y])), axis=1)
        return tf.math.sqrt(tf.math.abs(disquare))
    return kernel

class TrainData:
    def __init__(self):
        self.kernel = lambda x: x
        self.train_data_x_prekernel = tf.data.Dataset.from_tensor_slices([])
        self.train_data_x = tf.data.Dataset.from_tensor_slices([])
        self.train_data_y = tf.data.Dataset.from_tensor_slices([])

    def save_variables(self, filename_prefix):
        tf.data.experimental.save(self.train_data_x_prekernel, f"./outputs/{filename_prefix}/dataset/train_data_x_prekernel")
        tf.data.experimental.save(self.train_data_x, f"./outputs/{filename_prefix}/dataset/train_data_x")
        tf.data.experimental.save(self.train_data_y, f"./outputs/{filename_prefix}/dataset/train_data_y")

    def restore_variables(self, filename_prefix):
        self.train_data_x_prekernel = tf.data.experimental.load(f"./outputs/{filename_prefix}/dataset/train_data_x_prekernel")
        self.train_data_x = tf.data.experimental.load(f"./outputs/{filename_prefix}/dataset/train_data_x")
        self.train_data_y = tf.data.experimental.load(f"./outputs/{filename_prefix}/dataset/train_data_y")
        self.kernel = kernel_generator(self.train_data_x_prekernel)

def main():
    """
    calculate data and train model.
    """
    mbtr = MBTR(
        species=["H", "B", "N"],
        k2={
            "geometry": {"function": "inverse_distance"},
            "grid": {"min": MBTR_PARAMETERS.MIN, "max": MBTR_PARAMETERS.MAX, "sigma": MBTR_PARAMETERS.SIGMA, "n": MBTR_PARAMETERS.N},
            "weighting": {"function": "exp", "scale": MBTR_PARAMETERS.DECAY, "threshold": MBTR_PARAMETERS.THRESHOLD},
        },
        periodic=True,
        flatten=False,
        sparse=False
    )

    train_data = TrainData()
    if os.path.exists(f"./outputs/{FILENAME_PREFIX}/dataset"):
        train_data.restore_variables(FILENAME_PREFIX)
        train_data_x = train_data.train_data_x
        train_data_y = train_data.train_data_y
    else:
        frames_mbtr = []
        frames_energy = []
        traj = Trajectory(f"./data_archive/{FILENAME_PREFIX}.traj")
        # traj = traj[:20]
        for i, atoms in enumerate(traj):
            frames_mbtr.append(tf.constant(np.array(mbtr.create(atoms)["k2"]).flatten()))
            frames_energy.append(atoms.get_potential_energy())
            if (i + 1) % 10 == 0:
                print(f"Reading trajectory frame {i}th")
        train_data_x_prekernel = tf.data.Dataset.from_tensor_slices(frames_mbtr)
        train_data_y = tf.data.Dataset.from_tensor_slices(frames_energy)
        kernel = kernel_generator(train_data_x_prekernel)
        train_data_x = train_data_x_prekernel.map(kernel)
        train_data.train_data_x_prekernel = train_data_x_prekernel
        train_data.train_data_x = train_data_x
        train_data.train_data_y = train_data_y
        train_data.save_variables(FILENAME_PREFIX)

    shape = train_data_x.element_spec.shape

    if os.path.exists(f"./outputs/{FILENAME_PREFIX}/model"):
        model = keras.models.load_model(f"./outputs/{FILENAME_PREFIX}/model")
        keras.backend.set_value(model.optimizer.learning_rate, TRAIN_PARAMETERS.LEARNING_RATE)
    else:
        model = keras.Sequential([
            keras.layers.Input(shape=shape),
            keras.layers.Rescaling(1./100),
            keras.layers.Dense(1, activation=None, kernel_regularizer=keras.regularizers.L2(l2=L2_REGULARIZER))
        ])

        sgd = keras.optimizers.SGD(
            learning_rate=TRAIN_PARAMETERS.INITIAL_LEARNING_RATE,
            momentum=0.0,
            nesterov=False,
            name='SGD',
        )

        model.compile(optimizer=sgd, loss=keras.losses.MeanSquaredError())

    history = model.fit(tf.data.Dataset.zip((train_data_x, train_data_y)).batch(TRAIN_PARAMETERS.BATCH_SIZE),
                        epochs=TRAIN_PARAMETERS.EPOCHS)
    for i, item in enumerate(tf.data.Dataset.zip((train_data_x, train_data_y))):
        # print(item)
        print(model.predict(tf.reshape(item[0], [1] + shape))[0][0])
        print(item[1])
        if i >= 5:
            break

    model.save(f"./outputs/{FILENAME_PREFIX}/model")

    hist_df = pd.DataFrame(history.history) 

    # save to json:  
    hist_json_file = f"./outputs/{FILENAME_PREFIX}/history.json"
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)

    # save to csv: 
    hist_csv_file = f"./outputs/{FILENAME_PREFIX}/history.csv"
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

if __name__ == "__main__":
    main()