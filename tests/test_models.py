# Test Neural Network Achitectures
# Documentation:
# https://docs.pytest.org/en/stable/contents.html

import numpy as np

from ganpdfs.model import WGanModel
from ganpdfs.model import DWGanModel
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import RMSprop

# Define Toy PDF set
NB_REPLICAS = 100
NB_FLAVORS = 14
XGRID_SIZE = 125
DNN_TOY_PDF = np.random.random((NB_REPLICAS, NB_FLAVORS, XGRID_SIZE))
CNN_TOY_PDF = np.random.random((NB_REPLICAS, NB_FLAVORS, XGRID_SIZE, 1))

# Dictionary containing the information
# on the Neural Architercture.
PARAMS = {
    "use_saved_model": False,
    "architecture": "cnn",
    "gan_parameters": {
        "optimizer": {"optimizer_name": "RMSprop", "learning_rate": 5e-05}
    },
    "gen_parameters": {
        "size_networks": 1,
        "number_nodes": 128,
        "use_bias": False,
        "bias_initializer": "zeros",
        "kernel_initializer": "glorot_uniform",
        "weights_constraints": 1,
        "optimizer": {"optimizer_name": "RMSprop", "learning_rate": 5e-05},
        "loss": "binary_crossentropy",
        "activation": "leakyrelu",
    },
    "disc_parameters": {
        "size_networks": 1,
        "number_nodes": 450,
        "use_bias": False,
        "bias_initializer": "zeros",
        "kernel_initializer": "glorot_uniform",
        "weights_constraints": 1,
        "optimizer": {"optimizer_name": "RMSprop", "learning_rate": 5e-05},
        "loss": "binary_crossentropy",
        "activation": "leakyrelu",
    },
    "ConvoluteOutput": False,
    "nd_steps": 4,
    "ng_steps": 3,
    "batch_size": 70,
    "epochs": 600,
}

OPTMZ = {"rms": RMSprop(lr=0.00005)}
ACTIV = {"leakyrelu": LeakyReLU(alpha=0.2)}

NOISE_DIM = 100

# Init DNN Model
DNNWGAN_MODEL = WGanModel(DNN_TOY_PDF, PARAMS)

# Init DCNN Model
DCNNWGAN_MODEL = DWGanModel(CNN_TOY_PDF, PARAMS)


def test_dnn_model():
    """`test_dcnn_model` test that the outputs of each Architecture
    of the DNN model is exactly as expected.
    """
    # Init Models
    critic = DNNWGAN_MODEL.critic_model()
    generator = DNNWGAN_MODEL.generator_model()
    adversarial = DNNWGAN_MODEL.adversarial_model(generator, critic)
    # Expected Output Shapes
    c_exp_shape = (None, 1)
    a_exp_shape = (None, 1)
    g_exp_shape = (None, NB_FLAVORS, XGRID_SIZE)
    # Check Shapes
    assert critic.output_shape == c_exp_shape
    assert generator.output_shape == g_exp_shape
    assert adversarial.output_shape == a_exp_shape


def test_dcnn_model():
    """`test_dcnn_model` test that the outputs of each Architecture
    of the DCNN model is exactly as expected.
    """
    # Init Models
    critic = DCNNWGAN_MODEL.critic_model()
    generator = DCNNWGAN_MODEL.generator_model()
    adversarial = DCNNWGAN_MODEL.adversarial_model(generator, critic)
    # Expected Output Shapes
    c_exp_shape = (None, 1)
    a_exp_shape = (None, 1)
    g_exp_shape = (None, NB_FLAVORS, XGRID_SIZE, 1)
    # Check Shapes
    assert critic.output_shape == c_exp_shape
    assert generator.output_shape == g_exp_shape
    assert adversarial.output_shape == a_exp_shape
