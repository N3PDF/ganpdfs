How to use the code
===================


To install the **ganpdfs** package, just run the following:

.. code-block:: bash

   pip install --upgrade ganpdfs


The **ganpdfs** pogram takes as an input argument a run card file in which all the input parameters
such as the name of the PDF set and the architecture of the network models are defined.


.. code-block:: bash

   ganpdfs <runcard_folder>/<runcard_file>.yml


In addition, it takes optional arguments that are defined by the following flags:

    - ``-o`` / ``--output``: name of the output folder
    - ``-f`` / ``--force``: overwrite the previous folder
    - ``-n`` / ``--nreplicas``: number of input replicas to be considered (default: entire set)
    - ``k`` / ``--fake``: number of fake replicas to be generated (default: same size as the input)
    - ``--hyperopt``: number of hyper-parameter scan trials
    - ``--cluster``: enable hyper-parameter scan in cluster


In the run card file, the following keys are required:


    - ``pdf`` - *str* : name of the PDF set
    - ``q`` - *float* : initial scale at which the PDF grid will be computed
    - ``x_grid`` - *str* : format of the x-grid (*options: custom/lhapdf*)
    - ``architecture`` - *str*: architecture of the network models (*options: dnn/dcnn*)
    - ``g_nodes`` - *int* : number of nodes in the 1st layer of the Generator
    - ``d_nodes`` - *int* : number of nodes in the 1st layer of the Discriminator
    - ``gdnn_size`` - *int* : number of Generator's hidden layers (*optional, default=1*)
    - ``ddnn_size`` - *int* : number of Discriminator's hidden layers (*optional, default=1*)
    - ``g_act`` - *str* : activation function for the Generator
    - ``d_act`` - *str* : activation function for the Discriminator
    - ``batch_size`` - *int* : size of the batches per training
    - ``ng_steps`` - *int* : number of steps to train the Generator per epoch (*optional, default=3*)
    - ``nd_steps`` - *int* : number of steps to train the Discriminator per epoch (*optional, default=4*)
    - ``ConvoluteOutput`` - *bool* : convolute the output of the Generator with some samples from the input PDF


An example of standard run card is shown below:


.. code-block:: yaml

   #############################################################
   # Input PDF                                                 #
   #############################################################
   pdf: PN3_GLOBAL_NNPDF31_nnlo_as_0118_070219-001
   
   
   #############################################################
   # PDF Grids:                                                #
   # ---------                                                 #
   # * Inittial scale q (TODO: change to q^2)                  #
   # * Options for x-grid:                                     #
   #   - custom: Custom GANs xgrid as defined in the Module    #
   #   - lhapdf: Use the same xgrid as in the input PDF        #
   #############################################################
   q        : 1.65
   x_grid   : custom
   
   #############################################################
   # GAN setup:                                                #
   # ---------                                                 #
   # * Options for architecture:                               #
   #   - dnn : Deep Neural Network                             #
   #   - dcnn: Deep Convolutional Neural Network               #
   #############################################################
   architecture: dnn
   
   # Number of Nodes in 1st Layer
   g_nodes  : 128
   d_nodes  : 450
   
   # [Activations]
   g_act    : leakyrelu
   d_act    : leakyrelu
   
   # Architecture
   ddnn_size   : 1
   gdnn_size   : 1
   
   # Optimizers
   d_opt    : rms
   gan_opt  : rms
   epochs   : 5000
   
   # Intrinsic features
   ConvoluteOutput : True
   
   #############################################################
   # Training Setup:                                           #
   # --------------                                            #
   # * batch size                                              #
   # * {i}_steps: number of steps to train a {i}={generator,   #
   #              discriminator/critic} at each iteration.     #
   #############################################################
   batch_size : 64
   
   nd_steps : 4
   ng_steps : 3



In order a hyper-parameter scan, the run card has to be slightly modified:


.. code-block:: yaml

   #############################################################
   # Input PDF                                                 #
   #############################################################
   pdf : PN3_GLOBAL_NNPDF31_nnlo_as_0118_070219-001
   
   #############################################################
   # PDF Grids:                                                #
   # ---------                                                 #
   # * Inittial scale q (TODO: change to q^2)                  #
   # * Options for x-grid:                                     #
   #   - custom: Custom GANs xgrid as defined in the Module    #
   #   - lhapdf: Use the same xgrid as in the input PDF        #
   #############################################################
   q      : 1.65
   x_grid : custom
   
   # Intrinsic features
   ConvoluteOutput : True
   
   #############################################################
   # GAN setup:                                                #
   # ---------                                                 #
   # * Options for architecture:                               #
   #   - dnn : Deep Neural Network                             #
   #   - dcnn: Deep Convolutional Neural Network               #
   #############################################################
   architecture : dnn
   
   hyperopt:
       # Number of Nodes in 1st Layer
       g_nodes  : hp.choice('g_nodes', [64, 128, 200])
       d_nodes  : hp.choice('d_nodes', [128, 256, 450])
   
       # [Activations]
       g_act    : hp.choice('g_act', ['leakyrelu', 'elu', 'relu'])
       d_act    : hp.choice('d_act', ['leakyrelu', 'elu', 'relu'])
   
       # Architecture
       gdnn_size   : hp.choice('gdnn_size', [1, 2])
       ddnn_size   : hp.choice('ddnn_size', [1, 2])
   
       # Optimizers
       d_opt    : hp.choice('d_opt', ['adadelta', 'sgd', 'rms'])
       gan_opt  : hp.choice('gan_opt', ['adadelta', 'sgd', 'rms'])
   
       # Number of Epochs
       epochs   : hp.choice('epochs', [4000, 5000, 6000])
   
       #############################################################
       # Training Setup:                                           #
       # --------------                                            #
       # * batch size                                              #
       # * {i}_steps: number of steps to train a {i}={generator,   #
       #              discriminator/critic} at each iteration.     #
       #############################################################
       batch_size : hp.choice('batch_size', [50, 64, 70])
   
       nd_steps : hp.choice('nd_steps', [2, 4, 5])
       ng_steps : hp.choice('ng_steps', [2, 3, 4])


During the hyper-parameter scan, the models are optimized w.t.r to a similarity metric measue known
as the **Fréchet Inception Distance** which measure the quality of the generated PDF.



Post-Analysis
=============
