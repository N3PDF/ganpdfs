How to use the code
===================


To install the **ganpdfs** package, just run the following:

.. code-block:: bash

   pip install --upgrade ganpdfs


The **ganpdfs** pogram takes as an input argument a run card file in which all the input parameters
such as the name of the PDF set and the architecture of the network models are defined.


.. code-block:: bash

   ganpdfs runcard/reference.yml [-t TOT_REPLICAS_SIZE]


In addition, it takes optional arguments that are defined by the following flags:

    - ``-o`` / ``--output``: name of the output folder
    - ``-f`` / ``--force``: overwrite the previous folder
    - ``-t`` / ``--totrep``: total number of replicas (Np+Ns)
    - ``-s`` / ``--hyperopt``: number of hyper-parameter scan trials
    - ``-c`` / ``--cluster``: enable hyper-parameter scan in cluster


In the run card file, the following keys are required:


    - ``pdf`` - *str* : name of the PDF set
    - ``q`` - *float* : initial scale at which the PDF grid will be computed
    - ``x_grid`` - *str* : format of the x-grid (*options: custom/lhapdf*)
    - ``architecture`` - *str*: network models (*options: dnn/dcnn*)
    - ``gen_parameters`` - *dict*: architecture of the Generator
    - ``disc_parameters`` - *dict*: architecture of the Discriminator
    - ``gan_parameters`` - *dict*: parameters that defines the adversarial network
    - ``batch_size`` - *int* : size of the batches per training (in prercentage)
    - ``ng_steps`` - *int* : number of steps to train the Generator per epoch (*optional, default=3*)
    - ``nd_steps`` - *int* : number of steps to train the Discriminator per epoch (*optional, default=4*)


An example of standard run card is shown below:


.. code-block:: yaml

   #############################################################################################
   # Input PDF                                                                                 #
   #############################################################################################
   pdf: 210127-n3fit-002

   #############################################################################################
   # PDF Grids:                                                                                #
   # ---------                                                                                 #
   # * Inittial scale q (in GeV)                                                               #
   # * Options for x-grid:                                                                     #
   #   - custom: Custom GANs xgrid as defined in the Module                                    #
   #   - lhapdf: Use the same xgrid as in the input PDF                                        #
   #############################################################################################
   q        : 1.65
   x_grid   : standard

   #############################################################################################
   # GAN setup:                                                                                #
   # ---------                                                                                 #
   # * Options for architecture:                                                               #
   #   - dnn : Deep Neural Network                                                             #
   #   - dcnn: Deep Convolutional Neural Network                                               #
   #############################################################################################
   use_saved_model       : False


   architecture          : cnn

   gan_parameters:
     optimizer:
       optimizer_name    : RMSprop
       learning_rate     : 0.00005
     loss                : wasserstein

   gen_parameters:
     size_networks       : 2
     kernel_initializer  : glorot_uniform

   disc_parameters:
     size_networks       : 3
     number_nodes        : 450
     use_bias            : False
     bias_initializer    : zeros
     kernel_initializer  : glorot_uniform
     weights_constraints : 0.01
     optimizer:
       optimizer_name    : RMSprop
       learning_rate     : 0.00005
     loss                : wasserstein
     activation          : leakyrelu

   ConvoluteOutput       : False

   #############################################################################################
   # Training Setup:                                                                           #
   # --------------                                                                            #
   # * batch size                                                                              #
   # * {i}_steps: number of steps to train a {i}={generator, discriminator/critic} at each     #
   #   iteration.                                                                              #
   #############################################################################################
   nd_steps   : 4
   ng_steps   : 1
   batch_size : 70
   epochs     : 1000


In order a hyper-parameter scan, the run card has to be slightly modified:


.. code-block:: yaml

   #############################################################################################
   # Input PDF                                                                                 #
   #############################################################################################
   pdf: NNPDF40_nnlo_as_0118_1000rep

   #############################################################################################
   # PDF Grids:                                                                                #
   # ---------                                                                                 #
   # * Inittial scale q (in GeV)                                                               #
   # * Options for x-grid:                                                                     #
   #   - custom: Custom GANs xgrid as defined in the Module                                    #
   #   - lhapdf: Use the same xgrid as in the input PDF                                        #
   #############################################################################################
   q        : 1.65
   x_grid   : standard

   use_saved_model : False

   ConvoluteOutput : False

   architecture    : cnn

   hyperopt:
     #############################################################################################
     # GAN setup:                                                                                #
     # ---------                                                                                 #
     # * Options for architecture:                                                               #
     #   - dnn : Deep Neural Network                                                             #
     #   - dcnn: Deep Convolutional Neural Network                                               #
     #############################################################################################
     gan_parameters:
       optimizer:
         optimizer_name    : hp.choice('gan_opt', ['RMSprop', 'Adadelta'])
         learning_rate     : hp.choice('gan_lr', [0.00005, 0.0005])

     gen_parameters:
       size_networks       : hp.choice('g_nn', [1, 2])
       kernel_initializer  : hp.choice('g_kini', ['GlorotUniform', 'RandomUniform'])

     disc_parameters:
       size_networks       : hp.choice('d_nn', [1, 2])
       number_nodes        : hp.choice('d_nodes', [250, 450, 650, 1000])
       kernel_initializer  : hp.choice('d_kini', ['GlorotUniform', 'RandomUniform'])
       weights_constraints : hp.choice('d_wc', [0.01, 0.1, 1])
       optimizer:
         optimizer_name    : hp.choice('d_opt', ['RMSprop', 'Adadelta'])
         learning_rate     : hp.choice('d_lr', [0.00005, 0.0005])
       activation          : hp.choice('d_act', ['relu', 'leakyrelu'])
       trainable           : hp.choice('d_train', [True, False])


     #############################################################################################
     # Training Setup:                                                                           #
     # --------------                                                                            #
     # * batch size                                                                              #
     # * {i}_steps: number of steps to train a {i}={generator, discriminator/critic} at each     #
     #   iteration.                                                                              #
     #############################################################################################
     nd_steps   : hp.choice('nd_steps', [2, 3, 4])
     ng_steps   : hp.choice('ng_steps', [1, 2, 3])
     batch_size : hp.choice('batch_size', [30, 50, 70])
     epochs     : hp.choice('epochs', [1000, 1500, 2000])

During the hyper-parameter scan, the models are optimized w.t.r to a similarity metric measue known
as the **Fréchet Inception Distance** which measure the quality of the generated PDF.



How to generate PDF grid
========================


The above will generate a folder named after the prior PDF name, appended with **_enhanced**. In order
to evolve the PDFs, it suffices to run:


.. code-block:: bash

   evolven3fit <PRIOR_PDF_NAME>_enhanced <TOT_REPLICAS_SIZE>


Then, to link the enhanced PDF set to the LHAPDF data directory, run the following:


.. code-block:: bash

   postgans --pdf <PRIOR_PDF_NAME> --nenhanced <TOT_REPLICAS_SIZE>
