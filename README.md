### GANPDFs

Enhance the statistics of a prior PDF set by generating fake PDF replicas using Generative
Adversarial Neural Networks ([GANs](https://arxiv.org/abs/1406.2661)). Detailed documentation
will be available in https://n3pdf.github.io/ganpdfs/.

#### How to install

To install the `ganpdfs` package, just type
```bash
python setup.py install
```
or if you are a developer
```bash
python setup.py develop
```
The package can be installed via the Python Package Index (PyPI) by running:
```bash
pip install ganpdfs --upgrade
```

#### How to run

The code requires as an input a `runcard.yml` file in which the name of the PDF set and the characteristics 
of the Neural Networks Model are defined. Examples of runcards can be found in the `runcard` folder.
```bash
ganpdfs <runcard> -o <output> [--force] [--hyperopt n] [--nreplicas n] [--cluster url]
```
This will generate a folder named `<output>` that has the following structure:
```bash
output_folder/
├── checkpoint
├── iterations
└── nnfit
```
where `checkpoint` contains the saved models throughout the training, which can be helpful in case a long 
running training task is interrupted; `iterations` contains the information on the performance of the
models at each iteration, and `nnfit` contains the output grids from the generated replicas (this has
the exact same structure as the output from N3FIT). 

Hence, in order to evolve the generated output grids, just run:
```bash
evolven3fit <output>
```

#### Parameter Hyper-Optimization

The framework of parameter optimizations is currently being developed in a separete branch 
([DynamicArchitecture](https://github.com/N3PDF/ganpdfs/tree/DynamicArchitecture)).


#### Bottlenecks

Currently, in order to evolve the output grids from the GANs, the `filter.yml` file that contains
the information on theory ID used to generate the prior replicas has to be manually put in the
`<output>` folder. A systematic way to deal with this has to be implemented.
