## GANPDFs

Enhance the statistics of a prior PDF set by generating fake PDF replicas using Generative
Adversarial Neural Networks ([GANs](https://arxiv.org/abs/1406.2661)).

### How to install

To install the `ganpdfs` package, just type
```bash
python setup.py install
```
or if you are a developer
```bash
python setup.py develop
```

### How to run

The code can be run by feeding to it a `runcard.yml` file in which the name of the PDF set
and the characteristics of the Neural Networks Model are defined. Examples of runcards can
be found in the `runcard` folder.
```bash
ganpdfs <runcard> -o <output> [--force] [--hyperopt n] [--nreplicas n] [--cluster url]
```
