![pytest](https://github.com/N3PDF/ganpdfs/workflows/pytest/badge.svg)
[![documentation](https://github.com/N3PDF/ganpdfs/workflows/docs/badge.svg)](https://n3pdf.github.io/ganpdfs/)
[![DOI](https://zenodo.org/badge/178228962.svg)](https://zenodo.org/badge/latestdoi/178228962)

### GANPDFs

Enhance the statistics of a prior PDF set by generating fake PDF replicas using Generative
Adversarial Neural Networks ([GANs](https://arxiv.org/abs/1406.2661)). Documentation
is available at https://n3pdf.github.io/ganpdfs/.

#### How to install

To install the `ganpdfs` package, just type
```bash
python setup.py install or python setup.py develop (if you are a developper)
```
The package can be installed via the Python Package Index (PyPI) by running:
```bash
pip install ganpdfs --upgrade
```

#### How to run

The code requires as an input a `runcard.yml` file in which the name of the PDF set and the
characteristics of the Neural Network Models are defined. Examples of runcards can be found
in the `runcard` folder.
```bash
ganpdfs runcard/reference.yml [-t TOT_REPLICAS_SIZE]
```
In case one does not want to train the GANs and directly resort to a pre-trained one, a pre-trained
[model](https://github.com/N3PDF/ganpdfs/tree/DynamicArchitecture/pre-trained-model)
can be used out of the box by setting the entry `use_saved_model` to `True` in the runcard. 

In order to evolve the generated output grids, just run:
```bash
evolven3fit <PRIOR_PDF_NAME>_enhanced <TOT_REPLICAS_SIZE>
```

Then, to link the generated PDF set to the LHAPDF data directory, use the `postgans` script by
running:
```bash
postgans --pdf <PRIOR_PDF_NAME> --nenhanced <TOT_REPLICAS_SIZE>
```

#### Hyper-parameter opitmization

For more details on how to define specific parameters when running the code and on how to perform
a hyper-parameter scan, please head to the section [how to](https://n3pdf.github.io/ganpdfs/howto/howto.html)
of the documentation.
