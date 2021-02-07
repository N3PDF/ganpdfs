### GANPDFs

Enhance the statistics of a prior PDF set by generating fake PDF replicas using Generative
Adversarial Neural Networks ([GANs](https://arxiv.org/abs/1406.2661)). Detailed documentation
will be available in https://n3pdf.github.io/ganpdfs/.

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
ganpdfs runcard/default.yml [--fake TOT_FAKE_SIZE]
```
This will generate the following folders:
```bash
pre-trained-model/
├── assets
├── saved_model.pb
└── variables
    ├── variables.data-00000-of-00001
    └── variables.index
<PRIOR_PDF_NAME>_enhanced/
└── nnfit
```
where the `pre-trained-model` folder contains the just trained model. Hence in case you do not
want to train the GANs and directly resort to a pre-trained one, a pre-trained
[model](https://github.com/N3PDF/ganpdfs/tree/DynamicArchitecture/pre-trained-model)
can be used out of the box by setting the entry `use_saved_model` to `True` in the runcard. The 
`nnfit` subfolder contains the output grids from the generated replicas (this has the exact same
structure as the output from N3FIT). 

Hence, in order to evolve the generated output grids, just run:
```bash
evolven3fit <PRIOR_PDF_NAME>_enhanced
```

Then, to link the generated PDF set to the LHAPDF data directory, use the `postgans` script by
running:
```bash
postgans --pdf <PRIOR_PDF_NAME> --nenhanced TOT_FAKE_SIZE
```

#### Hyper-parameter opitmization

For more details on how to define specific parameters when running the code and on how to perform
a hyper-parameter scan, please head to the section [how to](https://n3pdf.github.io/ganpdfs/howto/howto.html)
of the documentation.
