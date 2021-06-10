# pycode2seq

Training and inference with multiple languages of PyTorch's implementation of code2seq model.

## Installation

```shell
python setup.py install
```

## Inference

Minimal code example:

```python
import sys
from pycode2seq import DefaultModelRunner

def main(argv):
    runner = DefaultModelRunner(
        save_path = "./tmp",
    )

    method_embeddings = runner.run_embeddings_on_file(argv[1], "kt") 

if __name__ == "__main__":
    main(sys.argv)
```

## Training

Download astminer and run:

```shell
./gradelw shadowJar
```

Mine projects for paths:

```shell
python training/mine_projects.py <data folder> <output folder> <path to astminer's cli.sh>
```

Combine mined paths:

```shell
python training/astminer_to_code2seq.py <data folder/holdout> <output folder> <holdout>
```

Build vocabulary with build_vocabulary.py from code2seq module

Combine vocabularies:

```shell
python training/combine_vocabularies.py
```

Expand weights:

```shell
python training/expand_weights.py
```
