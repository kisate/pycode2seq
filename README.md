# pycode2seq

Training and inference with multiple languages of PyTorch's implementation of code2seq model.

## Installation

```shell
pip install pycode2seq
```

## Inference

####File embeddings example

```python
from pycode2seq import Code2Seq

model = Code2Seq.load("kt_java")

# Dictionary of method names with their embeddings
method_embeddings = model.methods_embeddings("File.kt", "kt")
```

####Full functionality
```python
import sys
from pycode2seq import Code2Seq

def main(argv):
    model = Code2Seq.load("kt_java")

    # Dictionary of method names with their embeddings
    method_embeddings = model.methods_embeddings("File.kt", "kt") 

    #Code2seq predictions
    predictions = model.run_on_file(argv[1], "kt")

    #Predicted method names
    names = [model.prediction_to_text(prediction) for prediction in predictions]

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
