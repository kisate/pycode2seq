# pycode2seq

Pure Python library for `code2seq` embeddings. 

Support extension of existing pretrained code2seq embeddings to multilingual models. 
We provided an example of the Java model extension with Kotlin.
Pretrained model and its usage example provided below.  

## Installation

```shell
pip install pycode2seq
```

## Inference

#### File embeddings example

```python
from pycode2seq import Code2Seq

model = Code2Seq.load("kt_java")
method_embeddings = model.methods_embeddings("File.kt")
```

Pretrained Java and Kotlin common model will be downloaded automatically.

#### Full functionality
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

### Available models

- Java (`java`)
- Kotlin (`kt` or `kotlin`)
- Java & Kotlin (`kt_java`)

`kt_java` is compatible with `java` model and should have the same embeddings.
`kotlin` model is a part of `kt_java` model, so they are compatible too.

So you can use the common `kt_java` model and get **embeddings in one vector space for both languages**.

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

## Using speedy-antlr-tool

You can use [speedy-antlr](https://speedy-antlr-tool.readthedocs.io/en/latest/example.html) to speed up file parsing speed.

Clone and install [modified example](https://github.com/amykyta3/speedy-antlr-example).

Replace parser call with:

```python
stream = antlr4.FileStream(input_file)
tree = sa_kotlin.parse(stream, "kotlinFile", sa_kotlin.SA_ErrorListener())
```

You still need lexer to recover token values, though.

Note, that to make Java parser you will need to follow speedy-antlr tutorial and make another package.

## Using astminer to parse files

Clone [astminer fork with kotlin support](https://github.com/kisate/astminer) and run

```shell
./gradlew shadowJar
```

Extract methods with ```cli.sh``` arguments and usage can be found in ```training/mine_projects.py```.

Pass path to folder with csvs to ```run_model_on_astminer_csv()```.
