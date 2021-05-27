# pycode2seq

Training and inference with multiple languages of PyTorch's implementation of code2seq model.

## Inference

Download antlr and create python parsers with grammars in src/inference/antlr.

Download checkpoint and config.

Minimal code example:

```python
import sys
from pycode2seq import ModelRunner
from pycode2seq.inference.paths.extracting import ExtractingParams

def main(argv):
    runner = ModelRunner(
        config_path=argv[1],
        vocabulary_path=argv[2],
        checkpoint_path=argv[3],
        ExtractingParams(
            max_length=8,
            max_width=3,
            paths_per_method=200
        )
    )

    prediction = runner.run_model_on_file(argv[4])
    print(runner.prediction_to_text(prediction))

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
python training/mine_projects.py data <data folder> output <output folder> cli_path <path to astminer's cli.sh>
```

Combine mined paths:

```shell
python training/combine_path.py data <data folder>
```

Combine vocabularies:

```shell
python training/combine_vocabularies.py
```

Expand weights:

```shell
python training/expand_weights.py
```
