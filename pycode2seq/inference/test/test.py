from pycode2seq.inference.model.model import Model


def main():
    model = Model.load("kt_java")
    embs = model.methods_embeddings("FairSeqGeneration.kt", "kt")

    assert len(embs) == 9


if __name__ == '__main__':
    main()
