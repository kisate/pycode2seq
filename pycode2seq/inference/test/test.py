from pycode2seq import Code2Seq


def main():
    model = Code2Seq.load("kt_java")
    embs = model.methods_embeddings("FairSeqGeneration.kt", "kt")

    assert len(embs) == 9


if __name__ == '__main__':
    main()
