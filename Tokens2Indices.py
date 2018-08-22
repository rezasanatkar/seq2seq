from __future__ import print_function

import pickle
class Tokens2Indices:
    """The main functionality of this class is to use VocabBuilder module to map a given pair of source and target sequences to two 
    lists of integers and pass them to another module that convert them into TFRecords."""

    def __init__(self, sourceVocab, targetVocab, source, target):
        self.sourceVocab = sourceVocab
        self.targetVocab = targetVocab
        self.source = source
        self.target = target

    def _seq2Indices(self, sequence, vocab, start, end, unknown):
        """this method maps a given sequnce(list of tokens) to a list of integers that are the vocab indices.
        if start is not None, the start will be added to the begining of the sequence and if end is not None,
        the end will be added to the end of sequence."""
        if start:
            sequence.insert(0, start)
        if end:
            sequence.append(end)
        return [vocab[token] if token in vocab else vocab[unknown] for token in sequence]

    def map(self, start = "<START>", end = "<END>", unknown = "<UNKNOWN>"):
        """this method generates a list of tuples where each tuple first element is the list of vocab indices of the source sequence and
        the second element is the list of vocab indices of the target sequence."""
        return [(self._seq2Indices(s, self.sourceVocab, start = start, end = end, unknown = unknown),
                 self._seq2Indices(t, self.targetVocab, start = start, end = end, unknown = unknown)
        ) for s, t in zip(self.source, self.target)]

    @staticmethod
    def iwslt15Train(start = "<START>", end = "<END>", unknown = "<UNKNOWN>"):
        from VocabBuilder import VocabBuilder
        sourceVocab, targetVocab = VocabBuilder.iwslt15(start = start, end = end, unknown = unknown)
        source = []
        import io
        with io.open("datasets/iwslt15/train.en", mode = "r", encoding = "utf-8") as f:
            lines = f.readlines()
            source = [line.split() for line in lines]
        target = []
        with io.open("datasets/iwslt15/train.vi", mode = "r", encoding = "utf-8") as f:
            lines = f.readlines()
            target = [line.split() for line in lines]
        return (Tokens2Indices(sourceVocab = sourceVocab, targetVocab = targetVocab, source = source, target = target).map(
            start = None, end = None, unknown = "<UNKNOWN>"), (sourceVocab[start], sourceVocab[end], sourceVocab[unknown], len(sourceVocab)
            ), (targetVocab[start], targetVocab[end], targetVocab[unknown], len(targetVocab))) 
    
    @staticmethod
    def iwslt15Test(start = "<START>", end = "<END>", unknown = "<UNKNOWN>"):
        from VocabBuilder import VocabBuilder
        sourceVocab, targetVocab = VocabBuilder.iwslt15(start = start, end = end, unknown = unknown)
        source = []
        import io
        with io.open("datasets/iwslt15/tst2012.en", mode = "r", encoding = "utf-8") as f:
            lines = f.readlines()
            source = [line.split() for line in lines]
        target = []
        with io.open("datasets/iwslt15/tst2012.vi", mode = "r", encoding = "utf-8") as f:
            lines = f.readlines()
            target = [line.split() for line in lines]
        return (Tokens2Indices(sourceVocab = sourceVocab, targetVocab = targetVocab, source = source, target = target).map(
            start = None, end = None, unknown = "<UNKNOWN>"), (sourceVocab[start], sourceVocab[end], sourceVocab[unknown], len(sourceVocab)
            ), (targetVocab[start], targetVocab[end], targetVocab[unknown], len(targetVocab))) 

def main():
    t = Tokens2Indices.generateIwslt15Data()
    print(t[0][0])
    print(t[1])
    print(t[2])
if __name__ == "__main__":
    main()

