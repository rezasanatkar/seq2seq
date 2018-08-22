""" his module is used to create the vocab file for a training corpus. """

from __future__ import absolute_import #it enables the absolute import form Python3
from __future__ import division #It makes 5 / 2 to be equal to 2.5
from __future__ import print_function #the famous print from Python3

import itertools
import collections

class VocabBuilder:
    """This class is used to build a vocab corresponding to a given textual training dataset.
    In the constructor, you need to provide the filename that will be used to extract tokens.
    Then, you need to invoke its build method to create the vocabulary."""
    def __init__(self, filename = None, fileFormat = "utf-8", seqs = None, verbose = True):
        """The file format either could be pickle or utf-8. 
        If the file format is pickle, then the serialized object must be either a list of strings like
        ["apple iphone 6", "red nail polish", ...] or a list of lists of strings like the following
        [["apple", "iphone 6"], ["red", "nail polish"], ...].
        If the object is the list of strings, the vocabulary will be built at token level by white-space
        splitting each string in the list. If the object is the list of lists of strings, then the vocabulary 
        will be built at segment level. For example in the above example, "apple", "iphone 6", "red", 
        "nail polish" each will be treated as a token.
        seqs could be another way to pass tokens to this constructor and coulb be in either two formats that
        is already described for the pickle serialized object."""
        tokenFreq = collections.Counter()
        if filename and fileFormat == "utf-8":
            import io, string
            with io.open(filename, mode = "r", encoding = "utf-8") as f:
                lines = f.readlines()
                tokenFreq = collections.Counter(itertools.chain(*map(string.split, lines)))
        elif filename and fileFormat == "pickle":
            import pickle
            with open(filename, "r") as f:
                data = pickle.load(f)
                if type(data[0]) == list:
                    tokenFreq = collections.Counter(itertools.chain(*data))
                else:
                    import string
                    tokenFreq = collections.Counter(itertools.chain(*map(string.split, data)))
        elif filename:
            print("fileformat must be either utf-8 or pickle")
            exit(1)
        if seqs:
            if type(seqs[0]) == list:
                tokenFreq += collections.Counter(itertools.chain(*seqs))
            else:
                import string
                tokenFreq += collections.Counter(itertools.chain(*map(string.split, seqs)))

        self.tokenFreq = sorted(tokenFreq.items(), key = lambda x: x[1], reverse = True)
        if verbose:
            print("the size of vacab will be: ", len(self.tokenFreq)) 
            print("the 10 most common tokes are:")
            print(self.tokenFreq[:10])
            
    def build(self, vocabFilename = None, vocabularySize = None, freqThreshold = 0, start = None, end = None, unknown = None):
        """This method creates vocab indices for tokens in the order that a token with more 
        repeatition in the input file will have a smaller vocab index. 
        Finally, the generated vocab indices will be returned as a dictionary where keys are the tokens and 
        the values are their corresponsing vocab indices. 
        Also, if a vocabFilename is given, the generated vocab indices will be saved as a utf-8 text file where
        each line is a token followed by a tab, followed by its vocab index.
        If start(end) argument is not None, then a special token start(end) will be added to the vocab as well as
        their corresponding indices. The use of these two special tokens will be in sequence processing applications 
        like seq2seq model that each sequence might be prepended by the start token and be appended by the end token.
        The special token <UNKNOWN> will be added to the end of the vocab list to be used for those tokens in the 
        test file that are not part of the built vocab.
        If unkonwn is not None, it  will be added to the end of the vocab list to be used for those tokens in the 
        test file that are not part of the built vocab.
        """
        import copy
        tokenFreqP = self.tokenFreq[:vocabularySize - 1] if vocabularySize else copy.copy(self.tokenFreq)
        tokens = [x[0] for x in tokenFreqP if x[1] >= freqThreshold]
        if start:
            tokens.append(start)
        if end:
            tokens.append(end)
        if unknown:
            tokens.append(unknown)
        tokenId = zip(tokens, range(len(tokens)))
        if vocabFilename:
            import io
            with io.open(vocabFilename, mode = "w", encoding = "utf-8") as f:
                f.write("".join(["%s\t%d\n" % (item[0], item[1]) for item in tokenId]))
        return dict(tokenId)

    @staticmethod
    def generateVocabFromTxtFile(filename, vocabularySize = None, freqThreshold = 0, start = None, end = None, unknown = None):
        """This static method takes a utf-8 file and computes its corresponding token. Then, it stores 
        the generated vocab in filename.vocab.
        The special token <unknown> will be added to the end of the vocab list to be used for those tokens in the 
        test file that are not part of the built vocab.
        If start(end) argument is not None, then a special token start(end) will be added to the vocab as well as
        their corresponding indices."""
        vb = VocabBuilder(filename = filename)
        return vb.build(vocabFilename = filename + ".vocab", vocabularySize = vocabularySize, freqThreshold = freqThreshold, start = start,
                        end = end, unknown = unknown)

    @staticmethod
    def iwslt15(start, end, unknown):
        vocabEnglish = VocabBuilder.generateVocabFromTxtFile(filename = "datasets/iwslt15/train.en", start = start,
                                                             end = end, unknown = unknown, freqThreshold = 5)
        vocabViet = VocabBuilder.generateVocabFromTxtFile(filename = "datasets/iwslt15/train.vi", start = start,
                                                          end = end, unknown = unknown, freqThreshold = 5)
        return (vocabEnglish, vocabViet)
        
def main():
    VocabBuilder.generateIwslt15Vocabs()
    
if __name__ == "__main__":
    main()
