from __future__ import print_function

import tensorflow as tf

class TFRecordReader:
    def _scalarInt64(self):
        """this returns the signature for the tf.train.Feature of type int64 saved in a tfrecord file."""
        return tf.FixedLenFeature([], tf.int64)

    def _scalarBytes(self):
        """this returns the signature for the tf.train.Feature of type bytes saved in a tfrecord file."""
        return tf.FixedLenFeature([], tf.string)

    def _scalarFloat(self):
        """this returns the signature for the tf.train.Feature of type float saved in a tfrecord file."""
        #In python, all the floating point numbers are 64bits. That is the reason that I've chosen tf.float64
        return tf.FixedLenFeature([], tf.float64)

    def _sequenceInt64(self):
        """this returns the signature for the tf.train.FeatureList of type int64 saved in a tfrecord file."""
        return tf.FixedLenSequenceFeature([], tf.int64)

    def _sequenceBytes(self):
        """this returns the signature for the tf.train.FeatureList of type saved in a tfrecord file."""
        return tf.FixedLenSequenceFeature([], tf.string)

    def _sequenceFloat(self):
        """this returns the signature for the tf.train.FeatureList of type float saved in a tfrecord file."""
        #In python, all the floating point numbers are 64bits. That is the reason that I've chosen tf.float64
        return tf.FixedLenSequenceFeature([], tf.float64)

    def scalar2scalar(self, filename, xLabel = "input", yLabel = "output", xSignature = None, ySignature = None, num_parallel_calls = 64):
        """This method takes a TFRecord file where the serialized objects are tf.train.Example and returns its corresponding tf.Dataset.
        xSignature and ySignature represent the signature of the tf.train.Feature corresponding to xLabel and yLabel.
        num_parallel_calls determine the number of elemnts in the dataset during map transformation that will be processed in parallel
        using TensorFlow thread pools."""
        if not xSignature:
            xSignature = self._scalarInt64
        if not ySignature:
            ySignature = self._scalarInt64
        dataset = tf.data.TFRecordDataset(filename)
        return dataset.map(lambda x: tf.parse_single_example(x, features = {xLabel: xSignature(), yLabel: ySignature()}), num_parallel_calls = num_parallel_calls)

    def seq2seq(self, filename, xLabel = "input", yLabel = "output", xSignature = None, ySignature = None, num_parallel_calls = 64):
        """This method takes a TFRecord file where the serialized objects are tf.train.SequenceExample and returns its corresponding 
        tf.Dataset. xSignature and ySignature represent the signature of the serialized tf.train.FeatureList corresponding to xLable and 
        yLabel.
        num_parallel_calls determine the number of elemnts in the dataset during map transformation that will be processed in parallel
        using TensorFlow thread pools"""
        if not xSignature:
            xSignature = self._sequenceInt64
        if not ySignature:
            ySignature = self._sequenceInt64
        dataset = tf.data.TFRecordDataset(filename)
        return dataset.map(lambda x: tf.parse_single_sequence_example(x, sequence_features = {xLabel: xSignature(), yLabel: ySignature()}),
                           num_parallel_calls = num_parallel_calls)

    def seq2seqContext(self, filename, (cXLabel, cYLabel, xLabel, yLabel) = ("vocabSizeSource", "vocabSizeTarget", "source", "target"),
                       cXSignature = None, cYSignature = None, xSignature = None, ySignature = None, num_parallel_calls = 64):
        """This method takes a TFRecord file where the serialized objects are tf.train.SequenceExample and returns its corresponding 
        tf.Dataset. xSignature and ySignature represent the signature of the serialized tf.train.FeatureList corresponding to xLable and 
        yLabel. cXSignature and cYSignature represent the signature of the serialized tf.train.Feature corresponding to cXLable and 
        cYLabel.
        num_parallel_calls determine the number of elemnts in the dataset during map transformation that will be processed in parallel
        using TensorFlow thread pools."""
        if not cXSignature:
            cXSignature = self._scalarInt64
        if not cYSignature:
            cYSignature = self._scalarInt64
        if not xSignature:
            xSignature = self._sequenceInt64
        if not ySignature:
            ySignature = self._sequenceInt64
        dataset = tf.data.TFRecordDataset(filename)
        return dataset.map(lambda x: tf.parse_single_sequence_example(x, context_features = {cXLabel: cXSignature(), cYLabel: cYSignature()},
                                                                      sequence_features = {xLabel: xSignature(), yLabel: ySignature()}),
                           num_parallel_calls = num_parallel_calls)

    def machineTranslation(self, filename, vocabAsp, num_parallel_calls = 8):
        """This method takes a TFRecord file where the serialized objects are tf.train.SequenceExample and returns its corresponding 
        tf.Dataset. The input tfrecord file represents sequence examples with a predefined format.
        If vocabAsp is True then, only the first sequence example of the tfrecord file will be read and the aspects of vocab will be
        read from that first sequence example, and it will be the only returned variable.
        If vocabAsp is False, then the first sequence example will be skipped and the dataset corresponding to the rest of sequence 
        examples will be returned.
        num_parallel_calls determine the number of elemnts in the dataset during map transformation that will be processed in parallel
        using TensorFlow thread pools."""

        def getVocabContext():
            import collections
            vocab = collections.namedtuple("vocabAsc", ["sourceStart", "sourceEnd", "sourceUnknown", "sourceVocabSize", 
                                                        "targetStart", "targetEnd", "targetUnknown", "targetVocabSize"])
            graph = tf.Graph()
            with graph.as_default():
                dataset = tf.data.TFRecordDataset(filename).take(1)
                dataset = dataset.map(lambda x: tf.parse_single_sequence_example(
                    x, context_features = {"sourceStart": self._scalarInt64(), "sourceEnd": self._scalarInt64(),
                                           "sourceUnknown": self._scalarInt64(), "sourceVocabSize": self._scalarInt64(), 
                                           "targetStart": self._scalarInt64(), "targetEnd": self._scalarInt64(),
                                           "targetUnknown": self._scalarInt64(), "targetVocabSize": self._scalarInt64()}))
                dataset = dataset.batch(1)
                it = dataset.make_one_shot_iterator()
                next = it.get_next()
            with tf.Session(graph = graph) as sess:
                context, _ = sess.run(next)
            return vocab(context["sourceStart"][0], context["sourceEnd"][0], context["sourceUnknown"][0], context["sourceVocabSize"][0],
                      context["targetStart"][0], context["targetEnd"][0], context["targetUnknown"][0], context["targetVocabSize"][0])
                #the index 0 in above is because the teturned value by the context keys is a list of a single integer like [7789] and we
                #want to only store the integet and not the list.
        if vocabAsp:
            return getVocabContext()
        vocab = getVocabContext()
        dataset = tf.data.TFRecordDataset(filename).skip(1)
        dataset = dataset.map(lambda x: tf.parse_single_sequence_example(
            x, sequence_features = {"source": self._sequenceInt64(), "target": self._sequenceInt64()}), num_parallel_calls = num_parallel_calls)
        dataset = dataset.map(lambda x, y: (y["source"], tf.concat(([vocab.targetStart], y["target"]),0),
                                            tf.concat((y["target"], [vocab.targetEnd]),0)), num_parallel_calls = num_parallel_calls)
        #dataset = dataset.map(lambda x, y: (y["source"], y["target"]))
        return dataset

    def scalar2seq(self, filename, xLabel = "input", yLabel = "output", xSignature = None, ySignature = None, num_parallel_calls = 64):
        """This method takes a TFRecord file where the serialized objects are tf.train.SequenceExample and returns its corresponding 
        tf.Dataset. xSignature is the signature of the tf.train.Featue corresponding to xLabel and ySignature represent the signature of 
        the serialized tf.train.FeatureList corresponding to yLabel.
        num_parallel_calls determine the number of elemnts in the dataset during map transformation that will be processed in parallel
        using TensorFlow thread pools."""
        if not xSignature:
            xSignature = self._scalarBytes
        if not ySignature:
            ySignature = self._sequenceInt64
        dataset = tf.data.TFRecordDataset(filename)
        return dataset.map(lambda x: tf.parse_single_sequence_example(x, context_features = {xLabel: xSignature()},
                                                                      sequence_features = {yLabel: ySignature()}), num_parallel_calls = num_parallel_calls)

    @staticmethod
    def iwslt15Train(vocabAsp = False):
        return TFRecordReader().machineTranslation("iwslt15Train.tfrecord", vocabAsp = vocabAsp)

    @staticmethod
    def iwslt15Test(vocabAsp = False):
        return TFRecordReader().machineTranslation("iwslt15Test.tfrecord", vocabAsp = vocabAsp)

def test2():
    dataset = TFRecordReader.iwslt15()
    dataset.batch(1)
    it = dataset.make_one_shot_iterator()
    next = it.get_next()
    with tf.Session() as sess:
        print(sess.run(next))
        print(sess.run(next))

def test6():
    def getContext():
        iniGraph = tf.Graph()
        with iniGraph.as_default():
            iniDataset = TFRecordReader.iwslt15()
            iniDataset.batch(1)
            iniIt = iniDataset.make_one_shot_iterator()
            iniNext = iniIt.get_next()
        sess1 = tf.Session(graph = iniGraph)
        context, _ = sess1.run(iniNext)
        return context
    context = getContext()
    sourceEnd = context["sourceEnd"]
    targetEnd = context["targetEnd"]
    dataset = TFRecordReader.iwslt15()
    dataset = dataset.map(lambda x, y: ((y["source"], tf.size(y["source"])), (y["target"], tf.size(y["target"]))))
    dataset = dataset.padded_batch(4, padded_shapes = ((tf.TensorShape([None]), tf.TensorShape([])), (tf.TensorShape([None]), tf.TensorShape([]))), padding_values = ((context["sourceEnd"], 0), (context["targetEnd"], 0)))
    it = dataset.make_one_shot_iterator()
    ((source, source_lengths), (target, target_lengths)) = it.get_next()
    with tf.Session() as sess:
        for i in range(1):
            print(sess.run(source))
            print()

if __name__ == "__main__":
    TFRecordReader.iwslt15()
                                
                                
