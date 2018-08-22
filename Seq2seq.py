from __future__ import print_function
import tensorflow as tf
import time
import numpy as np
class Seq2seq:
    def __init__(self, batchSize, trainDataset, testDataset, inferenceDataset, embeddingSize, cellSize, checkpointPath, numEpochs, cpuOnly = True):
        self._cpuOnly = cpuOnly
        self._cellSize = cellSize
        self._batchSize = batchSize
        self._embeddingSize = embeddingSize
        vocabInfo = trainDataset(vocabAsp = True)
        self._sourceStart = vocabInfo.sourceStart
        self._sourceEnd = vocabInfo.sourceEnd
        self._sourceUnknown = vocabInfo.sourceUnknown
        self._sourceVocabSize = vocabInfo.sourceVocabSize
        self._targetStart = vocabInfo.targetStart
        self._targetEnd = vocabInfo.targetEnd
        self._targetUnknown = vocabInfo.targetUnknown
        self._targetVocabSize = vocabInfo.targetVocabSize
        self._checkpointPath = checkpointPath
        self._numEpochs = numEpochs
        with tf.variable_scope("encoder", initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.1),
                               reuse = tf.AUTO_REUSE) as scope:
            self._encoderScope = scope
        with tf.variable_scope("decoder", initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.1),
                               reuse = tf.AUTO_REUSE) as scope:
            self._decoderScope = scope
        #first I tried the following: self._scope = tf.variable_scope(...). But, tensorflow gave me an error.
        #the main goal of generating a variable_scope here, is to ensure variable sharing of the model graph among
        #train, test and prediction pipelines. In other words, we want to be sure that the variables in the model class
        #have a specific name across all the three train, test and prediction pipelines.
        lossTrain, trainStep, saver = self._buildModelGraph(trainDataset, "train")
        lossEval, iteratorEval = self._buildModelGraph(testDataset, "evaluation")
        generatedSequences, generatedSequenceLengths, targetOutput, targetOutputLengths = self._buildModelGraph(inferenceDataset, "inference")
        self._run(lossTrain, trainStep, lossEval, iteratorEval, saver)
        self._computeBleu(generatedSequences, generatedSequenceLengths, targetOutput, targetOutputLengths)
        
    def _printOperations(self):
        print(tf.get_default_graph().get_operations())

    def _printVariables(self):
        print(tf.get_default_graph().get_collection("variables"))

    def _run(self, lossTrain, trainStep, lossEval, iteratorEval, saver):

        def eval(sess):
            lossEvals = []
            try:
                sess.run(iteratorEval.initializer)
                #the above will restart the evaluation iterator.
                while True:
                    lossEvals.append(sess.run(lossEval))
            except tf.errors.OutOfRangeError:
                pass
            return np.mean(lossEvals)

        init = tf.global_variables_initializer()
        step, lossTrainT = 0, []
        with tf.Session() as sess:
            sess.run(init)
            try:
                while True:
                    step += 1
                    lossTrainT.append(sess.run([lossTrain, trainStep])[0])
                    if step % 1000 == 0:
                        saver.save(sess, self._checkpointPath, global_step = step)
                        print("[step", step, "] [train loss", np.mean(lossTrainT), "] [eval loss", eval(sess), "]")
                        lossTrainT =[]
            except tf.errors.OutOfRangeError:
                #this is how tensorFlow detects the end of file
                pass
            saver.save(sess, self._checkpointPath, global_step = step)

    def _computeBleu(self, generatedSequences, generatedSequenceLengths, targetOutput, targetOutputLengths):
        init = tf.global_variables_initializer()
        translations = []
        with tf.Session() as sess:
            sess.run(init)
            try:
                while True:
                    translations.append(sess.run([targetOutput, targetOutputLengths, generatedSequences, generatedSequenceLengths]))
            except tf.errors.OutOfRangeError:
                #this is how tensorFlow detects the end of file
                pass
        from bleu import compute_bleu
        print(compute_bleu(translations))
        
    def _buildModelGraph(self, dataset, scopeName):
        with tf.variable_scope(scopeName) as scope:
            dataset = dataset()
            #I noticed that while invoking the tf.data.dataset api and invoking several map transformation back to back, three Const
            #operation nodes will be added to the tensor graph (filename, buffer_size, compression_type). No node will be added for the
            #corresponding mapping transformation. We prefer to label these nodes with their corresponding pipeline names.
            (((source, sourceLengths), (targetInput, targetInputLengths), (targetOutput, targetOutputLengths)), iterator) = self._buildDatasetIterator(dataset, scopeName)
            #the returned iterator by _buildDatasetIterator only will be used for evaluation via running iterator.initializer to restart the
            #iterator in order to pass over data several times.
            source = tf.transpose(source, [1, 0], name = "source")
            targetInput = tf.transpose(targetInput, [1, 0], name = "targetInput")
            targetOutput = tf.transpose(targetOutput, [1, 0], name = "targetOutput")
            encoderOutput, encoderState = self._buildEncoderGraph(source, sourceLengths)
            logits, sampleId, finalState, generatedSequenceLengths = self._buildDecoderGraph(targetInput, targetInputLengths, encoderState,
                                                                                             sourceLengths, scopeName)
            if scopeName == "train":
                return self._buildTrainGraph(targetOutput, targetOutputLengths, logits, tf.train.AdamOptimizer(0.001))
            elif scopeName == "evaluation":
                return (self._buildEvaluationGraph(targetOutput, targetOutputLengths, logits), iterator)
            elif scopeName == "inference":
                return (sampleId, generatedSequenceLengths, targetOutput, targetOutputLengths)

    def _buildModelGraphT(self, dataset, scopeName):
        with tf.variable_scope(scopeName) as scope:
            dataset = dataset()
            #I noticed that while invoking the tf.data.dataset api and invoking several map transformation back to back, three Const
            #operation nodes will be added to the tensor graph (filename, buffer_size, compression_type). No node will be added for the
            #corresponding mapping transformation. We prefer to label these nodes with their corresponding pipeline names.
            ((source, sourceLengths), (targetInput, targetInputLengths), (targetOutput, targetOutputLengths)) = self._buildDatasetIterator(dataset)
            test12 = source
            test13 = targetInput
            test14 = targetOutput
            source = tf.transpose(source, [1, 0], name = "source")
            targetInput = tf.transpose(targetInput, [1, 0], name = "targetInput")
            targetOutput = tf.transpose(targetOutput, [1, 0], name = "targetOutput")
            encoderOutput, encoderState = self._buildEncoderGraph(source, sourceLengths)
            logits, sampleId, finalState, generatedSequenceLengths = self._buildDecoderGraph(targetInput, targetInputLengths, encoderState,
                                                                                             sourceLengths, scopeName)
            if scopeName == "train":
                loss, trainStep = self._buildTrainGraph(targetOutput, targetOutputLengths, logits, tf.train.AdamOptimizer(0.001))
            elif scopeName == "evaluation":
                loss = self._buildEvaluationGraph(targetOutput, targetOutputLengths, logits)
            test1 = tf.shape(encoderOutput[:,1, :])
            test2 = encoderOutput[:,1, :]
            test3 = encoderState[1, :]
            test4 = tf.shape(logits)
            test5 = logits[:, 1, :]
            test6 = tf.shape(logits)
            test7 = tf.shape(targetOutput)
            test8 = tf.shape(targetInput)
            test9 = targetInput[:, 1]
            test10 = sampleId[:, 1]
            test11 = tf.shape(finalState)
            init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(1):
                print(sess.run([targetOutputLengths, trainStep]))
                print()
            try:
                while True:
                    print(sess.run([loss, trainStep]))
            except tf.errors.OutOfRangeError:
                #this is how tensorFlow detects the end of file
                pass

    def _buildDatasetIterator(self, dataset, mode):
        if mode == "train":
            dataset = dataset.repeat(self._numEpochs)
            #it makes more sense to repeat the training dataset before shuffling to enforce the randomness of training examples
            #during training.
        dataset = dataset.shuffle(self._batchSize * 1000)
        #the argument in the shuffle transformation is the buffer size that determines the size of buffer that is used for shuffling
        #the examples. It means that the implemented shuffling operation is an approximation. The choice of batchSize multiplied by
        #1000 is identical to the choice of buffer size in the official tutorial of seq2seq from TensorFlow.
        #dataset = dataset.map(lambda x, y: ((y["source"], tf.size(y["source"])), (y["target"], tf.size(y["target"]))))
        dataset = dataset.map(lambda x, y, z: ((x, tf.size(x)), (y, tf.size(y)), (z, tf.size(z))), num_parallel_calls = 8)
        dataset = dataset.padded_batch(self._batchSize, padded_shapes = ((tf.TensorShape([None]), tf.TensorShape([])),
                                                                         (tf.TensorShape([None]), tf.TensorShape([])),
                                                                         (tf.TensorShape([None]), tf.TensorShape([]))),
                                       padding_values = ((self._sourceEnd, 0), (self._targetEnd, 0), (self._targetEnd, 0)))
        dataset = dataset.prefetch(1)
        if mode == "train" or mode == "inference":
            it = dataset.make_one_shot_iterator()
            #the above iterator only passes once over data
        elif mode == "evaluation":
            it = dataset.make_initializable_iterator()
            #the above iterator has the potential to pass over data several times by running its initializer via sess.run(it.initializer)
        return (it.get_next(), it)

    def _buildEncoderGraph(self, source, sourceLengths):
        with tf.device("/cpu:0"):
            with tf.variable_scope(self._encoderScope) as scope:
                embeddingEncoder = tf.get_variable("embeddingEncoder", shape = [self._sourceVocabSize, self._embeddingSize])
            sourceEmb = tf.nn.embedding_lookup(embeddingEncoder, source, name = "sourceEmb")
        device = "/cpu:0" if self._cpuOnly else "/gpu:0"
        encoderCell = tf.contrib.rnn.DeviceWrapper(tf.contrib.rnn.GRUCell(num_units = self._cellSize, reuse = tf.AUTO_REUSE), device)
        #using print(tf.trainale_variables()), I noticed that creating a cell by itself doesn't create any tf.Variable. In other words,
        #it appears the above command doesn't invoke any tf.get_variable(). Therefore, you don't need to be worried about variabel_scope
        #while defining the cell itself. When you execute print(encoderCell), it tellls you that encoder cell is a cell class and not a
        #Tensor class, which again shows that just defining an rnn cell in addition to not registering any variable even doesn't add any
        #tensor node to the tensor graph as confirmed by print(tf.get_default_graph().get_operations()).
        with tf.variable_scope(self._encoderScope) as scope:
            encoderOutput, encoderState = tf.nn.dynamic_rnn(encoderCell, sourceEmb, sequence_length = sourceLengths,
                                                            initial_state = encoderCell.zero_state(tf.size(sourceLengths), dtype = tf.float32)
                                                            ,dtype = tf.float32, parallel_iterations = 8,
                                                            swap_memory = True, time_major = True)
            #In above to create initial_state Tensor via the zero_state method of the encoderCell, you will need to pass the batchSize.
            #Since, the batchsize of the last minibatch almost always will be less than the chosen self._batchSize, we use
            #tf.size(sourceLenghts) to refer to the size of current minibatch.

        return (encoderOutput, encoderState)

    def _buildDecoderGraph(self, target, targetLengths, encoderState, sourceLengths, mode):
        with tf.device("/cpu:0"):
            with tf.variable_scope(self._decoderScope) as scope:
                embeddingDecoder = tf.get_variable("embeddingDecoder", shape = [self._targetVocabSize, self._embeddingSize])
            targetEmb = tf.nn.embedding_lookup(embeddingDecoder, target, name = "targetEmb")
        device = "/cpu:0" if self._cpuOnly else "/gpu:1"
        decoderCell = tf.contrib.rnn.DeviceWrapper(tf.contrib.rnn.GRUCell(num_units = self._cellSize, reuse = tf.AUTO_REUSE), device)
        #the decoderCell is just a blueprint of the decoder cell and doesn't create any tf.variable which was confirmed by ._printVariables()
        if mode == "train" or mode == "evaluation":
            helper = tf.contrib.seq2seq.TrainingHelper(targetEmb, targetLengths, time_major = True)
            #the helper function above also, doesn't create any tf.variable which is confirmed by self._printVariables()
            #the TrainingHelper fn takes targetLengths (the length of the target sequences) and causes the dynamic_decode module only
            #process each inpput sequence to the decoder provided by TrainingHelper upto the given sequence lengths, therefore the lengths
            #of generated sequences by dynamic_decode (referred by finalSequenceLengths) will be exactly equal to the sequence lengths
            #provided by targetLengths.
        elif mode == "inference":
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddingDecoder, tf.fill([tf.size(sourceLengths)],
                                                                                        tf.cast(self._targetStart, dtype = tf.int32)),
                                                              self._targetEnd)
            #the second argument of GreedyEmbeddingHelper must be a vector of TensorShape([batchSize]) filled by the start token on the
            #target side. Since, the batchsize of the last minibatch almost always will be less than the chosen self._batchSize, we use
            #tf.size(sourceLenghts) to refer to the size of current minibatch.

        decoder = tf.contrib.seq2seq.BasicDecoder(decoderCell, helper, encoderState,
                                                  output_layer = tf.layers.Dense(self._targetVocabSize, use_bias = False))
        #the basic decoder above doesn't create any tf.Variable which is confirmed by self._printVariables()
        #also tf.layers.Dense doesn't create any tf.Variable.
        if mode == "train" or mode == "evaluation":
            maximum_iterations = None
        elif mode == "inference":
            maximum_iterations = tf.round(tf.reduce_max(sourceLengths) * 2)
            #for training and evaluation, maximum_iterations is not needed sice the lengths of target sequences is given.
            #However, for inference, the target sequence is unknown and we want to avoid decoding process to runs forever,
            #therefore, we specify the maximum allowed length of generate sequences by maximum_iterations which is chosen
            #to be two times of the longest source sequence in the current minibatch.
        with tf.variable_scope(self._decoderScope) as scope:
            finalOutputs, finalState, finalSequenceLengths = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major = True,
                                                                                               swap_memory = True,
                                                                                               maximum_iterations = maximum_iterations)
            #dynamic_decode creates new variables and should be placed inside a variable_scope
            #dynamic_decode takes an optional maximum_iterations argument that halt the decoding process as soon as the length of a
            #generated sequence for an input sequence reaches the limit provided by maximum_iterations. However, because of
            #TrainingHelper, we don't need to be worries about maximum_iterations since it forces the stop of decoding process for each
            #target sequence as soon as the number of decoding steps reaches the sequence length.
        #finalOutputs is an instance of BasicDecoderOutput class with the following definition:
        #class BasicDecoderOutput(collections.namedtuple("BasicDecoderOutput", ("rnn_output", "sample_id"))):
        #    pass => the pass is  becasue you cannot have completely empty class definition in Python
        logits = finalOutputs.rnn_output
        #rnn_ouput is a tensor with the following shape [maxSequenceLength, batchSize, targetVocabSize]. For the sequences shorter than
        #maxSequenceLength when I printed the logits, it seems that dynamic_decode continue updating their corresponding slice of tensor
        #which can be accessed by logits[:, i, :] where i is the index of the sequence in the batch. However, I don't think this behavior
        #of dynamic_decode will hurt our seq2seq applications if we define the final loss fn in a smart way to mask out those generated
        #output be dynamic_decode with the indices larger than their corresponding sequenceLengths.
        sampleId = finalOutputs.sample_id
        #sampleId is a tensor of shape [maxSequenceLength, batchSize] where contains the indices of sampled tokens at the output of decoder.
        #In the case of TrainingHelper, we might expect the sampleId be same as target tensor that refers to the indice of target sequences.
        #That is not the case and they are different. Note that the TrainingHelper takes embeddings of target sequences as input and not
        #their indices so that the dynamic_decode even doesn't have acccess to the ground truth target sequences. Also, I think it doesn't
        #matter even if sampleID is different from the ground truth target sequence since the input to the decoder during decoding comes
        #directly from TrainingHelper and the sampleId generated by the decoder based on its output.

        #finalState as it is clear from its name, is the final state of the decoder and is a tensor with shape of [batchSize, cellSize].
        #Also, you don't need to be worried about shorter sequences and finalState corresponding to shorter sequences is not all zero
        #but they containg the state of the decoder upto lengths of sequences.
        return (logits, sampleId, finalState, finalSequenceLengths)

    def _buildTrainGraph(self, targetOutput, targetOutputLengths, logits, optimizer):
        """The shape of tragetOutput is [maxTime, batchSize] whereas the shape of logits is [maxTime, batchSize, vocabSize].
        The sparse softmax cross entropy loss takes into account the difference between the dimensions of targetOutput and 
        logits, and treats targetOutput as a sparse tensor. In other words, underhood, this method will convert the targetOutput 
        tensor to its corresponding one-hot represenation.
        optimizer must be a subclass of the Optimizer class."""
        crossnet = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = targetOutput, logits = logits)
        #For the sequences in the batch that are shorter than maxTime, the logits tensor will still have nonzero values after
        #their length corresponding to the padding tokens. In order to cancel out the loss terms corresponding to padding tokens,
        #we need to multiply crossnet by a max tensor. In particular, the shape of crossnet is [maxTime, batchSize]. In other words,
        #while computing the cross entropy loss, for each token in an input sequence, we will have a loss term. Knowing that, it will
        #be very straightforward to cancel out the loss terms corresponding to padding tokens.
        mask = tf.transpose(tf.sequence_mask(targetOutputLengths, dtype = logits.dtype))
        #tf.sequence_max will generate a masking tensor based on the length of tensors in order to cancel out the loss terms
        #corresponding to padding tokens.
        loss = tf.reduce_sum(crossnet * mask) / tf.to_float(tf.size(targetOutputLengths))
        #loss is normalized by dividing the loss by the batchSize in order to make the training process invariant to batchSize.
        #We don't normalize the loss by batchSize * maxTime where maxTime is the length of the longest sequence in the batch because
        #we don't want to reduce the impact of shorter sequences in a given minibatch compared to the longer sequences in that minibatch
        #Since, the batchsize of the last minibatch almost always will be less than the chosen self._batchSize, we use
        #tf.size(sourceLenghts) to refer to the size of current minibatch.
        params = tf.trainable_variables()
        #params will be simply a python list of all tf.Variable instances with their trainable property being True
        gradients = tf.gradients(loss, params)
        #gradients will a python list of tf.Tensor instances where the len of Python list will be same as the len of params python
        #list. Each tensor represent derivative tensor of loss with respect to each tf.Variable in params list.
        gradientsClipped, globalNorm = tf.clip_by_global_norm(gradients, 1.0)
        #here gradients is a list of tensors and the other input for this fn is scalar that is clip_norm.
        # where inside the function a single global_norm across all the tensors will be computed as follows:
        #sqrt(sum_{e being an element if one of the tensors}{e ** 2}). Then, the gradientsClipped will contain the transformed
        #version of the gradients tensors following this formula: e * clip_norm / max(clip_norm, global_norm).
        #As you can see in above, if clip_norm is bigger than global_norm, then this clipping transformation won't change the
        #values of the gradints tensors. But, if global_norm (the computed norm of tensors) is bigger than clip_norm, then
        #this clipping transformation will change the values of tensors in such a way that global_norm of generated tensors
        #will be equal to clip_norm
        trainStep = optimizer.apply_gradients(zip(gradientsClipped, params))
        saver = tf.train.Saver(sharded = True)
        #The saver object to store the variables. Since the var_list argument is None, all the varibales will be saved.
        #Since sharded is True, it will shard the checkpoints per device.
        return (loss, trainStep, saver)

    def _buildEvaluationGraph(self, targetOutput, targetOutputLengths, logits):
        """The shape of tragetOutput is [maxTime, batchSize] whereas the shape of logits is [maxTime, batchSize, vocabSize].
        The sparse softmax cross entropy loss takes into account the difference between the dimensions of targetOutput and 
        logits, and treats targetOutput as a sparse tensor. In other words, underhood, this method will convert the targetOutput 
        tensor to its corresponding one-hot represenation."""
        
        crossnet = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = targetOutput, logits = logits)
        #For the sequences in the batch that are shorter than maxTime, the logits tensor will still have nonzero values after
        #their length corresponding to the padding tokens. In order to cancel out the loss terms corresponding to padding tokens,
        #we need to multiply crossnet by a max tensor. In particular, the shape of crossnet is [maxTime, batchSize]. In other words,
        #while computing the cross entropy loss, for each token in an input sequence, we will have a loss term. Knowing that, it will
        #be very straightforward to cancel out the loss terms corresponding to padding tokens.
        mask = tf.transpose(tf.sequence_mask(targetOutputLengths, dtype = logits.dtype))
        #tf.sequence_max will generate a masking tensor based on the length of tensors in order to cancel out the loss terms
        #corresponding to padding tokens.
        loss = tf.reduce_sum(crossnet * mask) / tf.to_float(tf.size(targetOutputLengths))
        #loss is normalized by dividing the loss by the batchSize in order to make the training process invariant to batchSize.
        #We don't normalize the loss by batchSize * maxTime where maxTime is the length of the longest sequence in the batch because
        #we don't want to reduce the impact of shorter sequences in a given minibatch compared to the longer sequences in that minibatch
        #Since, the batchsize of the last minibatch almost always will be less than the chosen self._batchSize, we use
        #tf.size(sourceLenghts) to refer to the size of current minibatch.
        return loss

    @staticmethod
    def iwslt15():
        from TFRecordReader import TFRecordReader
        trainDataset = TFRecordReader.iwslt15Train
        testDataset = TFRecordReader.iwslt15Test
        inferenceDataset = TFRecordReader.iwslt15Test
        seq2seq = Seq2seq(batchSize = 128, trainDataset = trainDataset, testDataset = testDataset,
                          inferenceDataset = inferenceDataset, embeddingSize = 512, cellSize = 512,
                          checkpointPath = "model/iwslt15/model", numEpochs = 11)
        #In Google NMT tutorial, they use the same size as cellSize for embedding size.

def main():
    Seq2seq.iwslt15()

if __name__ == "__main__":
    main()
