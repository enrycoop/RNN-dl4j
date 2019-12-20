package com.enricocoppolecchia.NN;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class CreditCardDeep {
	public static void main(String[] args) throws IOException, InterruptedException {
		String path = args[0];
		int seed = 123;
		double learningRate = 0.5;
		int batchSize = 25;
		int nEpochs = 1500;
		int numInputs = 9;
		int numOutputs = 3;
		int numHiddenNodes = 4;

		RecordReader rr = new CSVRecordReader();
		rr.initialize(new FileSplit(new File(path + "/train.csv")));
		DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 2, 3);

		RecordReader rrTest = new CSVRecordReader();
		rrTest.initialize(new FileSplit(new File(path + "/balanced.csv")));
		DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 2, 3);

		// Normalize the training data

		DataNormalization normalizer = new NormalizerStandardize();
		normalizer.fit(trainIter); // Collect training data statistics
		testIter.reset();
		testIter.setPreProcessor(normalizer);

		// Use previously collected statistics to normalize on-the-fly. Each DataSet
		// returned by 'trainData' iterator will be normalized

		trainIter.setPreProcessor(normalizer);

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).weightInit(WeightInit.XAVIER)
				.updater(new Nesterovs(learningRate, 0.7)).list()
				.layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes).activation(Activation.HARDSIGMOID)
						.build())
				.layer(new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes).activation(Activation.RELU)
						.build())
				.layer(new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes).activation(Activation.RELU6)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
						.nIn(numHiddenNodes).nOut(numOutputs).build())
				.build();

		EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
				.epochTerminationConditions(new MaxEpochsTerminationCondition(nEpochs))
				.scoreCalculator(new DataSetLossCalculator(testIter, true)).evaluateEveryNEpochs(1)
				.modelSaver(new LocalFileModelSaver(path)).build();

		EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, conf, trainIter);

//Conduct early stopping training:
		EarlyStoppingResult result = trainer.fit();

//Print out the results:
		System.out.println("Termination reason: " + result.getTerminationReason());
		System.out.println("Termination details: " + result.getTerminationDetails());
		System.out.println("Total epochs: " + result.getTotalEpochs());
		System.out.println("Best epoch number: " + result.getBestModelEpoch());
		System.out.println("Score at best epoch: " + result.getBestModelScore());

//Get the best model:
		MultiLayerNetwork model = (MultiLayerNetwork) result.getBestModel();

		testIter.reset();
		System.out.println("Evaluate model: " + (result.getBestModelEpoch()));

		Evaluation eval = new Evaluation(numOutputs);
		while (testIter.hasNext()) {
			DataSet t = testIter.next();
			INDArray features = t.getFeatures();
			INDArray lables = t.getLabels();
			INDArray predicted = model.output(features, true);
			eval.eval(lables, predicted);
			System.out.println(eval.confusionMatrix() + " " + eval.accuracy());
		}

	}
}
