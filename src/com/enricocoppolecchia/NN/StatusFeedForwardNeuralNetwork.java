package com.enricocoppolecchia.NN;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
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
import java.util.LinkedList;
import java.util.List;

public class StatusFeedForwardNeuralNetwork {
	public static void main(String[] args) throws IOException, InterruptedException {
		String path = args[0];
		int seed = 123;
		double learningRate = 0.001;
		int batchSize = 600;
		int nEpochs = 1;
		int numInputs = 6;
		int numOutputs = 2;
		int numHiddenNodes = 5;
		List<Integer> tp = new LinkedList<Integer>();
		List<Integer> tn = new LinkedList<Integer>();
		List<Float> best = new LinkedList<Float>();
		List<Double> acc = new LinkedList<Double>();
		
		for (int i = 0; i < 20; i++) {
			// load the training data
			
			RecordReader rr = new CSVRecordReader();
			rr.initialize(new FileSplit(new File(path+"/unlabeled.csv")));
			DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 2);

			RecordReader rrTest = new CSVRecordReader();
			rrTest.initialize(new FileSplit(new File(path+"/labeled.csv")));
			DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0, 2);

			// Normalize the training data

			DataNormalization normalizer = new NormalizerStandardize();
			normalizer.fit(trainIter); // Collect training data statistics
			trainIter.reset();
			trainIter.setPreProcessor(normalizer);

			// Use previously collected statistics to normalize on-the-fly. Each DataSet
			// returned by 'trainData' iterator will be normalized

			testIter.setPreProcessor(normalizer);

			MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).weightInit(WeightInit.NORMAL)
					.updater(new Nesterovs(learningRate, 0.9)).list()
					.layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes).activation(Activation.GELU)
							.build())
					.layer(new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes).activation(Activation.RELU)
							.build())
					.layer(new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
							.activation(Activation.LEAKYRELU).build())
					.layer(new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
							.activation(Activation.RELU6).build())
					.layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT).activation(Activation.SIGMOID)
							.nIn(numHiddenNodes).nOut(numOutputs).build())
					.build();

			MultiLayerNetwork model = new MultiLayerNetwork(conf);
			model.init();

			model.fit(trainIter, nEpochs + i);

			System.out.println("Evaluate model: " + (nEpochs + i));

			Evaluation eval = new Evaluation(numOutputs);

			while (testIter.hasNext()) {

				DataSet t = testIter.next();

				INDArray features = t.getFeatures();
				INDArray lables = t.getLabels();
				INDArray predicted = model.output(features, false);
				
				eval.eval(lables, predicted);
				//System.out.println(eval.confusionMatrix()+ " "+ eval.accuracy());
				
			}

			// Print the evaluation statistics
			int p = eval.getConfusionMatrix().getCount(0,0);
			int n = eval.getConfusionMatrix().getCount(1,1);
			tp.add(p);
			tn.add(n);
			best.add((float) ((Math.sqrt(p*n))+eval.accuracy()));
			acc.add(eval.accuracy());
		}
		
		int max =0;
		for(int i=0;i<best.size();i++)
			if( best.get(i)>best.get(max)) 
				max=i;
		System.out.println("The best conf is: epochs: "+ (max+1));
		System.out.println("TP: "+tp.get(max));
		System.out.println("TN: "+tn.get(max));
		System.out.println("Accuracy: "+acc.get(max));
		
	}
}
