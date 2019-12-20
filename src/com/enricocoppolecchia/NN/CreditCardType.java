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

public class CreditCardType {
	public static void main(String[] args) throws IOException, InterruptedException {
		String path = args[0];
		int seed = 123;
		double learningRate = 0.3;
		int batchSize = 500;
		int nEpochs = 500;
		int K = 10;
		int numInputs = 3;
		int numOutputs = 3;
		int numHiddenNodes = 2;
		List<Double> tps = new LinkedList<Double>();
		List<Double> tns = new LinkedList<Double>();
		List<Double> trs = new LinkedList<Double>();
		List<Double> best = new LinkedList<Double>();
		List<Double> accs = new LinkedList<Double>();
		
		for (int i = 0; i < 5001; i+=500) {
			// load the training data
			List<Double> t1 = new LinkedList<Double>();
			List<Double> t2 = new LinkedList<Double>();
			List<Double> t3 = new LinkedList<Double>();
			List<Double> acc = new LinkedList<Double>();
			for (int j = 0; j < K; j++) {
				RecordReader rr = new CSVRecordReader();
				rr.initialize(new FileSplit(new File(path + "/train"+j+".csv")));
				DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 1, 3);

				RecordReader rrTest = new CSVRecordReader();
				rrTest.initialize(new FileSplit(new File(path + "/test"+j+".csv")));
				DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 1, 3);

				// Normalize the training data

				DataNormalization normalizer = new NormalizerStandardize();
				normalizer.fit(trainIter); // Collect training data statistics
				testIter.reset();
				testIter.setPreProcessor(normalizer);

				// Use previously collected statistics to normalize on-the-fly. Each DataSet
				// returned by 'trainData' iterator will be normalized

				trainIter.setPreProcessor(normalizer);

				MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).list()
						.layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
								.activation(Activation.SOFTMAX).build())
						.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
								.nIn(numHiddenNodes).nOut(numOutputs).build())
						.build();

				MultiLayerNetwork model = new MultiLayerNetwork(conf);
				model.init();
				model.fit(trainIter, nEpochs + i);
				//System.out.println("Evaluate model: " + (nEpochs + i));

				Evaluation eval = new Evaluation(numOutputs);
				while (testIter.hasNext()) {
					DataSet t = testIter.next();
					INDArray features = t.getFeatures();
					INDArray lables = t.getLabels();
					INDArray predicted = model.output(features, false);
					eval.eval(lables, predicted);
					//System.out.println(eval.confusionMatrix() + " " + eval.accuracy());
				}

				// Print the evaluation statistics
				int p = eval.getConfusionMatrix().getCount(0, 0);
				int n = eval.getConfusionMatrix().getCount(1, 1);
				int r = eval.getConfusionMatrix().getCount(2, 2);
				t1.add((double) p);
				t2.add((double) n);
				t3.add((double) r);
				acc.add(eval.accuracy());
			}
			double positive = 0;
			double negative = 0;
			double relative = 0;
			double accuracy = 0;
			
			for(int p=0;p<t1.size();p++) {
				positive+=t1.get(p);
				negative+=t2.get(p);
				relative+=t3.get(p);
				accuracy+=acc.get(p);
			}
			positive/=t1.size();
			negative/=t2.size();
			relative/=t3.size();
			accuracy/=K;
			tps.add(positive);
			tns.add(negative);
			trs.add(relative);
			
			accs.add((double) accuracy);
			best.add((double) ((Math.pow(positive * negative * relative* accuracy,1/4)) ));
			System.out.println("pos: "+positive+" neg: "+negative+" rel: "+relative+" accuracy: "+accuracy);
		}
		
		int max = 0;
		for (int i = 0; i < best.size(); i++)
			if (best.get(i) > best.get(max))
				max = i;
		System.out.println("The best conf is: epochs: " + (max + 1));
		System.out.println("T1: " + tps.get(max));
		System.out.println("T2: " + tns.get(max));
		System.out.println("T3: " + trs.get(max));
		System.out.println("Accuracy: " + accs.get(max));

	}
}
