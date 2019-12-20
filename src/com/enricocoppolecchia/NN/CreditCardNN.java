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

public class CreditCardNN {
	public static void main(String[] args) throws IOException, InterruptedException {
		String path = args[0];
		int seed = 123;
		double learningRate = 0.001;
		int batchSize = 600;
		int nEpochs = 1;
		int K = 10;
		int numInputs = 5;
		int numOutputs = 2;
		int numHiddenNodes = 6;
		List<Double> tps = new LinkedList<Double>();
		List<Double> tns = new LinkedList<Double>();
		List<Double> best = new LinkedList<Double>();
		List<Double> accs = new LinkedList<Double>();
		
		for (int i = 0; i < 10; i++) {
			// load the training data
			List<Double> tp = new LinkedList<Double>();
			List<Double> tn = new LinkedList<Double>();
			List<Double> acc = new LinkedList<Double>();
			for (int j = 0; j < K; j++) {
				RecordReader rr = new CSVRecordReader();
				rr.initialize(new FileSplit(new File(path + "/train"+j+".csv")));
				DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 5, 2);

				RecordReader rrTest = new CSVRecordReader();
				rrTest.initialize(new FileSplit(new File(path + "/test"+j+".csv")));
				DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 5, 2);

				// Normalize the training data

				DataNormalization normalizer = new NormalizerStandardize();
				normalizer.fit(trainIter); // Collect training data statistics
				testIter.reset();
				testIter.setPreProcessor(normalizer);

				// Use previously collected statistics to normalize on-the-fly. Each DataSet
				// returned by 'trainData' iterator will be normalized

				trainIter.setPreProcessor(normalizer);

				MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed)
						.list()
						.layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes).activation(Activation.SIGMOID)
								.build())
						.layer(new OutputLayer.Builder(LossFunctions.LossFunction.HINGE)
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
					//System.out.println(eval.confusionMatrix() + " " + eval.accuracy());
				}

				// Print the evaluation statistics
				int p = eval.getConfusionMatrix().getCount(0, 0);
				int n = eval.getConfusionMatrix().getCount(1, 1);
				tp.add((double) p);
				tn.add((double) n);
				acc.add(eval.accuracy());
			}
			double positive = 0;
			double negative = 0;
			double accuracy = 0;
			
			for(int p=0;p<tp.size();p++) {
				positive+=tp.get(p);
				negative+=tn.get(p);
				accuracy+=acc.get(p);
			}
			positive/=tp.size();
			negative/=tn.size();
			accuracy/=K;
			tps.add(positive);
			tns.add(negative);
			
			accs.add((double) accuracy);
			best.add((double) ((Math.sqrt(positive * negative)) + accuracy));
		}
		
		int max = 0;
		for (int i = 0; i < best.size(); i++)
			if (best.get(i) > best.get(max))
				max = i;
		System.out.println("The best conf is: epochs: " + (max + 1));
		System.out.println("TP: " + tps.get(max));
		System.out.println("TN: " + tns.get(max));
		System.out.println("Accuracy: " + accs.get(max));

	}
}
