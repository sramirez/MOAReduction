package moa.reduction.test;

import java.io.IOException;
import java.util.ArrayList;

import weka.classifiers.lazy.IBk;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.Classifier;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.competence.BBNRFullCB;
import moa.classifiers.competence.ICFFullCB;
import moa.classifiers.competence.NEFCSSRR;
import moa.classifiers.lazy.kNN;
import moa.classifiers.meta.FISH;
import moa.core.InstanceExample;
import moa.core.TimingUtils;
import moa.reduction.bayes.IDAdiscretize;
import moa.reduction.bayes.REBdiscretize;
import moa.reduction.core.NaiveBayesReduction;
import moa.streams.ArffFileStream;

public class ExperimentTest {

        public ExperimentTest(){
        }

        public void run(int numInstances, boolean isTesting){
        		//kNN knn = new kNN();
        		//knn.kOption.setValue(3);
        		//Classifier learner = knn;
                
                //RandomRBFGenerator stream = new RandomRBFGenerator();
                ArffFileStream stream = new ArffFileStream("/home/sramirez/datasets/drift/real/elecNormNew.arff", -1);
        		//ArffFileStream stream = new ArffFileStream("/home/sramirez/TEST_FUSINTER/datasets/spambase/spambase-10-1tra-weka.dat", -1);
        		//ArffFileStream stream = new ArffFileStream("/home/sramirez/datasets/drift/artificial/gradual_drift_100k.arff", -1);

                stream.prepareForUse();
                
                //REBdiscretize filter = new REBdiscretize();
                IDAdiscretize filter = new IDAdiscretize();
                filter.setInputStream(stream);
                filter.init();
                filter.prepareForUse();
                

        		Classifier learner = new NaiveBayes();
                learner.setModelContext(filter.getHeader());
                learner.prepareForUse();
                
                ArrayList<Boolean> predictions = new ArrayList<Boolean>();

                int numberSamplesCorrect = 0;
                int numberSamples = 0;
                long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
                while (filter.hasMoreInstances() && numberSamples < numInstances) {
                        Instance trainInst = filter.nextInstance().getData();
//                        System.out.println(trainInst);
                        if (isTesting) {
//                        	System.out.println(Arrays.toString(learner.getVotesForInstance(trainInst)));
                        	boolean predict = learner.correctlyClassifies(trainInst);
                            if (predict){
                                    numberSamplesCorrect++;
                            }
                            predictions.add(predict);
                        }
                        numberSamples++;
                        
                        learner.trainOnInstance(trainInst);
                }
                float accuracy = (float) numberSamplesCorrect/ numberSamples;
                double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread()- evaluateStartTime);
                System.out.println(numberSamples + " instances processed with " + accuracy + " acc. rate in "+time+" seconds.");
    	}
        
        public static void main(String[] args) throws IOException {
        		ExperimentTest exp = new ExperimentTest();
                exp.run(100000, true);
        }
}