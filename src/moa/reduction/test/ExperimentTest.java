package moa.reduction.test;

import java.io.IOException;

import weka.classifiers.lazy.IBk;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.Classifier;
import moa.classifiers.competence.BBNRFullCB;
import moa.classifiers.competence.ICFFullCB;
import moa.classifiers.competence.NEFCSSRR;
import moa.classifiers.lazy.kNN;
import moa.classifiers.meta.FISH;
import moa.classifiers.meta.LearnNSE;
import moa.core.InstanceExample;
import moa.core.TimingUtils;
import moa.reduction.core.NaiveBayesReduction;
import moa.streams.ArffFileStream;
import moa.streams.generators.RandomRBFGenerator;

public class ExperimentTest {

        public ExperimentTest(){
        }

        public void run(int numInstances, boolean isTesting){
        		//kNN knn = new kNN();
        		//knn.kOption.setValue(3);
        		//Classifier learner = knn;
        		Classifier learner = new NaiveBayesReduction();
                
                //RandomRBFGenerator stream = new RandomRBFGenerator();
                ArffFileStream stream = new ArffFileStream("/home/sramirez/datasets/drift/real/spambase.arff", -1);
        		//stream.numAttsOption.setValue(1000);
                stream.prepareForUse();

                learner.setModelContext(stream.getHeader());
                learner.prepareForUse();

                int numberSamplesCorrect = 0;
                int numberSamples = 0;
                long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
                while (stream.hasMoreInstances() && numberSamples < numInstances) {
                        InstanceExample trainInst = stream.nextInstance();
                        if (isTesting) {
                                if (learner.correctlyClassifies(trainInst.getData())){
                                        numberSamplesCorrect++;
                                }
                        }
                        numberSamples++;
                        learner.trainOnInstance(trainInst);
                }
                double accuracy = 100.0 * (double) numberSamplesCorrect/ (double) numberSamples;
                double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread()- evaluateStartTime);
                System.out.println(numberSamples + " instances processed with " + accuracy + "% accuracy in "+time+" seconds.");
        }

        public static void main(String[] args) throws IOException {
        		ExperimentTest exp = new ExperimentTest();
                exp.run(10000, true);
        }
}