package moa.reduction.test;

import java.io.IOException;

import moa.classifiers.Classifier;
import moa.classifiers.competence.NEFCSSRR;
import moa.classifiers.meta.FISH;
import moa.classifiers.meta.LearnNSE;
import moa.core.TimingUtils;
import moa.reduction.core.NaiveBayesReduction;
import moa.streams.ArffFileStream;
import moa.streams.generators.RandomRBFGenerator;
import weka.core.Instance;


public class ExperimentTest {

        public ExperimentTest(){
        }

        public void run(int numInstances, boolean isTesting){
                Classifier learner = new LearnNSE();
                
                RandomRBFGenerator stream = new RandomRBFGenerator();
                //stream.numAttsOption.setValue(1000);
                stream.prepareForUse();

                learner.setModelContext(stream.getHeader());
                learner.prepareForUse();

                int numberSamplesCorrect = 0;
                int numberSamples = 0;
                long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
                while (stream.hasMoreInstances() && numberSamples < numInstances) {
                        Instance trainInst = stream.nextInstance();
                        if (isTesting) {
                                if (learner.correctlyClassifies(trainInst)){
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