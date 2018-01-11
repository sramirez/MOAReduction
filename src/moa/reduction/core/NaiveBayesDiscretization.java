/*
 *    NaiveBayes.java
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.reduction.core;


//import weka.attributeSelection.InfoGainAttributeEval; 
//import weka.attributeSelection.Ranker;
//import weka.attributeSelection.AttributeSelection;
import java.util.HashSet;
import java.util.Set;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.GaussianNumericAttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.NominalAttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.NumericAttributeClassObserver;
import moa.core.AutoExpandVector;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.StringUtils;
import moa.core.TimingUtils;
import moa.reduction.bayes.IDAdiscretize;
import moa.reduction.bayes.IFFDdiscretize;
import moa.reduction.bayes.IncrInfoThAttributeEval;
import moa.reduction.bayes.LFODiscretizer;
import moa.reduction.bayes.OCdiscretize;
import moa.reduction.bayes.OFSGDAttributeEval;
import moa.reduction.bayes.PIDdiscretize;
import weka.attributeSelection.AttributeSelection;
import weka.core.Utils;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;

/**
 * Naive Bayes incremental learner.
 *
 * <p>Performs classic bayesian prediction while making naive assumption that
 * all inputs are independent.<br /> Naive Bayes is a classiﬁer algorithm known
 * for its simplicity and low computational cost. Given n different classes, the
 * trained Naive Bayes classiﬁer predicts for every unlabelled instance I the
 * class C to which it belongs with high accuracy.</p>
 *
 * <p>Parameters:</p> <ul> <li>-r : Seed for random behaviour of the
 * classifier</li> </ul>
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 7 $
 */
public class NaiveBayesDiscretization extends AbstractClassifier {

    private static final long serialVersionUID = 1L;

    @Override
    public String getPurposeString() {
        return "Naive Bayes classifier with feature selection: performs classic bayesian prediction while making naive assumption that all inputs are independent.";
    }
    protected DoubleVector observedClassDistribution;
    protected static AttributeSelection selector = null;
    protected AutoExpandVector<AttributeClassObserver> attributeObservers;

    public static IntOption numFeaturesOption = new IntOption("numFeatures", 'f', 
    		"The number of features to select", 10, 1, Integer.MAX_VALUE);
    public static IntOption fsmethodOption = new IntOption("fsMethod", 'm', 
    		"Infotheoretic method to be used in feature selection: 0. No method. 1. InfoGain 2. Symmetrical Uncertainty 3. OFSGD", 0, 0, 3);
    public static IntOption discmethodOption = new IntOption("discMethod", 'd', 
    		"Discretization method to be used: 0. No method. 1. PiD 2. IFFD 3. Online Chi-Merge 4. IDA 5. LOFD", 1, 0, 5);
    public static IntOption winSizeOption = new IntOption("winSize", 'w', 
    		"Window size for model updates", 5000, 1, Integer.MAX_VALUE);  
    public static IntOption thresholdOption = new IntOption("threshold", 't', 
    		"Threshold for initialization", 10000, 1, Integer.MAX_VALUE);  
    public static IntOption decimalsOption = new IntOption("decimals", 'e', 
    		"Number of decimals to round", 3, 0, Integer.MAX_VALUE); 
    public static IntOption maxLabelsOption = new IntOption("maxLabels", 'l', 
    		"Number of different labels to use in discretization", 10000, 10, Integer.MAX_VALUE); 
    public IntOption numClassesOption = new IntOption("numClasses", 'c', 
    		"Number of classes for this problem (Online Chi-Merge)", 100, 1, Integer.MAX_VALUE); 
    protected long trainTotalTime = 0, predictTotalTime = 0;
    
    protected static MOAAttributeEvaluator fselector = null;
    protected static MOADiscretize discretizer = null;
    protected int totalCount = 0, classified = 0, correctlyClassified = 0;
    protected Set<Integer> selectedFeatures = new HashSet<Integer>();
	//private double sumTime, sumTime2;
    
    @Override
    public void resetLearningImpl() {
	    this.observedClassDistribution = new DoubleVector();
        this.attributeObservers = new AutoExpandVector<AttributeClassObserver>();
        totalCount = 0; classified = 0; correctlyClassified = 0;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
    	
    	long trainStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
    	
    	Instance rinst = inst.copy();
    	// Update the FS evaluator (no selection is applied here)
    	if(fsmethodOption.getValue() != 0) {
    		if(fselector == null) {
    			if(fsmethodOption.getValue() == 3) {
    	    		fselector = new OFSGDAttributeEval(numFeaturesOption.getValue());
    	    	} else if (fsmethodOption.getValue() == 2 || fsmethodOption.getValue() == 1){
    	    		fselector = new IncrInfoThAttributeEval(fsmethodOption.getValue());
    	    	} else {
    	    		//fselector = null;
    	    	}
    		}
    		try {
    			if(inst == null) {
    				System.err.println("Error: null instance");
    			}
				fselector.updateEvaluator(inst);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
    	}
	    	
    	// Update the discretization scheme, and apply it to the given instance
    	//long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
		
    	if(discmethodOption.getValue() != 0) {
    		if(discretizer == null) {
    			if(discmethodOption.getValue() == 1) {
    	    		discretizer = new PIDdiscretize();
    	    	} else if(discmethodOption.getValue() == 2) {
    	    		discretizer = new IFFDdiscretize();	
    	    	} else if(discmethodOption.getValue() == 3) {
    	    		discretizer = new OCdiscretize(this.numClassesOption.getValue());
    	    	} else if(discmethodOption.getValue() == 4){
    	    		discretizer = new IDAdiscretize();
    	    	} else {
    	    		discretizer = new LFODiscretizer(winSizeOption.getValue(), 
    	    				thresholdOption.getValue(), decimalsOption.getValue(), maxLabelsOption.getValue());
    	    	}
    		} else {

    			discretizer.updateEvaluator(inst);

    		}
    		System.out.println("Number of new intervals: " + discretizer.getNumberIntervals());	
    		rinst = discretizer.applyDiscretization(inst);
    	}

		  //sumTime += TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - evaluateStartTime);
    	
        this.observedClassDistribution.addToValue((int) rinst.classValue(), rinst.weight());
        for (int i = 0; i < rinst.numAttributes() - 1; i++) {        		
        		if(!discretizedAttribute(i)) {
                    int instAttIndex = modelAttIndexToInstanceAttIndex(i, rinst);
                    AttributeClassObserver obs = this.attributeObservers.get(i);
                    com.yahoo.labs.samoa.instances.Attribute att = rinst.attribute(instAttIndex);
                    if (obs == null || (att.isNominal() && obs instanceof NumericAttributeClassObserver)) {
                        obs = att.isNominal() ? newNominalClassObserver()
                                : newNumericClassObserver();
                        this.attributeObservers.set(i, obs);
                    }                    

                    // Problem with spam assasin
                    double value = rinst.value(instAttIndex);
                    if(rinst.value(instAttIndex) == -1) {
                    	System.out.println("Value changed");
                    	value = 0;            	
                    }
                    obs.observeAttributeClass(value, (int) rinst.classValue(), rinst.weight());
        		}
        }                
        totalCount++;
        trainTotalTime += TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - trainStartTime);
        if(totalCount == 10000)
            	System.out.println("Partial train time: " + trainTotalTime);
    }
    
    private boolean discretizedAttribute(int attIndex){
    	return discretizer != null && discretizer.m_Init &&
    			discretizer.m_CutPoints[attIndex] != null && discretizer.provideProb;
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {

    	long predictStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
    	double[] out = doNaiveBayesPrediction(inst, this.observedClassDistribution,
				this.attributeObservers);
    	predictTotalTime += TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - predictStartTime);
    	if(totalCount == 10000)
        	System.out.println("Partial prediction time: " + predictTotalTime);
    	return out; 
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        for (int i = 0; i < this.observedClassDistribution.numValues(); i++) {
            StringUtils.appendIndented(out, indent, "Observations for ");
            out.append(getClassNameString());
            out.append(" = ");
            out.append(getClassLabelString(i));
            out.append(":");
            StringUtils.appendNewlineIndented(out, indent + 1,
                    "Total observed weight = ");
            out.append(this.observedClassDistribution.getValue(i));
            out.append(" / prob = ");
            out.append(this.observedClassDistribution.getValue(i)
                    / this.observedClassDistribution.sumOfValues());
            for (int j = 0; j < this.attributeObservers.size(); j++) {
                StringUtils.appendNewlineIndented(out, indent + 1,
                        "Observations for ");
                out.append(getAttributeNameString(j));
                out.append(": ");
                // TODO: implement observer output
                out.append(this.attributeObservers.get(j));
            }
            StringUtils.appendNewline(out);
        }
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    protected AttributeClassObserver newNominalClassObserver() {
        return new NominalAttributeClassObserver();
    }

    protected AttributeClassObserver newNumericClassObserver() {
        return new GaussianNumericAttributeClassObserver();
    }
    
    public double[] doNaiveBayesPrediction(Instance inst,
            DoubleVector observedClassDistribution,
            AutoExpandVector<AttributeClassObserver> attributeObservers) {
    	
    	// Feature selection process performed before
    	Instance sinst = inst.copy();
    	if(discmethodOption.getValue() != 0 && discretizer != null) {
    		sinst = discretizer.applyDiscretization(sinst);
    	}
    	
		// Naive Bayes predictions
        double[] votes = new double[observedClassDistribution.numValues()];
        double observedClassSum = observedClassDistribution.sumOfValues();
        double[] originalClassProb = new double[observedClassDistribution.numValues()];
        for (int classIndex = 0; classIndex < votes.length; classIndex++) {
            votes[classIndex] = observedClassDistribution.getValue(classIndex)
                    / observedClassSum;
            originalClassProb[classIndex] = votes[classIndex]; // copy this value
            for (int attIndex = 0; attIndex < sinst.numAttributes() - 1; attIndex++) {
            	if(selectedFeatures.isEmpty() || selectedFeatures.contains(attIndex)) {
	                int instAttIndex = modelAttIndexToInstanceAttIndex(attIndex,sinst);
	                if (!sinst.isMissing(instAttIndex)) {
	                	if(discretizedAttribute(instAttIndex)) {
	                		Float cond = discretizer.condProbGivenClass(instAttIndex, 
		                			inst.value(instAttIndex), (int) sinst.value(instAttIndex),
		                			//classIndex, (float) originalClassProb[classIndex]);
		                			classIndex, (int) observedClassDistribution.getValue(classIndex));
	                		if(cond != null)
	                			votes[classIndex] *= cond;
	                	} else {
	                		AttributeClassObserver obs = attributeObservers.get(instAttIndex);
	    	                if (obs != null) {
	    	                	votes[classIndex] *= obs.probabilityOfAttributeValueGivenClass(
	    	                			sinst.value(instAttIndex), classIndex);
	    	                }
	                	}
	                }
            	}
            }
        }
        // TODO: need logic to prevent underflow?
        // Compute some statistics about classification performance
        if(Utils.maxIndex(votes) == inst.classIndex())
        	correctlyClassified++;
        classified++;
        return votes;
    }

    // Naive Bayes Prediction using log10 for VFDR rules 
    public double[] doNaiveBayesPredictionLog(Instance inst,
            DoubleVector observedClassDistribution,
            AutoExpandVector<AttributeClassObserver> observers, AutoExpandVector<AttributeClassObserver> observers2) {
    	
    	Instance rinst = inst.copy();
    	
        AttributeClassObserver obs;
        double[] votes = new double[observedClassDistribution.numValues()];
        double observedClassSum = observedClassDistribution.sumOfValues();
        for (int classIndex = 0; classIndex < votes.length; classIndex++) {
            votes[classIndex] = Math.log10(observedClassDistribution.getValue(classIndex)
                    / observedClassSum);
            for (int attIndex = 0; attIndex < rinst.numAttributes() - 1; attIndex++) {
                int instAttIndex = modelAttIndexToInstanceAttIndex(attIndex,
                       rinst);
                if (rinst.attribute(instAttIndex).isNominal()) {
                    obs = observers.get(attIndex);
                } else {
                    obs = observers2.get(attIndex);
                }

                if ((obs != null) && !rinst.isMissing(instAttIndex)) {
                    votes[classIndex] += Math.log10(obs.probabilityOfAttributeValueGivenClass(rinst.value(instAttIndex), classIndex));

                }
            }
        }
        return votes;

    }
}