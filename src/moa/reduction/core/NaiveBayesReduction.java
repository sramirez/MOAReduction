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
import weka.attributeSelection.*; 
import moa.classifiers.AbstractClassifier;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.GaussianNumericAttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.NominalAttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.NumericAttributeClassObserver;
import moa.core.AutoExpandVector;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.StringUtils;
import moa.reduction.bayes.IDAdiscretize;
import moa.reduction.bayes.IFFDdiscretize;
import moa.reduction.bayes.IncrInfoThAttributeEval;
import moa.reduction.bayes.OCdiscretize;
import moa.reduction.bayes.OFSGDAttributeEval;
import moa.reduction.bayes.PIDdiscretize;
import moa.reduction.bayes.REBdiscretize;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;

import weka.core.Attribute;

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
public class NaiveBayesReduction extends AbstractClassifier {

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
    		"Discretization method to be used: 0. No method. 1. PiD 2. IFFD 3. Online Chi-Merge 4. IDA 5. RebDiscretize", 4, 0, 5);
    public static IntOption winSizeOption = new IntOption("winSize", 'w', 
    		"Window size for model updates", 1, 1, Integer.MAX_VALUE);  
    public IntOption numClassesOption = new IntOption("numClasses", 'c', 
    		"Number of classes for this problem (Online Chi-Merge)", 100, 1, Integer.MAX_VALUE);      
    
    protected static MOAAttributeEvaluator fselector = null;
    protected static MOADiscretize discretizer = null;
    protected int totalCount = 0, classified = 0, correctlyClassified = 0;
    protected Set<Integer> selectedFeatures = new HashSet<Integer>();
    
    @Override
    public void resetLearningImpl() {
	    this.observedClassDistribution = new DoubleVector();
        this.attributeObservers = new AutoExpandVector<AttributeClassObserver>();
        totalCount = 0; classified = 0; correctlyClassified = 0;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
    	
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
    	    		discretizer = new REBdiscretize();
    	    	}
    		}
    		if(discmethodOption.getValue() != 5)
    			discretizer.updateEvaluator(inst);
    		else
        		// REBdiscretize needs to know the error rate before removing instances
    			((REBdiscretize) discretizer).updateEvaluator(inst, 1 - ((float) correctlyClassified / classified)); 
    			
    		System.out.println("Number of new intervals: " + discretizer.getNumberIntervals());
    		rinst = discretizer.applyDiscretization(inst);
    	}
    	
        this.observedClassDistribution.addToValue((int) rinst.classValue(), rinst.weight());
        for (int i = 0; i < rinst.numAttributes() - 1; i++) {
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
        
        totalCount++;
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        return doNaiveBayesPrediction(inst, this.observedClassDistribution,
                this.attributeObservers);
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
    
    private void performFS(Instance rinst) {
    	// Feature selection process performed before
		weka.core.Instance winst = new weka.core.DenseInstance(rinst.weight(), rinst.toDoubleArray());
		
		if(fselector != null) {
			if(fselector.isUpdated() && totalCount % winSizeOption.getValue() == 0) {
		    	fselector.applySelection();
				selector = new AttributeSelection();
				Ranker ranker = new Ranker();
				ranker.setNumToSelect(Math.min(numFeaturesOption.getValue(), winst.numAttributes() - 1));
				selector.setEvaluator((ASEvaluation) fselector);
				selector.setSearch(ranker);
		    	
				ArrayList<Attribute> list = new ArrayList<Attribute>();
			  	//ArrayList<Attribute> list = Collections.list(winst.enumerateAttributes());
			  	//list.add(winst.classAttribute());
				for(int i = 0; i < rinst.numAttributes(); i++) 
					list.add(new Attribute(rinst.attribute(i).name(), i));
				//ArrayList<Attribute> list = Collections.list(winst.enumerateAttributes());
				//list.add(winst.classAttribute());
			  	weka.core.Instances single = new weka.core.Instances ("single", list, 1);
			  	single.setClassIndex(rinst.classIndex());
			  	single.add(winst);
			  	try {
					selector.SelectAttributes(single);
					System.out.println("Selected features: " + selector.toResultsString());
					selectedFeatures.clear();
					for(int att : selector.selectedAttributes())
						selectedFeatures.add(att);
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
    	}
    }

    public double[] doNaiveBayesPrediction(Instance inst,
            DoubleVector observedClassDistribution,
            AutoExpandVector<AttributeClassObserver> attributeObservers) {
    	
    	// Feature selection process performed before
    	Instance sinst = inst.copy();
    	if(fsmethodOption.getValue() != 0 && fselector != null) 
    		performFS(sinst);
    	if(discmethodOption.getValue() != 0 && discretizer != null) 
    		sinst = discretizer.applyDiscretization(sinst);
		
		// Naive Bayes predictions
        double[] votes = new double[observedClassDistribution.numValues()];
        double observedClassSum = observedClassDistribution.sumOfValues();
        for (int classIndex = 0; classIndex < votes.length; classIndex++) {
            votes[classIndex] = observedClassDistribution.getValue(classIndex)
                    / observedClassSum;
            for (int attIndex = 0; attIndex < sinst.numAttributes() - 1; attIndex++) {
            	if(selectedFeatures.isEmpty() || selectedFeatures.contains(attIndex)) {
	                int instAttIndex = modelAttIndexToInstanceAttIndex(attIndex,sinst);
	                AttributeClassObserver obs = attributeObservers.get(attIndex);
	                if ((obs != null) && !sinst.isMissing(instAttIndex)) {
	                	votes[classIndex] *= obs.probabilityOfAttributeValueGivenClass(
	                			sinst.value(instAttIndex), classIndex);
	                }
            	}
            }
        }
        // TODO: need logic to prevent underflow?
        // Compute some statistics about classification performance
        double maxValue = -1;
        int maxIndex = -1;
        for (int i = 0; i < votes.length; i++) {
			if(votes[i] > maxValue){
				maxIndex = i;
				maxValue = votes[i];
			}
		}
        if(maxIndex == inst.classIndex())
        	correctlyClassified++;
        classified++;
        return votes;
    }

    // Naive Bayes Prediction using log10 for VFDR rules 
    public double[] doNaiveBayesPredictionLog(Instance inst,
            DoubleVector observedClassDistribution,
            AutoExpandVector<AttributeClassObserver> observers, AutoExpandVector<AttributeClassObserver> observers2) {
    	
    	Instance rinst = inst.copy();
    	// Feature selection process performed before
    	performFS(rinst);
    	
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