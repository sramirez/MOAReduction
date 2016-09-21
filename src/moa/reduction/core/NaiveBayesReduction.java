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
import moa.core.AutoExpandVector;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.StringUtils;
import moa.options.IntOption;
import moa.reduction.bayes.IFFDdiscretize;
import moa.reduction.bayes.IncrInfoThAttributeEval;
import moa.reduction.bayes.OCdiscretize;
import moa.reduction.bayes.OFSGDAttributeEval;
import moa.reduction.bayes.PIDdiscretize;

import java.util.ArrayList;
import java.util.Collections;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

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
    		"The number of features to select", 25, 1, Integer.MAX_VALUE);
    public static IntOption fsmethodOption = new IntOption("fsMethod", 'm', 
    		"Infotheoretic method to be used in feature selection: 0. No method. 1. InfoGain 2. Symmetrical Uncertainty 3. OFSGD", 1, 0, 3);
    public static IntOption discmethodOption = new IntOption("discMethod", 'd', 
    		"Discretization method to be used: 0. No method. 1. PiD 2. IFFD 3. Online Chi-Merge", 0, 0, 3);
    public static IntOption winSizeOption = new IntOption("winSize", 'w', 
    		"Window size for model updates", 1000, 1, Integer.MAX_VALUE);    
    
    protected static MOAAttributeEvaluator fselector = null;
    protected static MOADiscretize discretizer = null;
    protected int totalCount = 0;
    
    public NaiveBayesReduction () {    	
    	if(fsmethodOption.getValue() == 3) {
    		fselector = new OFSGDAttributeEval(numFeaturesOption.getValue());
    	} else if (fsmethodOption.getValue() == 2 || fsmethodOption.getValue() == 1){
    		fselector = new IncrInfoThAttributeEval(fsmethodOption.getValue());
    	} else {
    		//fselector = null;
    	}
    	
    	if(discmethodOption.getValue() == 1) {
    		discretizer = new PIDdiscretize();
    	} else if(discmethodOption.getValue() == 2) {
    		discretizer = new IFFDdiscretize();	
    	} else if(discmethodOption.getValue() == 3) {
    		discretizer = new OCdiscretize();
    	}
    }

    @Override
    public void resetLearningImpl() {
	    this.observedClassDistribution = new DoubleVector();
        this.attributeObservers = new AutoExpandVector<AttributeClassObserver>();
    }

    @Override
    public void trainOnInstanceImpl(weka.core.Instance rinst) {
    	
    	if(fsmethodOption.getValue() != 0) {
    		try {
				fselector.updateEvaluator(rinst);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
    	}
	    	
    	if(discmethodOption.getValue() != 0) 
    		discretizer.updateEvaluator(rinst);
    	
        this.observedClassDistribution.addToValue((int) rinst.classValue(), rinst.weight());
        for (int i = 0; i < rinst.numAttributes() - 1; i++) {
            int instAttIndex = modelAttIndexToInstanceAttIndex(i, rinst);
            AttributeClassObserver obs = this.attributeObservers.get(i);
            if (obs == null) {
                obs = rinst.attribute(instAttIndex).isNominal() ? newNominalClassObserver()
                        : newNumericClassObserver();
                this.attributeObservers.set(i, obs);
            }
            obs.observeAttributeClass(rinst.value(instAttIndex), (int) rinst.classValue(), rinst.weight());
        }
        
        totalCount++;
    }

    @Override
    public double[] getVotesForInstance(weka.core.Instance inst) {
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
    
    private Instance performFS(Instance rinst) {
    	// Feature selection process performed before
    	Instance sinst = rinst;
		if(fselector != null) {
			if(fselector.isUpdated() && totalCount % winSizeOption.getValue() == 0) {
		    	fselector.applySelection();
				selector = new AttributeSelection();
				Ranker ranker = new Ranker();
				ranker.setNumToSelect(Math.min(numFeaturesOption.getValue(), rinst.numAttributes() - 1));
				selector.setEvaluator((ASEvaluation) fselector);
				selector.setSearch(ranker);
				ArrayList<Attribute> list = Collections.list(rinst.enumerateAttributes());
				list.add(rinst.classAttribute());
			  	Instances single = new Instances("single", list, 1);
			  	single.setClassIndex(rinst.classIndex());
			  	single.add(rinst);
			  	try {
					selector.SelectAttributes(single);
					System.out.println("Selected features: " + selector.toResultsString());
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
    	}    		
		
		if(selector != null) {
			try {
				sinst = selector.reduceDimensionality(rinst);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
    	
    	return sinst;
    }

    public double[] doNaiveBayesPrediction(weka.core.Instance inst,
            DoubleVector observedClassDistribution,
            AutoExpandVector<AttributeClassObserver> attributeObservers) {
    	
    	// Feature selection process performed before
    	Instance sinst = inst;
    	if(fsmethodOption.getValue() != 0) sinst = performFS(sinst);
    	if(discmethodOption.getValue() != 0) sinst = discretizer.applyDiscretization(sinst);
		
		// Naive Bayes predictions
        double[] votes = new double[observedClassDistribution.numValues()];
        double observedClassSum = observedClassDistribution.sumOfValues();
        for (int classIndex = 0; classIndex < votes.length; classIndex++) {
            votes[classIndex] = observedClassDistribution.getValue(classIndex)
                    / observedClassSum;
            for (int attIndex = 0; attIndex < sinst.numAttributes() - 1; attIndex++) {
                int instAttIndex = modelAttIndexToInstanceAttIndex(attIndex,
                        sinst);
                AttributeClassObserver obs = attributeObservers.get(attIndex);
                if ((obs != null) && !sinst.isMissing(instAttIndex)) {
                    votes[classIndex] *= obs.probabilityOfAttributeValueGivenClass(sinst.value(instAttIndex), classIndex);
                }
            }
        }
        // TODO: need logic to prevent underflow?
        return votes;
    }

    // Naive Bayes Prediction using log10 for VFDR rules 
    public double[] doNaiveBayesPredictionLog(weka.core.Instance rinst,
            DoubleVector observedClassDistribution,
            AutoExpandVector<AttributeClassObserver> observers, AutoExpandVector<AttributeClassObserver> observers2) {
    	
    	// Feature selection process performed before
    	Instance inst = performFS(rinst);
    	
        AttributeClassObserver obs;
        double[] votes = new double[observedClassDistribution.numValues()];
        double observedClassSum = observedClassDistribution.sumOfValues();
        for (int classIndex = 0; classIndex < votes.length; classIndex++) {
            votes[classIndex] = Math.log10(observedClassDistribution.getValue(classIndex)
                    / observedClassSum);
            for (int attIndex = 0; attIndex < inst.numAttributes() - 1; attIndex++) {
                int instAttIndex = modelAttIndexToInstanceAttIndex(attIndex,
                        inst);
                if (inst.attribute(instAttIndex).isNominal()) {
                    obs = observers.get(attIndex);
                } else {
                    obs = observers2.get(attIndex);
                }

                if ((obs != null) && !inst.isMissing(instAttIndex)) {
                    votes[classIndex] += Math.log10(obs.probabilityOfAttributeValueGivenClass(inst.value(instAttIndex), classIndex));

                }
            }
        }
        return votes;

    }

    public void manageMemory(int currentByteSize, int maxByteSize) {
        // TODO Auto-generated method stub
    }
}