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

import weka.attributeSelection.*; 
import moa.classifiers.AbstractClassifier;
import moa.classifiers.bayes.NaiveBayesMultinomial;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.functions.SGDMultiClass;
import moa.core.AutoExpandVector;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.StringUtils;
import moa.reduction.bayes.IDAdiscretize;
import moa.reduction.bayes.IFFDdiscretize;
import moa.reduction.bayes.IncrInfoThAttributeEval;
import moa.reduction.bayes.LFODiscretizer;
import moa.reduction.bayes.OCdiscretize;
import moa.reduction.bayes.OFSGDAttributeEval;
import moa.reduction.bayes.PIDdiscretize;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.WekaToSamoaInstanceConverter;

import weka.core.Attribute;

/**
 * Wrapper classifier with several preprocessing methods.
 *
 * <p>Performs classic bayesian prediction or multinomial SGD-based linear classification.
 * 
 * @author Sergio Ramirez (sramirez@decsai.ugr.es)
 * @version $Revision: 2 $
 */
public class ReductionClassifier extends AbstractClassifier {

    private static final long serialVersionUID = 1L;

    @Override
    public String getPurposeString() {
        return "Wrapper classifier with several preprocessing methods: up to date, only multinomial NB and SGD logistic regresion are considered.";
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
    		"Window size for model updates", 5000, 1, Integer.MAX_VALUE);  
    public static IntOption thresholdOption = new IntOption("threshold", 't', 
    		"Threshold for initialization", 10000, 1, Integer.MAX_VALUE);  
    public static IntOption decimalsOption = new IntOption("decimals", 'e', 
    		"Number of decimals to round", 3, 0, Integer.MAX_VALUE); 
    public static IntOption maxLabelsOption = new IntOption("maxLabels", 'l', 
    		"Number of different labels to use in discretization", 10000, 10, Integer.MAX_VALUE); 
    public IntOption numClassesOption = new IntOption("numClasses", 'c', 
    		"Number of classes for this problem (Online Chi-Merge)", 100, 1, Integer.MAX_VALUE);   
    public IntOption baseClassifier = new IntOption("baseClassifier", 'b', 
    		"Base classifier to be used: 0. Multinomial NB 1. LR (SGD Multiclass).", 1, 0, 1); 
    
    protected static MOAAttributeEvaluator fselector = null;
    protected static MOADiscretize discretizer = null;
    protected int totalCount = 0, classified = 0, correctlyClassified = 0;
    protected Set<Integer> selectedFeatures = new HashSet<Integer>();
    protected AbstractClassifier wrapperClassifier;
	//private double sumTime, sumTime2;
    
    public ReductionClassifier() {
		// TODO Auto-generated constructor stub
    	if(baseClassifier.getValue() == 0){
    		wrapperClassifier = new NaiveBayesMultinomial();
    	} else {
    		SGDMultiClass tmp = new SGDMultiClass();
    		tmp.setLossFunction(2);
    		wrapperClassifier = tmp;
    	}
    	wrapperClassifier.resetLearningImpl();
    }
    
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
    	wrapperClassifier.trainOnInstanceImpl(rinst);
        
        
        totalCount++;
        //if(totalCount == 50000)
        	//System.out.println("Total time: " + sumTime);
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {

    	// Profiling
    	/*long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
		double[] prediction = doNaiveBayesPrediction(inst, this.observedClassDistribution,
				this.attributeObservers);

		  sumTime2 += TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - evaluateStartTime);

	        if(totalCount == 49999)
	        	System.out.println("Total time: " + sumTime2);
        return prediction;*/
    	// Feature selection process performed before
    	
    	Instance sinst = inst.copy();
    	if(fsmethodOption.getValue() != 0 && fselector != null) 
    		sinst = performFS(sinst);
    	if(discmethodOption.getValue() != 0 && discretizer != null) 
    		sinst = discretizer.applyDiscretization(sinst);
    	
    	
    	double[] finalVotes = wrapperClassifier.getVotesForInstance(sinst);
    	
        double maxValue = Integer.MIN_VALUE;
        int maxIndex = Integer.MIN_VALUE;
        for (int i = 0; i < finalVotes.length; i++) {
			if(finalVotes[i] > maxValue){
				maxIndex = i;
				maxValue = finalVotes[i];
			}
		}
        if(maxIndex == inst.classIndex())
        	correctlyClassified++;
        classified++;
        return finalVotes;
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder result, int indent) {
        StringUtils.appendIndented(result, indent, toString());
        StringUtils.appendNewline(result);
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }
    
    private Instance performFS(Instance rinst) {
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
					WekaToSamoaInstanceConverter convWS = new WekaToSamoaInstanceConverter();
					return convWS.samoaInstance(selector.reduceDimensionality(winst));
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
    	}
		return rinst;
    }
}