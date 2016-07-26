/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    InfoGainAttributeEval.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package moa.classifiers.bayes;

import java.util.Arrays;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.core.AlgVector;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToBinary;

/**
 * <!-- globalinfo-start --> OFSGDAttributeEval :<br/>
 * <br/>
 * Evaluates the worth of an attribute through the computation of weights 
 * using a linear classifier with sparse projection.<br/>
 * <br/>
 * J. Wang, P. Zhao, S. C. H. Hoi and R. Jin, "Online Feature Selection and Its Applications," 
 * in IEEE Transactions on Knowledge and Data Engineering, vol. 26, no. 3, pp. 698-710, March 2014.
 * doi: 10.1109/TKDE.2013.32<br/>
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 * -M
 *  treat missing values as a seperate value.
 * </pre>
 * 
 * <pre>
 * -B
 *  just binarize numeric attributes instead 
 *  of properly discretizing them.
 * </pre>
 * 
 * <!-- options-end -->
 * 
 * @author Mark Hall (mhall@cs.waikato.ac.nz)
 * @version $Revision: 10172 $
 * @see Discretize
 * @see NumericToBinary
 */
public class OFSGDAttributeEval extends ASEvaluation implements
  AttributeEvaluator, MOAAttributeEvaluator {

  /** for serialization */
  static final long serialVersionUID = -1949849512589218930L;

  /** Treat missing values as a seperate value */
  private boolean m_missing_merge;

  /** Just binarize numeric attributes */
  private boolean m_Binarize;
  
  private AlgVector weights = null;
  
  static final double eta = 0.2; // According to authors' criterion  
  static final double lambda = 0.01; // According to authors' criterion  
  private int numFeatures = 10;

  /**
   * Returns a string describing this attribute evaluator
   * 
   * @return a description of the evaluator suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String globalInfo() {
    return "OFSGDAttributeEval :\n\nEvaluates the worth of an attribute "
      + "by measuring performing an stochastic gradient descent approach with feature truncation\n";
  }

  /**
   * Constructor
   */
  public OFSGDAttributeEval(int numFeatures) {
	this.numFeatures = numFeatures;
    resetOptions();
    
  }	  
  
  /**
   * Returns the tip text for this property
   * 
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String binarizeNumericAttributesTipText() {
    return "Just binarize numeric attributes instead of properly discretizing them.";
  }

  /**
   * Binarize numeric attributes.
   * 
   * @param b true=binarize numeric attributes
   */
  public void setBinarizeNumericAttributes(boolean b) {
    m_Binarize = b;
  }

  /**
   * get whether numeric attributes are just being binarized.
   * 
   * @return true if missing values are being distributed.
   */
  public boolean getBinarizeNumericAttributes() {
    return m_Binarize;
  }

  /**
   * Returns the tip text for this property
   * 
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String missingMergeTipText() {
    return "Distribute counts for missing values. Counts are distributed "
      + "across other values in proportion to their frequency. Otherwise, "
      + "missing is treated as a separate value.";
  }

  /**
   * distribute the counts for missing values across observed values
   * 
   * @param b true=distribute missing values.
   */
  public void setMissingMerge(boolean b) {
    m_missing_merge = b;
  }

  /**
   * get whether missing values are being distributed or not
   * 
   * @return true if missing values are being distributed.
   */
  public boolean getMissingMerge() {
    return m_missing_merge;
  }

  /**
   * Returns the capabilities of this evaluator.
   * 
   * @return the capabilities of this evaluator
   * @see Capabilities
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.DATE_ATTRIBUTES);

    // class
    result.enable(Capability.NOMINAL_CLASS);
    return result;
  }
  
  public void updateEvaluator(Instance inst) throws Exception {
	  
	  	if(weights == null) {
	  		weights = new AlgVector(new double[inst.numAttributes() - 1]);
	  		for(int i = 0; i < weights.numElements(); i++) weights.setElement(i, 0); 
	  	}	  	

	  	double[] rawx = Arrays.copyOfRange(inst.toDoubleArray(), 0, inst.numAttributes() - 1);
	  	AlgVector x = new AlgVector(rawx);
	  	double pred = weights.dotMultiply(x); 
	  	
	  	if(pred * inst.classValue() <= 1){
	  		x.scalarMultiply(eta * inst.classValue());
	  		weights = weights.add(x); 
	  		weights.scalarMultiply(Math.min(1.0, 1 / (Math.sqrt(lambda) * weights.norm())));
	  		
	  		int counts = 0;
	  		Pair[] array = new Pair[weights.numElements()];
	  		for(int i = 0; i < weights.numElements(); i++){
	  			array[i] = new Pair(i, weights.getElement(i));
	  			if(weights.getElement(i) != 0) counts++;
	  		}
	  		
	  		// Truncate
	  		if(counts > numFeatures) {
	  			Arrays.sort(array);
	  			for(int i = numFeatures + 1; i < array.length; i++)
	  				weights.setElement(array[i].index, 0);
	  		}
	  	}
  }


	@Override
	public void applySelection() {
		// TODO Auto-generated method stub
		System.out.println("Weight values: " + Arrays.toString(weights.getElements()));		
	}

  /**
   * Reset options to their default values
   */
  protected void resetOptions() {
    m_missing_merge = true;
    m_Binarize = false;
  }

  /**
   * evaluates an individual attribute by measuring the amount of information
   * gained about the class given the attribute.
   * 
   * @param attribute the index of the attribute to be evaluated
   * @return the info gain
   * @throws Exception if the attribute could not be evaluated
   */
  @Override
  public double evaluateAttribute(int attribute) throws Exception {
    return weights.getElement(attribute);
  }

  /**
   * Describe the attribute evaluator
   * 
   * @return a description of the attribute evaluator as a string
   */
  @Override
  public String toString() {
    StringBuffer text = new StringBuffer();

    if (weights == null) {
      text.append("First weights has not been built");
    } else {
      text.append("\n OFSGD Ranking Filter");
      if (!m_missing_merge) {
        text.append("\n\tMissing values treated as seperate");
      }
      if (m_Binarize) {
        text.append("\n\tNumeric attributes are just binarized");
      }
    }

    text.append("\n");
    return text.toString();
  }

  /**
   * Returns the revision string.
   * 
   * @return the revision
   */
  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 10172 $");
  }
  
  class Pair implements Comparable<Pair> {
	    public final int index;
	    public final double value;

	    public Pair(int index, double value) {
	        this.index = index;
	        this.value = value;
	    }

	    @Override
	    public int compareTo(Pair other) {
	        //descending sort order
	        return -1 * Double.valueOf(Math.abs(this.value)).compareTo(Math.abs(other.value));
	    }
	}
  
  @Override
	public void buildEvaluator(Instances arg0) throws Exception {
	// TODO Auto-generated method stub
	
	}

}