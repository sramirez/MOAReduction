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

package moa.reduction.bayes;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import moa.reduction.core.MOAAttributeEvaluator;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.ContingencyTables;
import weka.core.RevisionUtils;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToBinary;

import com.yahoo.labs.samoa.instances.Instance;

/**
 * <!-- globalinfo-start --> InfoGainAttributeEval :<br/>
 * <br/>
 * Evaluates the worth of an attribute by measuring the information gain with
 * respect to the class.<br/>
 * <br/>
 * InfoGain(Class,Attribute) = H(Class) - H(Class | Attribute).<br/>
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
public class IncrInfoThAttributeEval extends ASEvaluation implements
  AttributeEvaluator, MOAAttributeEvaluator {

  /** for serialization */
  static final long serialVersionUID = -1949849512589218930L;

  /** Treat missing values as a seperate value */
  private boolean m_missing_merge;

  /** Just binarize numeric attributes */
  private boolean m_Binarize;

  /** The info gain for each attribute */
  private double[] m_InfoValues = null;
  
  //private double[][][] counts = null;
  private HashMap<Key, Float>[] counts = null;
  
  private int classIndex;
  
  private boolean updated = false;
  
  private int method = 0;

  /**
   * Returns a string describing this attribute evaluator
   * 
   * @return a description of the evaluator suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String globalInfo() {
    return "InfoGainAttributeEval :\n\nEvaluates the worth of an attribute "
      + "by measuring the information gain with respect to the class.\n\n"
      + "InfoGain(Class,Attribute) = H(Class) - H(Class | Attribute).\n";
  }

  /**
   * Constructor
   */
  public IncrInfoThAttributeEval() {
    resetOptions();    
  }
  
  public IncrInfoThAttributeEval(int method) {
	  	this.method = method;
	    resetOptions();
	    
  }
	@Override
	public boolean isUpdated() {
		// TODO Auto-generated method stub
		return updated;
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
    //result.enable(Capability.MISSING_VALUES);

    // class
    result.enable(Capability.NOMINAL_CLASS);
    //result.enable(Capability.MISSING_CLASS_VALUES);

    return result;
  }

  /**
   * Initializes an information gain attribute evaluator. Discretizes all
   * attributes that are numeric.
   * 
   * @param data set of instances serving as training data
   * @throws Exception if the evaluator has not been generated successfully
   */
  @Override
  public void buildEvaluator(weka.core.Instances data) throws Exception {}
  
  /**
   * Updates an information gain attribute evaluator. Discretizes all
   * attributes that are numeric.
   * 
   * @param data set of instances serving as training data
   * @throws Exception if the evaluator has not been generated successfully
   */
  public void updateEvaluator(Instance inst) throws Exception {
	  
  	if(counts == null) {
	    // can evaluator handle data?
		weka.core.Instance winst = new weka.core.DenseInstance(inst.weight(), inst.toDoubleArray());
		ArrayList<Attribute> list = new ArrayList<Attribute>();
	  	//ArrayList<Attribute> list = Collections.list(winst.enumerateAttributes());
	  	//list.add(winst.classAttribute());
		for(int i = 0; i < inst.numAttributes(); i++) 
			list.add(new Attribute(inst.attribute(i).name(), i));
	  	weka.core.Instances data = new weka.core.Instances("single", list, 1);
	  	data.setClassIndex(inst.classIndex());
	  	data.add(winst);
	    //getCapabilities().testWithFail(data);
	    classIndex = inst.classIndex();
	    counts = (HashMap<Key, Float>[]) new HashMap[inst.numAttributes()];
	    for(int i = 0; i < counts.length; i++) counts[i] = new HashMap<Key, Float>();
  	}
      for (int i = 0; i < inst.numValues(); i++) {
        if (inst.index(i) != classIndex) {
        	Key key = new Key((float) inst.valueSparse(i), (float) inst.classValue());
        	Float cval = (float) (counts[inst.index(i)].getOrDefault(key, 0.0f) + inst.weight());
        	counts[inst.index(i)].put(key, cval);
        }
      }
      
      updated = true;
  }
  
  @Override
  /**
   * Update the contingency tables and the rankings for each features using the counters.
   * Counters are updated in each iteration.
   */
  public void applySelection(){
	  if(counts != null && updated) {
		  m_InfoValues = new double[counts.length];
		    for (int i = 0; i < counts.length; i++) {
		      if (i != classIndex) {
		    	Set<Key> keys = counts[i].keySet();
		    	Set<Entry<Key, Float>> entries = counts[i].entrySet();
		    	
		    	Set<Float> avalues = new HashSet<Float>();
		    	Set<Float> cvalues = new HashSet<Float>();
		    	for (Iterator<Key> it = keys.iterator(); it.hasNext(); ) {
		            Key key = it.next();
		            avalues.add(key.x);
		            cvalues.add(key.y);
		        }
		    	
		    	Map<Float, Integer> apos = new HashMap<Float, Integer>();
		    	Map<Float, Integer> cpos = new HashMap<Float, Integer>();
		    	
		    	int aidx = 0;
		    	for(Iterator<Float> it = avalues.iterator(); it.hasNext();) {
		    		Float f = it.next();
		    		apos.put(f, aidx++);
		    	} 
		    	
		    	int cidx = 0;
		    	for(Iterator<Float> it = cvalues.iterator(); it.hasNext();) {
		    		Float f = it.next();
		    		cpos.put(f, cidx++);
		    	} 
		    	
		    	double[][] lcounts = new double[avalues.size()][cvalues.size()];		    	
	            for (Iterator<Entry<Key, Float>> it = entries.iterator(); it.hasNext(); ) {
		            Entry<Key, Float> entry = it.next();
		            lcounts[apos.get(entry.getKey().x)][cpos.get(entry.getKey().y)] = entry.getValue();
	            }
	            
	            switch (method) {
				case 1:
					m_InfoValues[i] = ContingencyTables.symmetricalUncertainty(lcounts);
					break;

				default:
					m_InfoValues[i] = (ContingencyTables.entropyOverColumns(lcounts) - ContingencyTables
					          .entropyConditionedOnRows(lcounts));
					break;
				}		        
		      }
		    }
		    //System.out.println("Attribute values: " + Arrays.toString(m_InfoValues));
            updated = false;
	  }
  }

  /**
   * Reset options to their default values
   */
  protected void resetOptions() {
    m_InfoValues = null;
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

    return m_InfoValues[attribute];
  }

  /**
   * Describe the attribute evaluator
   * 
   * @return a description of the attribute evaluator as a string
   */
  @Override
  public String toString() {
    StringBuffer text = new StringBuffer();

    if (m_InfoValues == null) {
      text.append("Information Gain attribute evaluator has not been built");
    } else {
      text.append("\tInformation Gain Ranking Filter");
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

  // ============
  // Test method.
  // ============
  /**
   * Main method for testing this class.
   * 
   * @param args the options
   */
  public static void main(String[] args) {
    runEvaluator(new IncrInfoThAttributeEval(), args);
  }
  
  
  private class Key {

	    final float x;
	    final float y;

	    public Key(float x, float y) {
	        this.x = x;
	        this.y = y;
	    }
	    

	    @Override
	    public boolean equals(Object o) {
	        if (this == o) return true;
	        if (!(o instanceof Key)) return false;
	        Key key = (Key) o;
	        return x == key.x && y == key.y;
	    }

	    @Override
	    public int hashCode() {
	        int result = Float.floatToIntBits(x);
	        result = 31 * result + Float.floatToIntBits(y);
	        return result;
	    }

  }
}