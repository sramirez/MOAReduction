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
 *    Discretize.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package moa.reduction.bayes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import moa.reduction.core.MOADiscretize;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.ContingencyTables;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Range;
import weka.core.RevisionUtils;
import weka.core.SparseInstance;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.SupervisedFilter;

/**
 <!-- globalinfo-start -->
 * An instance filter that discretizes a range of numeric attributes in the dataset into nominal attributes. Discretization is by Fayyad &amp; Irani's MDL method (the default).<br/>
 * <br/>
 * For more information, see:<br/>
 * <br/>
 * Usama M. Fayyad, Keki B. Irani: Multi-interval discretization of continuousvalued attributes for classification learning. In: Thirteenth International Joint Conference on Articial Intelligence, 1022-1027, 1993.<br/>
 * <br/>
 * Igor Kononenko: On Biases in Estimating Multi-Valued Attributes. In: 14th International Joint Conference on Articial Intelligence, 1034-1040, 1995.
 * <p/>
 <!-- globalinfo-end -->
 * 
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;inproceedings{Fayyad1993,
 *    author = {Usama M. Fayyad and Keki B. Irani},
 *    booktitle = {Thirteenth International Joint Conference on Articial Intelligence},
 *    pages = {1022-1027},
 *    publisher = {Morgan Kaufmann Publishers},
 *    title = {Multi-interval discretization of continuousvalued attributes for classification learning},
 *    volume = {2},
 *    year = {1993}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 * 
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -R &lt;col1,col2-col4,...&gt;
 *  Specifies list of columns to Discretize. First and last are valid indexes.
 *  (default none)</pre>
 * 
 * <pre> -V
 *  Invert matching sense of column indexes.</pre>
 * 
 * <pre> -D
 *  Output binary attributes for discretized attributes.</pre>
 * 
 * <pre> -Y
 *  Use bin numbers rather than ranges for discretized attributes.</pre>
 * 
 * <pre> -E
 *  Use better encoding of split point for MDL.</pre>
 * 
 * <pre> -precision &lt;integer&gt;
 *  Precision for bin boundary labels.
 *  (default = 6 decimal places).</pre>
 * 
 <!-- options-end -->
 * 
 * @author Len Trigg (trigg@cs.waikato.ac.nz)
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 12037 $
 */
public class PIDdiscretize extends Filter implements SupervisedFilter, 
	TechnicalInformationHandler, MOADiscretize{

  /** for serialization */
  static final long serialVersionUID = -3141006402280129097L;

  /** Stores which columns to Discretize */
  protected Range m_DiscretizeCols = new Range();

  /** The number of bins to divide the attribute into */
  protected float alpha = 0.75f;
  
  protected float totalCount = 0;

  protected int minExamples = 1000;
  
  protected int l2UpdateExamples = 10000;
  
  /** The desired weight of instances per bin */
  protected double m_DesiredWeightOfInstancesPerInterval = -1;

  /** Store the current cutpoints */  
  protected List<List<Double>> m_CutPointsL1 = null;
  
  protected List<List<Integer>> m_Counts = null;
  
  protected List<List<Map<Integer, Integer>>> m_Distrib = null;
  
  protected double[][] m_CutPointsL2 = null;
  
  protected int initialBinsL1 = 200;
  
  protected double min = 0.0;
  
  protected double max = 1.0;
  
  protected double step = -1.0;

  /** Output binary attributes for discretized attributes. */
  protected boolean m_MakeBinary = false;

  /** Use bin numbers rather than ranges for discretized attributes. */
  protected boolean m_UseBinNumbers = false;

  /** Use better encoding of split point for MDL. */
  protected boolean m_UseBetterEncoding = false;

  /** Precision for bin range labels */
  protected int m_BinRangePrecision = 6;

  /** Constructor - initialises the filter */
  public PIDdiscretize() {

    setAttributeIndices("first-last");
  }

  /**
   * Returns the Capabilities of this filter.
   * 
   * @return the capabilities of this object
   * @see Capabilities
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enableAllAttributes();
    result.enable(Capability.MISSING_VALUES);

    // class
    result.enable(Capability.NOMINAL_CLASS);

    return result;
  }

  /**
   * Returns a string describing this filter
   * 
   * @return a description of the filter suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String globalInfo() {

    return "An instance filter that discretizes a range of numeric"
      + " attributes in the dataset into nominal attributes."
      + " Discretization is by Fayyad & Irani's MDL method (the default).\n\n"
      + "For more information, see:\n\n" + getTechnicalInformation().toString();
  }

  /**
   * Returns an instance of a TechnicalInformation object, containing detailed
   * information about the technical background of this class, e.g., paper
   * reference or book this class is based on.
   * 
   * @return the technical information about this class
   */
  @Override
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation result;

    result = new TechnicalInformation(Type.INPROCEEDINGS);
    result.setValue(Field.AUTHOR, "Usama M. Fayyad and Keki B. Irani");
    result
      .setValue(
        Field.TITLE,
        "Multi-interval discretization of continuousvalued attributes for classification learning");
    result.setValue(Field.BOOKTITLE,
      "Thirteenth International Joint Conference on Articial Intelligence");
    result.setValue(Field.YEAR, "1993");
    result.setValue(Field.VOLUME, "2");
    result.setValue(Field.PAGES, "1022-1027");
    result.setValue(Field.PUBLISHER, "Morgan Kaufmann Publishers");

    return result;
  }
  
  /**
   * Gets the current range selection
   * 
   * @return a string containing a comma separated list of ranges
   */
  public String getAttributeIndices() {

    return m_DiscretizeCols.getRanges();
  }

  /**
   * Sets which attributes are to be Discretized (only numeric attributes among
   * the selection will be Discretized).
   * 
   * @param rangeList a string representing the list of attributes. Since the
   *          string will typically come from a user, attributes are indexed
   *          from 1. <br>
   *          eg: first-3,5,6-last
   * @throws IllegalArgumentException if an invalid range list is supplied
   */
  public void setAttributeIndices(String rangeList) {

    m_DiscretizeCols.setRanges(rangeList);
  }

  /**
   * Sets which attributes are to be Discretized (only numeric attributes among
   * the selection will be Discretized).
   * 
   * @param attributes an array containing indexes of attributes to Discretize.
   *          Since the array will typically come from a program, attributes are
   *          indexed from 0.
   * @throws IllegalArgumentException if an invalid set of ranges is supplied
   */
  public void setAttributeIndicesArray(int[] attributes) {

    setAttributeIndices(Range.indicesToRangeList(attributes));
  }

  /**
   * Gets the cut points for an attribute
   * 
   * @param attributeIndex the index (from 0) of the attribute to get the cut
   *          points of
   * @return an array containing the cutpoints (or null if the attribute
   *         requested isn't being Discretized
   */
  public double[] getCutPoints(int attributeIndex) {

    if (m_CutPointsL2 == null) {
      return null;
    }
    
    return m_CutPointsL2[attributeIndex];
  }
  
  private void initializeL1(Instance inst){
	  m_DiscretizeCols.setUpper(inst.numAttributes() - 1);
	  step = (max - min) / (double) initialBinsL1;
	  m_CutPointsL1 = new ArrayList<List<Double>>(inst.numAttributes());
	  m_Counts = new ArrayList<List<Integer>>(inst.numAttributes());
	  m_Distrib = new ArrayList<List<Map<Integer, Integer>>>(inst.numAttributes());
	  for (int i = 0; i < inst.numAttributes(); i++) {
		  Double[] initialb = new Double[initialBinsL1 + 1];
		  Integer[] initialc = new Integer[initialBinsL1 + 1];
		  List<Map<Integer, Integer>> initiald = new ArrayList<Map<Integer, Integer>>(initialBinsL1 + 1);
		  
		  for (int j = 0; j < initialBinsL1 + 1; j++) {
			initialb[j] = min + j * step;
			initialc[j] = 0;
			initiald.add(new HashMap<Integer, Integer>());
		  }
		  m_CutPointsL1.add(new ArrayList<Double>(Arrays.asList(initialb)));
		  m_Counts.add(new ArrayList<Integer>(Arrays.asList(initialc)));
		  m_Distrib.add(initiald);			  
	  }  
  }
  
  public Instance applyDiscretization(Instance inst) {
	  return convertInstance(inst);
  }
  
  
  public void updateEvaluator(Instance instance) {
	  
	  if(m_CutPointsL1 == null) {
		  initializeL1(instance);
	  }
		  
	    for (int i = instance.numAttributes() - 1; i >= 0; i--) {
	      if ((m_DiscretizeCols.isInRange(i))
	        && (instance.attribute(i).isNumeric())
	        && (instance.classIndex() != i)) {
	    	  updateLayer1(instance, i);
	    	  
	      }
	    }
	    
	    if(totalCount > 0 && totalCount % l2UpdateExamples == 0){
  		  updateLayer2(instance);
    	  System.out.println("New layer 2\n\n");
	  		for (int i = instance.numAttributes() - 1; i >= 0; i--) {
	  	      if ((m_DiscretizeCols.isInRange(i))
	  	        && (instance.attribute(i).isNumeric())
	  	        && (instance.classIndex() != i)) {
	  	    	  //System.out.println(layer1toString(i));
	  	    	  System.out.println(layer2toString(i));
	  	    	  	    	  
	  	      }
	  	    }
  	  	}
  }
  
  private void updateLayer2(Instance instance) {
	// TODO Auto-generated method stub
	m_CutPointsL2 = new double[m_CutPointsL1.size()][];
    for (int i = instance.numAttributes() - 1; i >= 0; i--) {
      if ((m_DiscretizeCols.isInRange(i))
        && (instance.attribute(i).isNumeric())) {
    	  double[] attCutPoints = cutPointsForSubset(i, 0, m_CutPointsL1.get(i).size());
    	  if(attCutPoints != null) {
    		  m_CutPointsL2[i] = new double[attCutPoints.length];
    		  System.arraycopy(attCutPoints, 0, m_CutPointsL2[i], 0, attCutPoints.length);  
    	  } else {
    		  m_CutPointsL2[i] = new double[m_CutPointsL1.get(i).size()];
    		  for (int j = 0; j < m_CutPointsL1.get(i).size(); j++) {
    			  m_CutPointsL2[i][j] = m_CutPointsL1.get(i).get(j);
    		  }
    	  }    	  
      }
    }
	
  }

  private void updateLayer1(Instance inst, int index) {
	  int k = 0;
	  if (!inst.isMissing(index)) {		  
	        double x = inst.value(index);
        	if(x < m_CutPointsL1.get(index).get(0)){
        		k = 0;
        	} else if(x > m_CutPointsL1.get(index).get(m_CutPointsL1.get(index).size() - 1)) {
        		k = m_CutPointsL1.get(index).size() - 1;
        	} else {
        		k = (int) Math.ceil((x - m_CutPointsL1.get(index).get(0)) / step);
        		while(x < m_CutPointsL1.get(index).get(k - 1)) k -= 1;
    	        while(x > m_CutPointsL1.get(index).get(k)) k += 1;
    	        
        	}
        	m_Counts.get(index).set(k, m_Counts.get(index).get(k) + 1);
        	int nvalue = m_Distrib.get(index).get(k).getOrDefault((int) inst.classValue(), 0) + 1;
        	m_Distrib.get(index).get(k).put((int) inst.classValue(), nvalue);	        
	        totalCount += 1;
        	
        	// Launch the split process
	        if(totalCount > minExamples && ((double) m_Counts.get(index).get(k)) / totalCount > alpha) {
	        	System.out.println("Entra");
	        	int tmp = m_Counts.get(index).get(k) / 2;
	        	m_Counts.get(index).set(k, tmp);
	        	Map<Integer, Integer> classDist = m_Distrib.get(index).get(k); 
	        	Map<Integer, Integer> halfDistrib = new HashMap<Integer, Integer>();
			    for (Map.Entry<Integer, Integer> entry : classDist.entrySet()){
		    	  halfDistrib.put(entry.getKey(), entry.getValue() / 2);
			    }
			    m_Distrib.get(index).set(k, halfDistrib);
	        	
	        	if(k == 0) {
	        		m_CutPointsL1.get(index).add(0, m_CutPointsL1.get(index).get(0) - step);
	        		m_Counts.get(index).add(0, tmp);
	        		m_Distrib.get(index).add(0, new HashMap<Integer,Integer>(halfDistrib));
	        	} else if(k > m_CutPointsL1.get(index).size()) {
	        		m_CutPointsL1.get(index).add(m_CutPointsL1.get(index).get(m_CutPointsL1.get(index).size() - 1) + step);
	        		m_Counts.get(index).add(tmp);
	        		m_Distrib.get(index).add(new HashMap<Integer,Integer>(halfDistrib));
	        	} else {
	        		double nBreak = m_CutPointsL1.get(index).get(k) + m_CutPointsL1.get(index).get(k + 1) / 2;
	        		m_CutPointsL1.get(index).add(k, nBreak); // Important to use add function
	        		m_Counts.get(index).add(k, tmp);
	        		m_Distrib.get(index).add(k, new HashMap<Integer,Integer>(halfDistrib));
	        	}
	        }
	        
      }
  }

  /**
   * Gets the bin ranges string for an attribute
   * 
   * @param attributeIndex the index (from 0) of the attribute to get the bin
   *          ranges string of
   * @return the bin ranges string (or null if the attribute requested has been
   *         discretized into only one interval.)
   */
  public String layer2toString(int attributeIndex) {

    if (m_CutPointsL2 == null) {
      return null;
    }

    double[] cutPoints = m_CutPointsL2[attributeIndex];

    if (cutPoints == null) {
      return "All";
    }

    StringBuilder sb = new StringBuilder();
    boolean first = true;

    for (int j = 0, n = cutPoints.length; j <= n; ++j) {
      if (first) {
        first = false;
      } else {
        sb.append(',');
      }

      sb.append(binRangeString(cutPoints, j, m_BinRangePrecision));
    }

    return sb.toString();
  }
  
  public String layer1toString(int attributeIndex) {

	    if (m_CutPointsL1 == null) {
	      return null;
	    }

	    double[] cutPoints = new double[m_CutPointsL1.get(attributeIndex).size()];
	    for (int i = 0; i < m_CutPointsL1.get(attributeIndex).size(); i++) {
			cutPoints[i] = m_CutPointsL1.get(attributeIndex).get(i);
		}

	    StringBuilder sb = new StringBuilder();
	    boolean first = true;

	    for (int j = 0, n = cutPoints.length; j <= n; ++j) {
	      if (first) {
	        first = false;
	      } else {
	        sb.append(',');
	      }

	      sb.append(binRangeString(cutPoints, j, m_BinRangePrecision));
	    }

	    return sb.toString();
	  }

  /**
   * Get a bin range string for a specified bin of some attribute's cut points.
   * 
   * @param cutPoints The attribute's cut points; never null.
   * @param j The bin number (zero based); never out of range.
   * @param precision the precision for the range values
   * 
   * @return The bin range string.
   */
  private static String binRangeString(double[] cutPoints, int j, int precision) {
    assert cutPoints != null;

    int n = cutPoints.length;
    assert 0 <= j && j <= n;

    return j == 0 ? "" + "(" + "-inf" + "-"
      + Utils.doubleToString(cutPoints[0], precision) + "]" : j == n ? "" + "("
      + Utils.doubleToString(cutPoints[n - 1], precision) + "-" + "inf" + ")"
      : "" + "(" + Utils.doubleToString(cutPoints[j - 1], precision) + "-"
        + Utils.doubleToString(cutPoints[j], precision) + "]";
  }

  /**
   * Test using Fayyad and Irani's MDL criterion.
   * 
   * @param priorCounts
   * @param bestCounts
   * @param numInstances
   * @param numCutPoints
   * @return true if the splits is acceptable
   */
  private boolean FayyadAndIranisMDL(double[] priorCounts,
    double[][] bestCounts, double numInstances, int numCutPoints) {

    double priorEntropy, entropy, gain;
    double entropyLeft, entropyRight, delta;
    int numClassesTotal, numClassesRight, numClassesLeft;

    // Compute entropy before split.
    priorEntropy = ContingencyTables.entropy(priorCounts);

    // Compute entropy after split.
    entropy = ContingencyTables.entropyConditionedOnRows(bestCounts);

    // Compute information gain.
    gain = priorEntropy - entropy;

    // Number of classes occuring in the set
    numClassesTotal = 0;
    for (double priorCount : priorCounts) {
      if (priorCount > 0) {
        numClassesTotal++;
      }
    }

    // Number of classes occuring in the left subset
    numClassesLeft = 0;
    for (int i = 0; i < bestCounts[0].length; i++) {
      if (bestCounts[0][i] > 0) {
        numClassesLeft++;
      }
    }

    // Number of classes occuring in the right subset
    numClassesRight = 0;
    for (int i = 0; i < bestCounts[1].length; i++) {
      if (bestCounts[1][i] > 0) {
        numClassesRight++;
      }
    }

    // Entropy of the left and the right subsets
    entropyLeft = ContingencyTables.entropy(bestCounts[0]);
    entropyRight = ContingencyTables.entropy(bestCounts[1]);

    // Compute terms for MDL formula
    delta = Utils.log2(Math.pow(3, numClassesTotal) - 2)
      - ((numClassesTotal * priorEntropy) - (numClassesRight * entropyRight) - (numClassesLeft * entropyLeft));

    // Check if split is to be accepted
    return (gain > (Utils.log2(numCutPoints) + delta) / numInstances);
  }
  
  private double[] cutPointsForSubset(int attIndex,
		    int first, int lastPlusOne) {

		    //Map<Integer, Double> counts, bestCounts;
		    double[] left, right, cutPoints;
		    //double step = ((float) totalCount) / m_CutPointsL1.get(index).size();
		    double currentCutPoint = -Double.MAX_VALUE, bestCutPoint = -1, currentEntropy, bestEntropy, priorEntropy, gain;
		    int bestIndex = -1, numCutPoints = 0;
		    double numInstances = 0;

		    // Compute number of instances in set
		    if ((lastPlusOne - first) < 2) {
		      return null;
		    }
		    
		    // Get the greatest class observed till here
		    int numClasses = 0;
		    for (int i = first; i < lastPlusOne; i++) {
		    	Map<Integer, Integer> classDist = m_Distrib.get(attIndex).get(i); 
		    	for (Integer key : classDist.keySet()){
		    		if(key > numClasses) {
		    			numClasses = key;
		    		}
	      		}
		    }
		    numClasses += 1;

		    // Compute class counts.
		    double[][] counts = new double[2][numClasses];
		    for (int i = first; i < lastPlusOne; i++) {
		    	Map<Integer, Integer> classDist = m_Distrib.get(attIndex).get(i); 
			    for (Map.Entry<Integer, Integer> entry : classDist.entrySet()){
		    	  counts[1][entry.getKey()] += entry.getValue();
		          numInstances += entry.getValue();
			    }
		    }
		    
		    // Save prior counts
		    double[] priorCounts = new double[numClasses];
		    System.arraycopy(counts[1], 0, priorCounts, 0, numClasses);

		    // Entropy of the full set
		    priorEntropy = ContingencyTables.entropy(priorCounts);
		    bestEntropy = priorEntropy;
		    
		    // Compute class counts.
		    /*counts = new HashMap<Integer, Double>();
		    for (int i = first; i < lastPlusOne; i++) {
		      
		      Map<Integer, Integer> classDist = m_Distrib.get(attIndex).get(i); 
		      for (Map.Entry<Integer, Integer> entry : classDist.entrySet()){
		    	  double c = counts.getOrDefault(entry.getKey(), 0.0) + entry.getValue();
		          counts.put(entry.getKey(), c);
		          numInstances += c;
		      }
		    }*/

		    // Entropy of the full set
		    
		    /*Double[] priorCounts = counts.values().toArray(new Double[counts.size()]);
		    double[] prior = new double[counts.size()];
		    for (int i = 0; i < counts.size(); i++) {
		    	prior[i] = priorCounts[i];
		    }*/
		    
		    priorEntropy = ContingencyTables.entropy(priorCounts);
		    bestEntropy = priorEntropy;

		    // Find best entropy.
		    double[][] bestCounts = new double[2][numClasses];
		    for (int i = first; i < (lastPlusOne - 1); i++) {
		    	Map<Integer, Integer> classDist = m_Distrib.get(attIndex).get(i); 
		    	for (Map.Entry<Integer, Integer> entry : classDist.entrySet()){
		    		counts[0][entry.getKey()] += entry.getValue();
		    		counts[1][entry.getKey()] -= entry.getValue();
		    	}		
		    	currentCutPoint = m_CutPointsL1.get(attIndex).get(i);
		    	currentEntropy = ContingencyTables.entropyConditionedOnRows(counts);
  		        if (currentEntropy < bestEntropy) {
  		          bestCutPoint = currentCutPoint;
  		          bestEntropy = currentEntropy;
  		          bestIndex = i;
  		          System.arraycopy(counts[0], 0, bestCounts[0], 0, numClasses);
  		          System.arraycopy(counts[1], 0, bestCounts[1], 0, numClasses);
  		        }
  		        numCutPoints++;
		    }

		    // Use worse encoding?
		    if (!m_UseBetterEncoding) {
		      numCutPoints = (lastPlusOne - first) - 1;
		    }

		    // Checks if gain is zero
		    gain = priorEntropy - bestEntropy;
		    
		    if (gain <= 0) {
		      return null;
		    }

		    // Check if split is to be accepted
		    if (FayyadAndIranisMDL(priorCounts, bestCounts,
		        numInstances, numCutPoints)) {

		      // Select split points for the left and right subsets
		      left = cutPointsForSubset(attIndex, first, bestIndex + 1);
		      right = cutPointsForSubset(attIndex, bestIndex + 1,
		        lastPlusOne);

		      // Merge cutpoints and return them
		      if ((left == null) && (right) == null) {
		        cutPoints = new double[1];
		        cutPoints[0] = bestCutPoint;
		      } else if (right == null) {
		        cutPoints = new double[left.length + 1];
		        System.arraycopy(left, 0, cutPoints, 0, left.length);
		        cutPoints[left.length] = bestCutPoint;
		      } else if (left == null) {
		        cutPoints = new double[1 + right.length];
		        cutPoints[0] = bestCutPoint;
		        System.arraycopy(right, 0, cutPoints, 1, right.length);
		      } else {
		        cutPoints = new double[left.length + right.length + 1];
		        System.arraycopy(left, 0, cutPoints, 0, left.length);
		        cutPoints[left.length] = bestCutPoint;
		        System.arraycopy(right, 0, cutPoints, left.length + 1, right.length);
		      }

		      return cutPoints;
		    } else {
		      return null;
		    }
  	}
  
  /**
   * Convert a single instance over. The converted instance is added to the end
   * of the output queue.
   * 
   * @param instance the instance to convert
   */
  protected Instance convertInstance(Instance instance) {

    int index = 0;
    double[] vals = new double[instance.numAttributes()];
    // Copy and convert the values
    for (int i = 0; i < instance.numAttributes(); i++) {
      if (m_DiscretizeCols.isInRange(i)
        && instance.attribute(i).isNumeric()) {
        int j;
        double currentVal = instance.value(i);
        if (m_CutPointsL1.get(i) == null) {
          if (instance.isMissing(i)) {
            vals[index] = Utils.missingValue();
            instance.setValue(index, Utils.missingValue());
          } else {
            vals[index] = 0;
            instance.setValue(index, 0);
          }
          index++;
        } else {
          if (!m_MakeBinary) {
            if (instance.isMissing(i)) {
              vals[index] = Utils.missingValue();
              instance.setValue(index, Utils.missingValue());
            } else {
              for (j = 0; j < m_CutPointsL1.get(i).size(); j++) {
                if (currentVal <= m_CutPointsL1.get(i).get(j)) {
                  break;
                }
              }
              vals[index] = j;
              instance.setValue(index, j);
            }
            index++;
          } else {
            for (j = 0; j < m_CutPointsL1.get(i).size(); j++) {
              if (instance.isMissing(i)) {
                vals[index] = Utils.missingValue();
                instance.setValue(index, Utils.missingValue());
              } else if (currentVal <= m_CutPointsL1.get(i).get(j)) {
                vals[index] = 0;
                instance.setValue(index, 0);
              } else {
                vals[index] = 1;
                instance.setValue(index, 1);
              }
              index++;
            }
          }
        }
      } else {
        vals[index] = instance.value(i);
        index++;
      }
    }

    /*Instance inst = null;
    if (instance instanceof SparseInstance) {
      inst = new SparseInstance(instance.weight(), vals);
    } else {
      inst = new DenseInstance(instance.weight(), vals);
    }

    copyValues(inst, false, instance.dataset(), outputFormatPeek());
    */
    //push(inst); // No need to copy instance
    
    return(instance);
  }

  /**
   * Returns the revision string.
   * 
   * @return the revision
   */
  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 12037 $");
  }

  /**
   * Main method for testing this class.
   * 
   * @param argv should contain arguments to the filter: use -h for help
   */
  public static void main(String[] argv) {
    runFilter(new PIDdiscretize(), argv);
  }


}
