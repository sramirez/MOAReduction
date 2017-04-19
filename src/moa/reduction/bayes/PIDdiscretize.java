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
import weka.core.ContingencyTables;

import com.yahoo.labs.samoa.instances.Instance;

/**
 <!-- globalinfo-start -->
 * Partition Incremental Discretization (PiD)
 * <br/>
 * For more information, see:<br/>
 * <br/>
 * João Gama and Carlos Pinto. 2006. Discretization from data streams: applications to histograms and data mining. 
 * In Proceedings of the 2006 ACM symposium on Applied computing (SAC '06). ACM, New York, NY, USA, 662-667. DOI=http://dx.doi.org/10.1145/1141277.1141429
 * <br/>
 * @author Sergio Ramírez (sramirez at decsai dot ugr dot es)
 */
public class PIDdiscretize extends MOADiscretize {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	/** Parameters */	  
	protected int l2UpdateExamples = 1000;
	/** The number of bins to divide the attribute into */
	protected float alpha = 0.75f;	

	protected double min = 0;
		
	protected double max = 1;
	
	protected int initialBinsL1 = 200;
  
	/** Instance limit before starting the splitting process */
	protected int initialElements = 100;

  protected int totalCount = 0;

  /** Store the current cutpoints */  
  protected List<List<Double>> m_CutPointsL1 = null;
  
  protected List<List<Float>> m_Counts = null;
  
  protected List<List<Map<Integer, Float>>> m_Distrib = null;
  protected List<List<Map<Integer, Float>>> m_Distrib2 = null;
  
  protected double step;
  
  /** Constructor - initializes the filter */
  public PIDdiscretize() {
	  setAttributeIndices("first-last");
  }
  
  public PIDdiscretize(int initialElements, int initialBinsL1, int min, int max, int alpha, int l2UpdateExamples) {
	  super();
	  this.initialElements = initialElements;
	  this.initialBinsL1 = initialBinsL1;
	  this.min = min;
	  this.max = max;
	  this.alpha = alpha;
	  this.l2UpdateExamples = l2UpdateExamples;
  }

  public Instance applyDiscretization(Instance inst) {
	  if(m_CutPoints != null)
		  return convertInstance(inst);
	  return inst;
  }
  
  
  public void updateEvaluator(Instance instance) {	  
	  if(m_CutPoints == null) {
		  initializeLayers(instance);
	  }
	  
	  totalCount++;
		  
	  for (int i = instance.numAttributes() - 1; i >= 0; i--) {
		  if ((m_DiscretizeCols.isInRange(i))
				  && (instance.attribute(i).isNumeric())
				  && (instance.classIndex() != i)) {
			  updateLayer1(instance, i);
    	  
		  }
	  }
    
	  if(totalCount > 0 && totalCount % l2UpdateExamples == 0){
		  updateLayer2(instance);
		  m_Init = true;
	  }

	 if(totalCount % 101 == 0) 
		 writeCPointsToFile(1, 2, totalCount, "PiD");
  }
  
  private void initializeLayers(Instance inst){
	  m_DiscretizeCols.setUpper(inst.numAttributes() - 1);
	  step = (max - min) / (double) initialBinsL1;
	  m_CutPointsL1 = new ArrayList<List<Double>>(inst.numAttributes());
	  m_Counts = new ArrayList<List<Float>>(inst.numAttributes());
	  m_Distrib = new ArrayList<List<Map<Integer, Float>>>(inst.numAttributes());
	  m_Distrib2 = new ArrayList<List<Map<Integer, Float>>>(inst.numAttributes());
	  
	  for (int i = 0; i < inst.numAttributes(); i++) {
		  Double[] initialb = new Double[initialBinsL1 + 1];
		  Float[] initialc = new Float[initialBinsL1 + 1];
		  List<Map<Integer, Float>> initiald = new ArrayList<Map<Integer, Float>>(initialBinsL1 + 1);
		  
		  for (int j = 0; j < initialBinsL1 + 1; j++) {
			initialb[j] = min + j * step;
			initialc[j] = 0.f;
			initiald.add(new HashMap<Integer, Float>());
		  }
		  m_CutPointsL1.add(new ArrayList<Double>(Arrays.asList(initialb)));
		  m_Counts.add(new ArrayList<Float>(Arrays.asList(initialc)));
		  m_Distrib.add(initiald);
	  }  
	  initL2FromL1();
  }
  
  private void updateLayer2(Instance instance) {
	// TODO Auto-generated method stub
	m_CutPoints = new double[m_CutPointsL1.size()][];
    for (int i = instance.numAttributes() - 1; i >= 0; i--) {
      if ((m_DiscretizeCols.isInRange(i))
        && (instance.attribute(i).isNumeric())) {
    	  double[] attCutPoints = cutPointsForSubset(i, 0, m_CutPointsL1.get(i).size());
    	  if(attCutPoints != null) {
    		  m_CutPoints[i] = new double[attCutPoints.length];
    		  System.arraycopy(attCutPoints, 0, m_CutPoints[i], 0, attCutPoints.length);  
    	  } else {
    		  m_CutPoints[i] = new double[m_CutPointsL1.get(i).size()];
    		  for (int j = 0; j < m_CutPointsL1.get(i).size(); j++) {
    			  m_CutPoints[i][j] = m_CutPointsL1.get(i).get(j);
    		  }
    	  } 
    	  updateDistributionsL2(i, m_CutPoints[i]);
      }
    }	
  }
  
  
  private void updateDistributionsL2(int att, double[] newpoints) {
	  boolean cont = true;
	  List<Map<Integer, Float>> intervals = new ArrayList<Map<Integer, Float>>(newpoints.length + 1);
	  Map<Integer, Float> interv = new HashMap<Integer, Float>();
	  for (int i = 0, j = 0; i < m_CutPointsL1.get(att).size() && cont; i++) {
		  double cp = m_CutPointsL1.get(att).get(i);  
		  while(newpoints[j] <= cp && cont){
			  Map<Integer, Float> aux = m_Distrib.get(att).get(i);			  
			  // Aggregate by sum the maps
			  for(Integer key : aux.keySet()) {
				    if(interv.containsKey(key)) {
				        interv.put(key, aux.get(key) + interv.get(key));
				    } else {
				        interv.put(key, aux.get(key));
				    }
			  }
			  j++;
			  if(j >= newpoints.length){
				  cont = false; // get out
				  break;
			  }
		  }
		  intervals.add(interv);
		  interv = new HashMap<Integer, Float>();			
	  }	
	  m_Distrib2.set(att, intervals);
  }
  
  private void initL2FromL1() {
	  m_CutPoints = new double[m_CutPointsL1.size()][];
	  for (int i = 0; i < m_CutPointsL1.size(); i++) {
		  m_CutPoints[i] = new double[m_CutPointsL1.get(i).size()];
		  for (int j = 0; j < m_CutPointsL1.get(i).size(); j++) {
			  m_CutPoints[i][j] = m_CutPointsL1.get(i).get(j);
		  }
	  }	  
  }

  private void updateLayer1(Instance inst, int index) {
	  int k = 0;
	  if (!inst.isMissing(index)) {		  
        double x = inst.value(index);
    	if(x <= m_CutPointsL1.get(index).get(0)){
    		k = 0;
    	} else if(x > m_CutPointsL1.get(index).get(m_CutPointsL1.get(index).size() - 1)) {
    		k = m_CutPointsL1.get(index).size() - 1;
    	} else {
    		k = (int) Math.ceil((x - m_CutPointsL1.get(index).get(0)) / step);
    		while(x <= m_CutPointsL1.get(index).get(k - 1)) k -= 1;
	        while(x > m_CutPointsL1.get(index).get(k)) k += 1;    	        
    	}
    	m_Counts.get(index).set(k, m_Counts.get(index).get(k) + 1);
    	float nvalue = m_Distrib.get(index).get(k).getOrDefault((int) inst.classValue(), 0.f) + 1;
    	m_Distrib.get(index).get(k).put((int) inst.classValue(), nvalue);
    	
    	// Launch the split process
        double prop = ((double) m_Counts.get(index).get(k)) / totalCount;
        if(totalCount > initialElements && prop > alpha) {
        	float tmp = m_Counts.get(index).get(k) / 2;
        	m_Counts.get(index).set(k, tmp);
        	Map<Integer, Float> classDist = m_Distrib.get(index).get(k); 
        	Map<Integer, Float> halfDistrib = new HashMap<Integer, Float>();
		    for (Map.Entry<Integer, Float> entry : classDist.entrySet()){
	    	  halfDistrib.put(entry.getKey(), entry.getValue() / 2);
		    }
		    m_Distrib.get(index).set(k, halfDistrib);
        	
        	if(k == 0) {
        		m_CutPointsL1.get(index).add(0, m_CutPointsL1.get(index).get(0) - step);
        		m_Counts.get(index).add(0, tmp);
        		m_Distrib.get(index).add(0, new HashMap<Integer,Float>(halfDistrib));
        	} else if(k >= m_CutPointsL1.get(index).size() - 1) {
        		m_CutPointsL1.get(index).add(m_CutPointsL1.get(index).get(m_CutPointsL1.get(index).size() - 1) + step);
        		m_Counts.get(index).add(tmp);
        		m_Distrib.get(index).add(new HashMap<Integer,Float>(halfDistrib));
        	} else {
        		double nBreak = (m_CutPointsL1.get(index).get(k) + m_CutPointsL1.get(index).get(k + 1)) / 2;
        		m_CutPointsL1.get(index).add(k + 1, nBreak); // Important to use add function
        		m_Counts.get(index).add(k + 1, tmp);
        		m_Distrib.get(index).add(k, new HashMap<Integer,Float>(halfDistrib));
        	}
        }	        
      }
  }

  private double[] cutPointsForSubset(int attIndex, int first, int lastPlusOne) {

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
	    	Map<Integer, Float> classDist = m_Distrib.get(attIndex).get(i); 
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
	    	Map<Integer, Float> classDist = m_Distrib.get(attIndex).get(i); 
		    for (Map.Entry<Integer, Float> entry : classDist.entrySet()){
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
	    
	    priorEntropy = ContingencyTables.entropy(priorCounts);
	    bestEntropy = priorEntropy;

	    // Find best entropy.
	    double[][] bestCounts = new double[2][numClasses];
	    for (int i = first; i < (lastPlusOne - 1); i++) {
	    	Map<Integer, Float> classDist = m_Distrib.get(attIndex).get(i); 
	    	for (Map.Entry<Integer, Float> entry : classDist.entrySet()){
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

	      // Merge cut points and return them
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


  @Override
  public float jointProbValueClass(int attI, double rVal, int dVal, int classVal) {
		// TODO Auto-generated method stub
		int c = getAttValGivenClass(attI, rVal, dVal, classVal);
		return c / (float) totalCount;
  }
  
  @Override
  public int getAttValGivenClass(int attI, double rVal, int dVal, int classVal) {
	  return m_Distrib2.get(attI).get(dVal).get(classVal).intValue();		
  }
}
