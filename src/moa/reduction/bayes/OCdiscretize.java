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
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.TreeMap;

import com.yahoo.labs.samoa.instances.Instance;

import moa.reduction.core.MOADiscretize;
import weka.core.Range;
import weka.core.Utils;

/**
 * Online ChiMerge Algorith<br/>
 * <br/>
 * For more information, see:<br/>
 * <br/>
 * https://link.springer.com/chapter/10.1007%2F978-3-642-23241-1_10<br/>
 * <p/>
 * 
 * @author Sergio Ramírez (sramirez at decsai dot ugr dot es)
 */
public class OCdiscretize implements MOADiscretize{


  /** Stores which columns to Discretize */
  protected Range m_DiscretizeCols;

  /** The number of bins to divide the attribute into */
  protected float totalCount;
  
  protected ArrayList<TreeMap<Double, Bin>> trees;  
  protected List<LinkedList<Double>> it_bin;
  protected List<Bin> previous_bin;
  protected List<Interval> last_interval;  
  protected List<PriorityQueue<Interval>> interval_q;
  protected List<List<Interval>> interval_l;
  protected List<List<Interval>> interval_l2;
  protected List<LinkedList<Pair>> example_q;
  protected int[] phases;
  protected int initialElements = 100;
  protected int numClasses = 2;

  /** Output binary attributes for discretized attributes. */
  protected boolean m_MakeBinary = false;

  /** Constructor - initializes the filter */
  public OCdiscretize(int nc) {
	  this.numClasses = nc;
	  m_DiscretizeCols = new Range();	  
	  totalCount = 0;
	  trees = null;
	  setAttributeIndices("first-last");
  }
  
  public OCdiscretize(int numClasses, int[] attributes) {
	  this(numClasses);
	  setAttributeIndicesArray(attributes);
  }
  
  public OCdiscretize(int numClasses, int initial) {
	  this(numClasses);
	  this.initialElements = initial;
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

    if (interval_l2 == null) {
      return null;
    }
    
    double[] cutpoints = new double[interval_l2.get(attributeIndex).size()];
    for (int i = 0; i < cutpoints.length; i++) {
		cutpoints[i] = interval_l2.get(attributeIndex).get(i).lower;
	}
    return cutpoints;
  }
  

  
  public Instance applyDiscretization(Instance inst) {
	  if(trees != null)
		  return convertInstance(inst);
	  return inst;
  } 
  
  public void updateEvaluator(Instance instance) {
	  
	  if(trees == null) {
		  initialize(instance);
	  }
		  
	  totalCount++;
	  	
	  for (int i = instance.numAttributes() - 1; i >= 0; i--) {
		  if ((m_DiscretizeCols.isInRange(i))
			  && (instance.attribute(i).isNumeric())
			  && (instance.classIndex() != i)) {
			  	if (!instance.isMissing(i)) {		  
			  		OnlineChiMerge(i, instance); 
			  	}  
		  }
	  }
  }

  private void initialize(Instance inst){
	  
	  m_DiscretizeCols.setUpper(inst.numAttributes() - 1);	  
	  trees = new ArrayList<TreeMap<Double, Bin>>(inst.numAttributes());
	  interval_q = new ArrayList<PriorityQueue<Interval>>();
	  interval_l = new ArrayList<List<Interval>>();
	  interval_l2 = new ArrayList<List<Interval>>();
	  example_q = new ArrayList<LinkedList<Pair>>();
	  it_bin = new ArrayList<LinkedList<Double>>();
	  previous_bin = new ArrayList<Bin>();
	  last_interval = new ArrayList<Interval>();
	  
	  phases = new int[inst.numAttributes()];
	  for (int i = 0; i < inst.numAttributes() - 1; i++) {
		  trees.add(new TreeMap<Double, Bin>());
		  phases[i] = 0;
		  interval_q.add(new PriorityQueue<Interval>());
		  interval_l.add(new ArrayList<Interval>());
		  interval_l2.add(new ArrayList<Interval>());
		  example_q.add(new LinkedList<Pair>());
		  it_bin.add(new LinkedList<Double>());
		  previous_bin.add(new Bin());
		  last_interval.add(new Interval());		  
	  }  
  }
  
  private void OnlineChiMerge(int index, Instance inst){
	  
	  double value = inst.value(index);
	  if(phases[index] != 1) addToMainTree(index, value, inst.classValue());		
	  
	  if(phases[index] == 0){		    
		  if(totalCount >= initialElements) reInit(index); // go to next phase
	  } else if (phases[index] == 1) {		  
		  example_q.get(index).add(new Pair(value, inst.classValue()));
		  if(!it_bin.get(index).isEmpty()) {
			  double key = it_bin.get(index).pollFirst();
			  Bin cbin = trees.get(index).get(key);
			  if(interval_l.get(index).isEmpty()){
				  double minus_inf = Float.NEGATIVE_INFINITY;
				  last_interval.set(index, new Interval(minus_inf, cbin.distrib));
				  interval_l.get(index).add(last_interval.get(index));				  
			  } else {
				  double bound = (previous_bin.get(index).value + cbin.value) / 2;
				  Interval new_interval = new Interval(bound, cbin.distrib);
				  interval_l.get(index).add(new_interval);
				  last_interval.get(index).qval = computeQ(last_interval.get(index), new_interval);
				  
				  interval_q.get(index).offer(last_interval.get(index));
				  last_interval.set(index, new_interval);				  
			  }
			  previous_bin.set(index, cbin);
		  } else {
			  phases[index] = 2;
		  }
	  } else if (phases[index] == 2) {		  
		  Pair pel = example_q.get(index).pollLast();		  
		  addToMainTree(index, pel.value, pel.clas);
		  
		  Interval[] tmp = new Interval[interval_q.get(index).size()];
		  Interval[] array = interval_q.get(index).toArray(tmp);

		  Interval next = array[0]; 
		  interval_q.get(index).remove(next);			  
		  interval_l.get(index).remove(next);

		  if(array.length > 1) {
			  Interval best = array[1];
			  best.merge(next);		  
			  
			  if(interval_l.get(index).size() > 2){ // There are more elements
				  next = array[2];
				  best.qval = computeQ(best, next);
				  interval_q.get(index).remove(best);
				  interval_q.get(index).offer(best);
			  }
			  
			  if(!best.equals(interval_l.get(index).get(0))) {
				  Interval prev = array[array.length - 1];
				  prev.qval = computeQ(prev, best);

				  interval_q.get(index).remove(prev);
				  interval_q.get(index).offer(prev);
			  }			  
		  } else {		  
			  if(interval_q.get(index).isEmpty()){
				  Pair p = example_q.get(index).pollLast();
				  addToMainTree(index, p.value, p.clas);
				  phases[index] = 3;
			  }		  
		  }
	  } else { // phase = 3
		  if(interval_l.get(index).size() > 0) {
			  Interval e = interval_l.get(index).remove(0);
			  interval_l2.get(index).add(e);
			  if(example_q.get(index).isEmpty()){
				  reInit(index);
			  }
		  }
	  }	
  }
  
  private void addToMainTree(int index, double value, double clas){
	  Bin bin = trees.get(index).getOrDefault(value, new Bin(value));
	  bin.feed((int) clas);
	  trees.get(index).put(value, bin);
	  it_bin.get(index).add(value);
  }
  
  private void reInit(int index){	  
	  it_bin.get(index).addAll(index, trees.get(index).navigableKeySet());
	  previous_bin.set(index, trees.get(index).lastEntry().getValue());
	  int tam = trees.get(index).size() - 1;
	  if(tam < 1) tam = 1; 
	  interval_q.set(index, new PriorityQueue<Interval>(tam));
	  phases[index] = 1;
  }
  
  
  private float computeQ(Interval int1, Interval int2){
	// Total number of examples in the two intervals
      int N = int1.n + int2.n;

      // The result
      float qval = 0;

      // Sum over all the classes
      for(int j = 0; j < int1.distrib.length; ++j)
      {
          int Cj = int1.distrib[j] + int2.distrib[j];
          float Eij, d;

          // interval 1
          Eij = int1.n * Cj / (float) N;
          if(Eij == 0) {
        	  qval += Float.POSITIVE_INFINITY;
          } else {
              d = (int1.distrib[j] - Eij);
              qval += d*d/Eij;
          }

          // interval 2
          Eij = int1.n * Cj / (float) N;
          if(Eij == 0)
              qval += Float.POSITIVE_INFINITY;
          else {
              d = (int2.distrib[j] - Eij);
              qval += d*d/Eij;
          }
      }

      return qval;
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
        if (interval_l2.get(i) == null) {
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
              for (j = 0; j < interval_l2.get(i).size(); j++) {
                if (currentVal <= interval_l2.get(i).get(j).lower) {
                  break;
                }
              }
              vals[index] = j;
              instance.setValue(index, j);
            }
            index++;
          } else {
            for (j = 0; j < interval_l2.get(i).size(); j++) {
              if (instance.isMissing(i)) {
                vals[index] = Utils.missingValue();
                instance.setValue(index, Utils.missingValue());
              } else if (currentVal <= interval_l2.get(i).get(j).lower) {
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

  private class Bin {

	    final double value;
	    int[] distrib;

	    public Bin(){
	    	this(Double.NEGATIVE_INFINITY);
	    }
	    
	    public Bin(double value) {
	    	this.value = value;
	    	distrib = new int[numClasses];
	    	for(int i = 0; i < numClasses; ++i)
                distrib[i] = 0;
	    }
	    
	    public void feed(int label) {
	    	distrib[label] += 1;
	    }
  }
  
  private class Interval implements Comparable<Interval> {

	    double lower;
	    int[] distrib;
	    int n;
	    float qval;
	    
	    public Interval(){
	    	this(Double.NEGATIVE_INFINITY, new int[1]);
	    }

	    public Interval(double lower, int[] distr) {
	    	this.lower = lower;
	    	distrib = new int[distr.length];
	    	n = 0;
	    	for(int i = 0; i < distr.length; ++i){
	    		distrib[i] = distr[i];
		    	n += distr[i];
	    	}
	    	qval = Float.POSITIVE_INFINITY;              
	    }
	    
	    public void merge(Interval other){
	    	 for(int i = 0; i < distrib.length; ++i)
	                distrib[i] += other.distrib[i];
	    	 n += other.n;
	    	 lower = Math.min(lower, other.lower);
	    }
		
		public boolean equals(final Interval other) {
			  return (this.lower == other.lower);
		}

		@Override
		public int compareTo(Interval o) {
			// TODO Auto-generated method stub
			return Float.compare(this.qval, o.qval);
		}
  }
  
  
  private class Pair {

	    double value;
	    double clas;

	    public Pair(double value, double clas) {
	        this.value = value;
	        this.clas = clas;
	    }	    

	    @Override
	    public boolean equals(Object o) {
	        if (this == o) return true;
	        if (!(o instanceof Pair)) return false;
	        Pair key = (Pair) o;
	        return value == key.value && clas == key.clas;
	    }

  }
  
	@Override
	public int getNumberIntervals() {
		// TODO Auto-generated method stub
		if(interval_l2 != null) {
			int ni = 0;
			for(List<Interval> cp: interval_l2){
				if(cp != null)
					ni += cp.size();
			}
			return ni;	
		}
		return 0;
	}
  
}
