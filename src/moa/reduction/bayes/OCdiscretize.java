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
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.TreeMap;
import java.util.TreeSet;

import moa.reduction.core.MOADiscretize;
import weka.core.Range;

import com.yahoo.labs.samoa.instances.Instance;

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
public class OCdiscretize extends MOADiscretize{


  /**
	 * 
	 */
	private static final long serialVersionUID = 1L;

  /** The number of bins to divide the attribute into */
  protected int totalCount;
  
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
  private boolean created = false;

  /** Output binary attributes for discretized attributes. */
  protected boolean m_MakeBinary = false;

  /** Constructor - initializes the filter */
  public OCdiscretize(int nc) {
	  super();
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

  
  public Instance applyDiscretization(Instance inst) {
	  if(created)
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
			  		if(!interval_l2.get(i).isEmpty())  {
			  			created = true;
			  			// Transform intervals in list to a matrix
				  		List<Interval> l = interval_l2.get(i);
				  		TreeSet<Double> ss = new TreeSet<Double>();
				  		for(Interval inter: l) ss.add(inter.lower);
		  				m_CutPoints[i] = new double[ss.size()];
		  				int j = 0;
				  		for(Double lower: ss) m_CutPoints[i][j++] = lower;
			  		}			  		
			  	}  
		  }
	  }
	  

	 if(totalCount % 101 == 0) 
		 writeCPointsToFile(1, 2, totalCount, "OC");
  }

  private void initialize(Instance inst){
	  
	  m_DiscretizeCols.setUpper(inst.numAttributes() - 1);	  
	  trees = new ArrayList<TreeMap<Double, Bin>>(inst.numAttributes());
	  interval_q = new ArrayList<PriorityQueue<Interval>>();
	  interval_l = new ArrayList<List<Interval>>();
	  interval_l2 = new ArrayList<List<Interval>>();
	  example_q = new ArrayList<LinkedList<Pair>>();	  
	  it_bin = new ArrayList<LinkedList<Double>>();
	  m_CutPoints = new double[inst.numAttributes()][];	
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

		  if(array.length > 1) {
			  Interval best = array[0]; 
			  Interval next = array[1];
			  interval_q.get(index).remove(next);			  
			  interval_l.get(index).remove(next);
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
			  } else {
				  // Only one element left
				  Interval best = array[0]; 
				  interval_q.get(index).remove(best);			  
				  interval_l.get(index).remove(best);
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
	  double lastKey = it_bin.get(index).peekLast();
	  previous_bin.set(index, trees.get(index).get(lastKey));
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
  
}
