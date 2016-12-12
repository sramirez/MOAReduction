package moa.reduction.bayes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import java.util.Map.Entry;
import java.util.NavigableSet;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;

import moa.reduction.core.MOADiscretize;

import com.google.common.collect.Multimap;
import com.google.common.collect.TreeMultimap;
import com.yahoo.labs.samoa.instances.Instance;

public class REBdiscretize extends MOADiscretize {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private int totalCount, isample = 0, numClasses, numAttributes;
	TreeMap<Float, Interval>[] allIntervals;
	Instance[] sample;
	boolean init = false;
	private long seed = 317901561;
	private float lambda, alpha, globalCrit = 0f, globalDiff = 0f, errorRate = 0.25f;
	private static int MAX_OLD = 5;
	private static float ERR_TH = 0.25f;
	private Random rand;
	
	public REBdiscretize() {
		// TODO Auto-generated constructor stub
		setAttributeIndices("first-last");
		this.rand = new Random(seed);
		this.alpha = 0.5f;
		this.lambda = 0.5f;
		this.sample = new Instance[1000];
	}
	
	public REBdiscretize(float alpha, float lambda, int sampleSize) {
		// TODO Auto-generated constructor stub
		this();
		this.alpha = alpha;
		this.lambda = lambda;
		this.sample = new Instance[sampleSize];
	}	


	public Instance applyDiscretization(Instance inst) {
		  if(init){
			  for (int i = 0; i < inst.numAttributes(); i++) {
				 // if numeric and not missing, discretize
				 if(inst.attribute(i).isNumeric() && !inst.isMissing(i)) {
					 double[] boundaries = new double[allIntervals[i].size()];
					 int j = 0;
					 	for (Iterator iterator = allIntervals[i].values().iterator(); iterator
					 			.hasNext();) {
					 		boundaries[j++] = ((Interval) iterator.next()).end;
					 	}
					 	m_CutPoints[i] = boundaries;
				 }
			  }
			  return convertInstance(inst);
		  }		  
		  return inst;
	}
  
  public void updateEvaluator(Instance instance, float errorRate) {	  
	  this.errorRate = errorRate;
	  updateEvaluator(instance);
  }
  
  public void updateEvaluator(Instance instance) {	  
	  if(m_CutPoints == null) {
		  initializeLayers(instance);
	  }
	  
	  totalCount++;
    
	  if(totalCount > sample.length){
		  if(init) {
			  boolean updated = updateSampleByError(instance);
		  } else {
			  batchFusinter();
			  for (int i = 0; i < allIntervals.length; i++) {
				  printIntervals(i, allIntervals[i].values());
			  }
			  init = true;
		  }
	  } else {
		  sample[totalCount - 1] = instance;
	  }
  }
  
  private boolean updateSampleByError(Instance instance){
	  boolean updated = false;
	  Instance replacedInstance = null;
	  if(rand.nextFloat() < errorRate + ERR_TH){
		 int pos = isample % sample.length;
		 replacedInstance = sample[pos];
		 sample[pos] = instance; 
		 updated = true;
	  } 
	  isample++;
	  if(updated) {
		  for (int i = 0; i < instance.numAttributes(); i++) {
			 // if numeric and not missing, discretize
			 if(instance.attribute(i).isNumeric() && !instance.isMissing(i)) {
				 float val = (float) instance.value(i);
				 Map.Entry<Float, Interval> entry = allIntervals[i].ceilingEntry(val);
				 Interval interv = entry.getValue();
				 Float prevKey = ((NavigableSet<Float>) interv.histogram.keySet()).floor(val);
				 if(prevKey == null)
					 
				 if(prevKey != val){
					 interv.histogram.get(prevKey);
				 }
				 Collection<Integer> vcls = interv.histogram.get((float) instance.value(i));				 
			 }
		  }
		  
	  }
	  return updated;
  }
  
  private void printIntervals(int att, Collection<Interval> intervals){
	  System.out.println("Atributo: " + att);
	  for (Iterator<Interval> iterator = intervals.iterator(); iterator.hasNext();) {
		Interval interval = (Interval) iterator.next();
		System.out.println(interval.label + "-" + interval.end + ",");
	  }
	  
  }
  
  private void batchFusinter() {
	  // TODO Auto-generated method stub
	  float[][] sorted = new float[numAttributes][sample.length];
	  
	  for (int i = numAttributes - 1; i >= 0; i--) {
		  if ((m_DiscretizeCols.isInRange(i))
				  && (sample[0].attribute(i).isNumeric())
				  && (sample[0].classIndex() != i)) {
			  Integer[] idx = new Integer[sample.length];
			  for (int j = 0; j < sample.length; j++) {
			  //updateLayer1(instance, i);
				  idx[j] = j;
				  sorted[i][j] = (float) sample[j].value(i);
			  }
			  final float[] data = sorted[i];
			  
			  Arrays.sort(idx, new Comparator<Integer>() {
			      @Override public int compare(final Integer o1, final Integer o2) {
			          return Float.compare(data[o1], data[o2]);
			      }
			  });
			  
			  allIntervals[i] = initIntervals(i, idx);
			  printIntervals(i, allIntervals[i].values());
			  fusinter(i, allIntervals[i]);
			  printIntervals(i, allIntervals[i].values());
		  }
	  }
  }
  
  private void fusinter(int att, TreeMap <Float, Interval> intervals) {
	int posMin = 0;
	globalDiff = Float.MIN_VALUE;
	globalCrit = 0f;
	float newMaxCrit = 0;
	ArrayList<Interval> intervalList = new ArrayList<Interval>(intervals.values());
	
	/* Initialize criterion values and the global threshold */
	for (Iterator<Interval> iterator = intervalList.iterator(); iterator.hasNext();) {
		Interval interval = (Interval) iterator.next();
		interval.setCrit(evalInterval(interval.cd)); // Important 
		globalCrit += interval.crit;		
	}
	
	while(intervalList.size() > 1) {
		globalDiff = 0;
		for(int i = 0; i < intervalList.size() - 1; i++) {
			float newCrit = evaluteMerge(globalCrit, intervalList.get(i), intervalList.get(i+1));
			float difference = globalCrit - newCrit;
			if(difference > globalDiff){
				posMin = i;
				globalDiff = difference;
				newMaxCrit = newCrit;
			}
		}
	
		if(globalDiff > 0) {
			globalCrit = newMaxCrit;
			Interval int1 = intervalList.get(posMin);
			Interval int2 = intervalList.remove(posMin+1);
			int1.mergeIntervals(int2);
		} else {
			break;
		}
	}
	allIntervals[att] = new TreeMap<Float, Interval>();
	for (int i = 0; i < intervalList.size(); i++) {
		allIntervals[att].put(intervalList.get(i).end, intervalList.get(i));
	}
	
  }
  
  // EvalIntervals must be executed before calling this method */
  private float evaluteMerge(float currentCrit, Interval int1, Interval int2) {
	  int[] cds = int1.cd.clone();
	  for (int i = 0; i < cds.length; i++) {
		cds[i] += int2.cd[i];
	  }
	  return currentCrit - int1.crit - int2.crit + evalInterval(cds);
  }
  

	
	private float evalInterval(int cd[]) {
		int Nj = 0;
		float suma, factor;
		for (int j = 0; j < numClasses; j++) {
			Nj += cd[j];
		}
		suma = 0;
		for (int j = 0; j < numClasses; j++) {
			factor = (cd[j] + lambda) / (Nj + numClasses * lambda);
			suma += factor * (1 - factor);
		}
		float crit = (alpha * ((float) Nj / totalCount) * suma) + ((1 - alpha) * (((float) numClasses * lambda) / Nj));
		return crit;
	}
  
  private TreeMap <Float, Interval> initIntervals(int att, Integer[] idx) {
		TreeMap <Float, Interval> intervals = new TreeMap<Float, Interval> ();
		float valueAnt = (float) sample[idx[0]].value(att);
		int classAnt = (int) sample[idx[0]].classValue();
		Interval lastInterval =  new Interval(1, valueAnt, classAnt);
		intervals.put(valueAnt, lastInterval);
	
		for(int i = 1; i < sample.length;i++) {
			float val = (float) sample[idx[i]].value(att);
			int clas = (int) sample[idx[i]].classValue();
			if(i == 416 && att == 3) {
				System.out.println("Hola");
			}
			if(val != valueAnt && clas != classAnt) {
				lastInterval = new Interval(i + 1, val, clas);
				intervals.put(val, lastInterval);
				valueAnt = val;
				classAnt = clas;
			} else {
				lastInterval.addPoint(val, clas);
			}
		}
		return intervals;
	}
  
  
  private void initializeLayers(Instance inst) {
	  m_DiscretizeCols.setUpper(inst.numAttributes() - 1);
	  numClasses = inst.numClasses();
	  numAttributes = inst.numAttributes();
	  allIntervals = new TreeMap[numAttributes];
	  m_CutPoints = new double[numAttributes][];
	  for (int i = 0; i < inst.numAttributes(); i++) {
		  allIntervals[i] = new TreeMap<Float, Interval>();
	  }  
  }
  
	
	class Interval {
		/**
		 * <p>
		 * Interval class.
		 * </p>
		 */	
		int label;
		float end;
		int [] cd;
		LinkedList<Float> oldPoints; // Implemented as an evicted stack
		Multimap<Float, Integer> histogram;
		float crit;
		
		/**
		 * <p>
		 * Compute the interval ratios.
		 * </p>
		 * @param _attribute
		 * @param []_values
		 * @param _begin
		 * @param _end
		 */
		public Interval(int _label, float _end, int _class) {
			label = _label;
			end = _end;
			oldPoints = new LinkedList<Float>();
			// Initialize histogram and class distribution
			histogram = TreeMultimap.create();
			histogram.put(_end, _class);
			cd = new int[numClasses];
			cd[_class] = 1;
			crit = Float.MIN_VALUE;
		}
		
		public void addPoint(float value, int cls){
			histogram.put(value, cls);
			// Update values
			cd[cls]++;
			label++;
			if(value > end) 
				end = value;
		}
		
		public void mergeIntervals(Interval interv2){
			label = (this.label > interv2.label) ? this.label : interv2.label;
			float innerpoint = (this.end < interv2.end) ? this.end : interv2.end;
			if(oldPoints.size() >= MAX_OLD)
				oldPoints.pollFirst(); // Remove the oldest element
			oldPoints.add(innerpoint);
			// Set the new end
			end = (this.end > interv2.end) ? this.end : interv2.end;
			// Merge histograms and class distributions
			for (int i = 0; i < cd.length; i++) {
				cd[i] += interv2.cd[i];
			}
			histogram.putAll(interv2.histogram);
		}
		
		public void setCrit(float crit) {
			this.crit = crit;
		}
	}
	
}
