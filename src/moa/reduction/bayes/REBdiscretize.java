package moa.reduction.bayes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.NavigableSet;
import java.util.Random;
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
	private float lambda, alpha, globalCrit = 0f, errorRate = 0.25f;
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
		  // INSERTION
		  int cls = (int) instance.classValue();
		  for (int i = 0; i < instance.numAttributes(); i++) {
			 // if numeric and not missing, discretize
			 if(instance.attribute(i).isNumeric() && !instance.isMissing(i)) {
				 float val = (float) instance.value(i);
				 // Get the ceiling interval for the given value
				 Map.Entry<Float, Interval> centralE = allIntervals[i].ceilingEntry(val);
				 Interval central = centralE.getValue();
				 // If it is a boundary point, evaluate six different cutting alternatives
				 // If not, just add the point to the interval
				 if(isBoundary(central, val, cls)){
					  // Add splitting point before dividing the interval
					  central.addPoint(val, cls);
					  Interval splitI = central.splitInterval(val);
					  Map.Entry<Float, Interval> lowerE = allIntervals[i].lowerEntry(central.end);
					  Map.Entry<Float, Interval> higherE = allIntervals[i].higherEntry(central.end);
					  LinkedList<Interval> intervToEvaluate = new LinkedList<Interval>();
					  intervToEvaluate.add(lowerE.getValue()); intervToEvaluate.add(central);
					  intervToEvaluate.add(splitI); intervToEvaluate.add(higherE.getValue());
					  localFusinter(allIntervals[i], intervToEvaluate);
				 } else {
					 central.addPoint(val, cls);
					 central.updateCriterion();
					 if(centralE.getKey() != central.end) {
						 allIntervals[i].remove(centralE.getKey());
						 allIntervals[i].put(central.end, central);
					 }						  
				 }				 
			 }
		  }
		  // REPLACED INSTANCE
		  cls = (int) replacedInstance.classValue();
		  for (int i = 0; i < replacedInstance.numAttributes(); i++) {
			 // if numeric and not missing, discretize
			 if(replacedInstance.attribute(i).isNumeric() && !replacedInstance.isMissing(i)) {
				 float val = (float) replacedInstance.value(i);
				 // Find the interval containing the point to be removed
				 Map.Entry<Float, Interval> ceilingE = allIntervals[i].ceilingEntry(val);
				 Interval central = ceilingE.getValue();
				 // Get the new interval from the splitting to test
				 central.removePoint(val, cls);
				 Interval nextInterval = central.splitByPrevMerge();				 

				// Get the surrounding intervals around the chosen interval
				 LinkedList<Interval> intervalList = new LinkedList<Interval>();
				 Map.Entry<Float, Interval> lowerE = allIntervals[i].lowerEntry(central.end);
				 Map.Entry<Float, Interval> higherE = allIntervals[i].higherEntry(central.end);
				 intervalList.add(lowerE.getValue()); intervalList.add(central);
				 if(nextInterval != null) 
					 intervalList.add(nextInterval); 
				 intervalList.add(higherE.getValue());				 
				 // Evaluate changes in class distributions
				 localFusinter(allIntervals[i], intervalList);				 
			 }
		  }
		  
	  }
	  return updated;
  }
  
  /** Intervals is updated here **/
  private void localFusinter(TreeMap<Float, Interval> intervals, LinkedList<Interval> intervalList) {

	  List<Interval> oldList = (List<Interval>) intervalList.clone();
	  // Apply fusinter locally
	  fusinter(intervalList);
	  // Remove old intervals and add new ones
	  if(intervalList.size() < oldList.size()) {
		  // Delete old intervals
		  for (Iterator iterator = oldList.iterator(); iterator.hasNext();) {
			Interval interval = (Interval) iterator.next();
			intervals.remove(interval.end);
		  }
		  
		  for (Iterator<Interval> iterator = intervalList.iterator(); iterator
				.hasNext();) {
			  Interval interval = iterator.next();
			  intervals.put(interval.end, interval);		
		  }
	 }
  }
  
  private boolean isBoundary(Interval ceiling, float value, int clas){
	  NavigableSet<Float> keys = (NavigableSet<Float>) ceiling.histogram.keySet();
	  float greater = keys.ceiling(value);
	  if(greater == value) 
		  return false;
	  Collection<Integer> vcls = ceiling.histogram.get(greater);
	  if(vcls.contains(clas))
		  return false;
	  return true;
	 
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
			  ArrayList<Interval> intervalList = new ArrayList<Interval>(allIntervals[i].values());
			  fusinter(intervalList);
			  allIntervals[i] = new TreeMap<Float, Interval>();
			  // Update keys in the tree map
			  for (int j = 0; j < intervalList.size(); j++) {
				allIntervals[i].put(intervalList.get(j).end, intervalList.get(j));
			  }
			  printIntervals(i, allIntervals[i].values());
		  }
	  }
  }
  
  private void fusinter(List<Interval> intervalList) {
	
	while(intervalList.size() > 1) {
		float globalDiff = 0;
		float maxGlobalCrit = 0;
		int posMin = 0;
		for(int i = 0; i < intervalList.size() - 1; i++) {
			float newLocalCrit = evaluteMerge(intervalList.get(i).cd, intervalList.get(i+1).cd);
			float newGlobalCrit = globalCrit - intervalList.get(i).crit - intervalList.get(i+1).crit + newLocalCrit;
			float difference = globalCrit - newGlobalCrit;
			if(difference > globalDiff){
				posMin = i;
				globalDiff = difference;
				maxGlobalCrit = newGlobalCrit;
			}
		}
	
		if(globalDiff > 0) {
			globalCrit = maxGlobalCrit;
			Interval int1 = intervalList.get(posMin);
			Interval int2 = intervalList.remove(posMin+1);
			int1.mergeIntervals(int2);			
		} else {
			break;
		}
	}
	
  }
  
  // EvalIntervals must be executed before calling this method */
  private float evaluteMerge(int[] cd1, int[] cd2) {
	  int[] cds = cd1.clone();
	  for (int i = 0; i < cds.length; i++) {
		cds[i] += cd2[i];
	  }
	  return  evalInterval(cds);
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
	
		for(int i = 1; i < sample.length;i++) {
			float val = (float) sample[idx[i]].value(att);
			int clas = (int) sample[idx[i]].classValue();
			if(val != valueAnt && clas != classAnt) {
				Interval nInt = new Interval(lastInterval);
				nInt.updateCriterion();// Important
				globalCrit += nInt.crit;		
				intervals.put(valueAnt, nInt);
				lastInterval = new Interval(i + 1, val, clas);
			} else {
				lastInterval.addPoint(val, clas);
			}
			valueAnt = val;
			classAnt = clas;
		}
		intervals.put(valueAnt, lastInterval);
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
    
  private int[][] evaluateSplit(Interval original, float value, int cls){
	  int[][] nCd = new int[2][original.cd.length];
	  nCd[1] = original.cd.clone();
	  nCd[0][cls]++;
	  for (Iterator iterator = original.histogram.entries().iterator(); iterator.hasNext();) {
		  Entry<Float, Integer> entry = (Entry) iterator.next();
		  if(entry.getKey() <= value){
			  nCd[0][entry.getValue()]++;
			  nCd[1][entry.getValue()]--;
		  }				
	  }
	  return nCd;
	  
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
		
		public Interval() {
			// TODO Auto-generated constructor stub
		}
		
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
			histogram = TreeMultimap.create();
			histogram.put(_end, _class);
			cd = new int[numClasses];
			cd[_class] = 1;
			crit = Float.MIN_VALUE;
		}
		
		public Interval(Interval other) {
			// TODO Auto-generated constructor stub
			label = other.label;
			end = other.end;
			cd = other.cd.clone();
			crit = other.crit;
			oldPoints = new LinkedList<Float>(other.oldPoints);
			histogram = TreeMultimap.create();
			histogram.putAll(other.histogram);
		}
		
		public void addPoint(float value, int cls){
			histogram.put(value, cls);
			// Update values
			cd[cls]++;
			label++;
			if(value > end) 
				end = value;
		}
		
		public void removePoint(float value, int cls) {
			histogram.remove(value, cls);
			cd[cls]--;
			label--;
			if(value == end) {
				NavigableSet<Float> keyset = (NavigableSet<Float>) histogram.keySet();
				Float nend = keyset.floor(value); // get the new maximum
				if(nend != null)
					end = nend;
				else
					end = -1;
			}
		}
		
		public Interval splitByPrevMerge() {			
			Float value = oldPoints.pollLast();
			if(value != null)
				return splitInterval(value);
			return null;
		}
		
		public Interval splitInterval(float value) {
			TreeMultimap<Float, Integer> nHist = TreeMultimap.create();
			int[] nCd = new int[cd.length];
			LinkedList<Float> nOP = new LinkedList<Float>();
			for (Iterator iterator = histogram.entries().iterator(); iterator.hasNext();) {
				Entry<Float, Integer> entry = (Entry) iterator.next();
				if(entry.getKey() <= value){
					//histogram.entries().remove(entry);
					nHist.put(entry.getKey(), entry.getValue());
					nCd[entry.getValue()]++;
					cd[entry.getValue()]--;
					iterator.remove();
				}				
			}
			// Split old points
			for (Iterator iterator = oldPoints.iterator(); iterator.hasNext();) {
				float float1 = (float) iterator.next();
				if(float1 <= value){
					nOP.add(float1);
					iterator.remove();
				}
			}
			this.setCrit(evalInterval(nCd));
			/** New interval **/
			Interval nInterval = new Interval();
			nInterval.cd = nCd;
			nInterval.histogram = nHist;
			nInterval.oldPoints = nOP;
			nInterval.label = this.label - this.histogram.size();
			//nInterval.addPoint(value, cls);
			nInterval.setCrit(evalInterval(nCd));
			return nInterval;
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
			setCrit(evaluteMerge(this.cd, interv2.cd));
		}
		
		public void setCrit(float crit) {
			this.crit = crit;
		}
		
		public void updateCriterion(){
			this.crit = evalInterval(cd);
		}
		
	}
	
}
