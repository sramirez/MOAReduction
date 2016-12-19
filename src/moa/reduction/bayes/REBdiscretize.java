package moa.reduction.bayes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Queue;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;

import moa.reduction.core.MOADiscretize;

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
	private float lambda, alpha, errorRate = 0.25f;
	private static int MAX_OLD = 5;
	private static float ERR_TH = 0.25f;
	private Random rand;
	private Queue<Integer>[] labelsToUse; 
	private float[] globalCrits;
	
	public REBdiscretize() {
		// TODO Auto-generated constructor stub
		setAttributeIndices("first-last");
		this.rand = new Random(seed);
		this.alpha = 0.5f;
		this.lambda = 0.5f;
		int sampleSize = 1000;
		this.sample = new Instance[sampleSize];
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
					 String[] labels = new String[allIntervals[i].size()];
					 int j = 0;
					 Set<Integer> labels2 = new HashSet<Integer>();
				 	for (Iterator<Interval> iterator = allIntervals[i].values().iterator(); iterator
				 			.hasNext();) {
				 		Interval interv = iterator.next();
				 		labels[j] = Integer.toString(interv.label);
				 		boundaries[j++] = interv.end;
				 		if(labels2.contains(interv.label))
							System.err.println("Asd");
						labels2.add(interv.label);

				 	}
				 	m_Labels[i] = labels;
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
	  if(totalCount == 1008)
		  System.err.println("asd");
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
				 
				 if(i == 4) {
					  System.out.println("Global Crit: " + globalCrits[i]);
					  System.out.println("Size: " + allIntervals[i].size());
				  }
				  insertExample(i, instance);
				  if(i == 4) {
					  System.out.println("Global Crit: " + globalCrits[i]);
					  System.out.println("Size: " + allIntervals[i].size());
				  }
				  checkIntervals(instance);
				  deleteExample(i, replacedInstance);
				  checkIntervals(instance);
				  if(i == 4) {
					  System.out.println("Global Crit: " + globalCrits[i]);
					  System.out.println("Size: " + allIntervals[i].size());
				  }
			 }
		  }
	  }
	  return updated;
  }
  
  private void checkIntervals(Instance instance){
		 for (int i = 0; i < instance.numAttributes(); i++) {
			 Set<Integer> labels2 = new HashSet<Integer>();
			 for (Iterator<Interval> iterator = allIntervals[i].values().iterator(); iterator
		 			.hasNext();) {
		 		Interval interv = iterator.next();
		 		
		 		if(labels2.contains(interv.label))
					System.err.println("Asd");
				labels2.add(interv.label);

		 	}
		 }
  }
  
  private void insertExample(int att, Instance instance){
  	 // INSERTION
     int cls = (int) instance.classValue();
	 float val = (float) instance.value(att);
	 if(val == 0.041161 && cls == 1)
		 System.err.println("as");
	 // Get the ceiling interval for the given value
	 Map.Entry<Float, Interval> centralE = allIntervals[att].ceilingEntry(val);
	 // The point is within the range defined by centralE, if not a new maximum interval is created
	 if(centralE != null) {
		 Interval central = centralE.getValue();
		 // If it is a boundary point, evaluate six different cutting alternatives
		 if(isBoundary(central, val, cls)){
			  // remove before changing end value in central
			  allIntervals[att].remove(centralE.getKey());
			  // Add splitting point before dividing the interval
			  central.addPoint(val, cls);
			  Interval splitI = central.splitInterval(att, val);
			  Map.Entry<Float, Interval> lowerE = allIntervals[att].lowerEntry(central.end);
			  Map.Entry<Float, Interval> higherE = allIntervals[att].higherEntry(central.end);
			  LinkedList<Interval> intervalList = new LinkedList<Interval>();
			  // Insert in the specific order
			  if(lowerE != null) {
				 intervalList.add(lowerE.getValue());
				 allIntervals[att].remove(lowerE.getKey());
			  }
			  intervalList.add(central);
			  if(splitI != null)
				  intervalList.add(splitI); 
			  if(higherE != null) {
				  intervalList.add(higherE.getValue());
				  allIntervals[att].remove(higherE.getKey());
			  }
			  evaluateLocalMerges(att, intervalList);
			  insertIntervals(att, intervalList);
			  checkIntervals(instance);
		 } else {
			 // If not, just add the point to the interval
			 central.addPoint(val, cls);
			 central.updateCriterion();
			 // Update the key with the bigger end
			 if(centralE.getKey() != central.end) {
				 allIntervals[att].remove(centralE.getKey());
				 allIntervals[att].put(central.end, central);
			 }						  
		 }
	 } else {
		 // New interval with a new maximum limit
		 Map.Entry<Float, Interval> priorE = allIntervals[att].lowerEntry(val);
		 Interval nInt = new Interval(labelsToUse[att].poll(), val, cls);
		 allIntervals[att].put(val, nInt);
		 checkIntervals(instance);
	 }
  }
  
  private void deleteExample(int att, Instance instance) {
	 int cls = (int) instance.classValue();
	 float val = (float) instance.value(att);
	 if(val == 0.041161 && cls == 1)
		 System.err.println("as");
	 // Find the interval containing the point to be removed
	 Map.Entry<Float, Interval> ceilingE = allIntervals[att].ceilingEntry(val);
	 if(ceilingE != null){ // The point must be contained in any previously inserted interval
		 Interval central = ceilingE.getValue();
		 Map.Entry<Float, Interval> lowerE = allIntervals[att].lowerEntry(central.end);
		 Map.Entry<Float, Interval> higherE = allIntervals[att].higherEntry(central.end);
		 // Delete surrounding interval before starting
		 LinkedList<Interval> intervalList = new LinkedList<Interval>();
		 if(lowerE != null) {					 
			 intervalList.add(lowerE.getValue());
			 allIntervals[att].remove(lowerE.getKey());
		 }
		 if(central != null){					 
			 intervalList.add(central);
			 allIntervals[att].remove(ceilingE.getKey());
		 }
		 if(higherE != null) {					 
			 intervalList.add(higherE.getValue());
			 allIntervals[att].remove(higherE.getKey());
		 }
		 
		 // Get the new interval from the splitting to test (if it is not the last interval and point)
		 if(allIntervals[att].size() > 1 && central.histogram.size() > 1) {
			 // remove before changing end value in central
			 //allIntervals[i].remove(ceilingE.getKey());
			 if(val == 0.041161)
				 System.err.println("asd");
			 central.removePoint(val, cls);
		 }
		 Interval splittedInterval = null;
		 // If central is empty, so we merge it with its prior interval or we just remove it
		 if(central.histogram.isEmpty()){
			 if(lowerE != null) {
				 //allIntervals[i].remove(lowerE.getKey());
				 int oldlab = lowerE.getValue().mergeIntervals(central);
				 labelsToUse[att].add(oldlab);
				 // Replace old end with the new compelling interval 
				 //allIntervals[i].put(lowerE.getValue().end, lowerE.getValue());	 
			 }// else {
				 // Just remove central from map
				 //allIntervals[i].remove(central.end);
			 //}
			 central = null;
		 } else {
			 splittedInterval = central.splitByPrevMerge(att);
		 }
		 // Get the surrounding intervals around the chosen interval
		 intervalList = new LinkedList<Interval>();
		 if(lowerE != null)
			 intervalList.add(lowerE.getValue());
		 if(central != null)
			 intervalList.add(central);
		 if(splittedInterval != null) 
			 intervalList.add(splittedInterval); 
		 if(higherE != null)
			 intervalList.add(higherE.getValue());				 
		 // Evaluate changes in class distributions (allIntervals is updated)
		 evaluateLocalMerges(att, intervalList);		
		 insertIntervals(att, intervalList);
	 }
  }
  
  private void deleteSurroundingIntervals(int att, LinkedList<Interval> intervalList){
	// Delete old intervals
	  for (Iterator<Interval> iterator = intervalList.iterator(); iterator.hasNext();) {
		Interval interval = iterator.next();
		if(interval != null)
			allIntervals[att].remove(interval.end);
	  }
  }
  
  private void insertIntervals(int att, LinkedList<Interval> intervalList){
	  for (Iterator<Interval> iterator = intervalList.iterator(); iterator
				.hasNext();) {
			  Interval interval = iterator.next();
			  if(interval != null)
				  allIntervals[att].put(interval.end, interval);		
		  }
	  
  }
  
  private boolean isBoundary(Interval ceiling, float value, int clas){
	  Entry<Float, int[]> greater = ceiling.histogram.ceilingEntry(value);
	  // The new greatest point in the attribute
	  if(greater == null)
		  return true;
	  if(greater.getKey() == value) 
		  return false;
	  if(greater.getValue()[clas] > 0)
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
			  evaluateLocalMerges(i, intervalList);
			  allIntervals[i] = new TreeMap<Float, Interval>();
			  // Update keys in the tree map
			  for (int j = 0; j < intervalList.size(); j++) {
				allIntervals[i].put(intervalList.get(j).end, intervalList.get(j));
			  }
			  printIntervals(i, allIntervals[i].values());
		  }
	  }
  }
  
  private void evaluateLocalMerges(int att, List<Interval> intervalList) {
	
	while(intervalList.size() > 1) {
		float globalDiff = 0;
		float maxGlobalCrit = 0;
		int posMin = 0;
		for(int i = 0; i < intervalList.size() - 1; i++) {
			float newLocalCrit = evaluteMerge(intervalList.get(i).cd, intervalList.get(i+1).cd);
			float newGlobalCrit = globalCrits[att] - intervalList.get(i).crit - intervalList.get(i+1).crit + newLocalCrit;
			if(Float.isInfinite(newGlobalCrit))
				System.out.println("Error");
			float difference = globalCrits[att] - newGlobalCrit;
			if(difference > globalDiff){
				posMin = i;
				globalDiff = difference;
				maxGlobalCrit = newGlobalCrit;
			}
		}
	
		if(globalDiff > 0) {
			globalCrits[att] = maxGlobalCrit;
			Interval int1 = intervalList.get(posMin);
			Interval int2 = intervalList.remove(posMin+1);
			int oldlab = int1.mergeIntervals(int2);
			labelsToUse[att].add(oldlab);
		} else {
			break;
		}
	}
	
  }
  
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
		Interval lastInterval =  new Interval(labelsToUse[att].poll(), valueAnt, classAnt);
		if(valueAnt == 0.041161 && classAnt == 1)
			System.err.println("asd");
		for(int i = 1; i < sample.length;i++) {
			if(att == 3) {
				System.err.println("asd");
			}
			float val = (float) sample[idx[i]].value(att);
			int clas = (int) sample[idx[i]].classValue();
			if(val == 0.041161 && clas == 1)
				System.err.println("asd");
			if(val != valueAnt && clas != classAnt) {
				Interval nInt = new Interval(lastInterval);
				nInt.updateCriterion();// Important
				globalCrits[att] += nInt.crit;		
				intervals.put(valueAnt, nInt);
				lastInterval = new Interval(labelsToUse[att].poll(), val, clas);
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
	  m_Labels = new String[numAttributes][];
	  labelsToUse = new Queue[numAttributes];
	  globalCrits = new float[numAttributes];
	  for (int i = 0; i < inst.numAttributes(); i++) {
		  allIntervals[i] = new TreeMap<Float, Interval>();
		  labelsToUse[i] = new LinkedList<Integer>();
		  for (int j = 0; j < sample.length; j++) {
				labelsToUse[i].add(j);
			}
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
		TreeMap<Float, int[]> histogram;
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
			histogram = new TreeMap<Float, int[]>();
			cd = new int[numClasses];
			cd[_class] = 1;
			histogram.put(_end, cd.clone());
			crit = Float.MIN_VALUE;
			int s1 = 0, s2 = 0, total1 = 0, total2 = 0;
			for (int i = 0; i < cd.length; i++) {
				s1 += cd[i];
			}
			
			for (Iterator iterator = histogram.entrySet().iterator(); iterator.hasNext();) {
				Entry<Float, int[]> entry = (Entry) iterator.next();
				int[] v = entry.getValue();
				for (int i = 0; i < v.length; i++) {
					total2 += v[i];
				}
			}
			if(s1 != total2)
				System.err.println("asd");
		}
		
		public Interval(Interval other) {
			// TODO Auto-generated constructor stub
			label = other.label;
			end = other.end;
			cd = other.cd.clone();
			crit = other.crit;
			oldPoints = new LinkedList<Float>(other.oldPoints);
			histogram = new TreeMap<Float, int[]>();
			histogram.putAll(other.histogram);
		}
		
		public void addPoint(float value, int cls){
			if(value == 0.041161f && cls == 1)
				System.err.println("asd");
			int[] pd = histogram.get(value);
			if(pd != null) {
				pd[cls]++;
			} else {
				pd = new int[cd.length];
				pd[cls]++;
				histogram.put(value, pd);
			}
			// Update values
			cd[cls]++;
			if(value > end) 
				end = value;
		}
		
		public void removePoint(float value, int cls) {
			int[] pd = histogram.get(value);
			if(pd != null) {
				if(pd[cls] > 0) {
					pd[cls]--;
					cd[cls]--;
				}
				boolean delete = true;
				// If all values are equal to zero, remove the point from the map
				for (int i = 0; i < pd.length; i++) {
					if(pd[cls] == 0){
						delete = false;
						break;
					}						
				}
				if(delete) histogram.remove(value);
			} else {
				// Error, no point in this range.
				if(value == 0.041161)
					System.err.println("asd");
				System.err.println("Point not found in the given range.");
			}
			// Find a new maximum if the point removed is the maximum
			if(value == end) {
				Float newend = histogram.floorKey(value); // get the new maximum
				if(newend != null)
					end = newend;
			}
		}
		
		public Interval splitByPrevMerge(int att) {			
			Float value = oldPoints.pollLast();
			if(value != null)
				return splitInterval(att, value);
			return null;
		}
		
		public Interval splitInterval(int att, float value) {
			TreeMap<Float, int[]> nHist = new TreeMap<Float, int[]>();
			int[] nCd = new int[cd.length];
			LinkedList<Float> nOP = new LinkedList<Float>();
			for (Iterator iterator = histogram.entrySet().iterator(); iterator.hasNext();) {
				Entry<Float, int[]> entry = (Entry) iterator.next();
				if(entry.getKey() > value){
					//histogram.entries().remove(entry);
					nHist.put(entry.getKey(), entry.getValue());
					for(int i = 0; i < entry.getValue().length; i++){
						nCd[i] += entry.getValue()[i];
						cd[i] -= entry.getValue()[i];
					}					
					iterator.remove();
				}				
			}
			
			if(nHist.isEmpty())
				return null;
			// Split old points
			for (Iterator<Float> iterator = oldPoints.iterator(); iterator.hasNext();) {
				float float1 = (float) iterator.next();
				if(float1 > value){
					nOP.add(float1);
					iterator.remove();
				}
			}
			
			/** New interval (which lays at the right of the cut point) **/
			int s1 = 0, s2 = 0;
			for (int i = 0; i < nCd.length; i++) {
				s1 += cd[i];
				s2 += nCd[i];
			}
			
			Interval nInterval = new Interval();
			if(s1 > s2){
				nInterval.label = labelsToUse[att].poll();
			} else {
				nInterval.label = this.label;
				this.label = labelsToUse[att].poll();
			}
			nInterval.cd = nCd;
			nInterval.histogram = nHist;
			nInterval.oldPoints = nOP;
			nInterval.end = this.end;
			nInterval.updateCriterion();
			/** Old interval (which lays at the left of the cut point) **/
			//this.label -= nHist.size();
			this.end = value;
			updateCriterion();
			return nInterval;
		}
		
		public int mergeIntervals(Interval interv2){
			
			

			// The interval with less elements lose its label
			//label = (this.label > interv2.label) ? this.label : interv2.label;
			int s1 = 0, s2 = 0, total1 = 0, total2 = 0;
			for (int i = 0; i < cd.length; i++) {
				s1 += cd[i];
				s2 += interv2.cd[i];
			}
			
			for (Iterator iterator = histogram.entrySet().iterator(); iterator.hasNext();) {
				Entry<Float, int[]> entry = (Entry) iterator.next();
				int[] v = entry.getValue();
				for (int i = 0; i < v.length; i++) {
					total2 += v[i];
				}
			}
			if(s1 != total2)
				System.err.println("asd");
			
			
			int oldlab = interv2.label;
			if(s1 < s2) {
				oldlab = this.label;
				this.label = interv2.label;
			}
			
			float innerpoint = (this.end < interv2.end) ? this.end : interv2.end;
			if(oldPoints.size() >= MAX_OLD)
				oldPoints.pollFirst(); // Remove the oldest element
			oldPoints.add(innerpoint);
			// Set the new end
			end = (this.end > interv2.end) ? this.end : interv2.end;
			// Merge histograms and class distributions
			total1 = 0;
			total2 = 0;
			for (int i = 0; i < cd.length; i++) {
				cd[i] += interv2.cd[i];
				total1 += cd[i];
			}
			
			for (Iterator iterator = interv2.histogram.entrySet().iterator(); iterator.hasNext();) {
				Entry<Float, int[]> entry = (Entry) iterator.next();
				int[] v = histogram.get(entry.getKey());
				if(v != null) {
					for (int i = 0; i < v.length; i++){
						v[i] += entry.getValue()[i];
					}
				} else {
					histogram.put(entry.getKey(), entry.getValue());
				}
			}
			
			for (Iterator iterator = histogram.entrySet().iterator(); iterator.hasNext();) {
				Entry<Float, int[]> entry = (Entry) iterator.next();
				int[] v = entry.getValue();
				for (int i = 0; i < v.length; i++) {
					total2 += v[i];
				}
			}
			if(total1 != total2)
				System.err.println("asd");
			updateCriterion();
			return oldlab;
		}
		
		public void setCrit(float crit) {
			this.crit = crit;
		}
		
		public void updateCriterion(){
			this.crit = evalInterval(cd);
		}
		
	}

}
