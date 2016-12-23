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
	private static int MAX_OLD = 5, INIT_TH = 4137;
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
		int sampleSize = 4137;
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

	  Instance replacedInstance = null;
	  if(totalCount > sample.length){
		  if(rand.nextFloat() < errorRate + ERR_TH){
			 int pos = isample % sample.length;
			 replacedInstance = sample[pos];
			 sample[pos] = instance.copy();
		  } 
		  isample++;
	  } else {
		  sample[totalCount - 1] = instance.copy();
	  }
    
	  // If there are enough instances to initialize cut points, do it!
	  if(totalCount >= INIT_TH){
		  if(init) {
			  for (int i = 0; i < instance.numAttributes(); i++) {
				 // if numeric and not missing, discretize
				 if(instance.attribute(i).isNumeric() && !instance.isMissing(i)) {
					 insertExample(i, instance);
					 if(replacedInstance != null)
						 deleteExample(i, replacedInstance);
				 }
			  }
		  } else {
			  batchFusinter();
			  for (int i = 0; i < allIntervals.length; i++) {
				  printIntervals(i, allIntervals[i].values());
			  }
			  init = true;
		  }
	  }
  }
  
  private void checkHistogramIntervals(Interval interval){
	  	int s1 = 0, total1 = 0;
		for (int i = 0; i < interval.cd.length; i++) {
			s1 += interval.cd[i];
		}
		
		for (Iterator iterator = interval.histogram.entrySet().iterator(); iterator.hasNext();) {
			Entry<Float, int[]> entry = (Entry) iterator.next();
			int[] v = entry.getValue();
			for (int i = 0; i < v.length; i++) {
				total1 += v[i];
			}
		}
		if(s1 != total1)
			System.err.println("asd");
  }
  
  private void checkLabelIntervals(Instance instance){
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
	 // Get the ceiling interval for the given value
	 Map.Entry<Float, Interval> centralE = allIntervals[att].ceilingEntry(val);
	 // The point is within the range defined by centralE, if not a new maximum interval is created
	 if(centralE != null) {
		 Interval central = centralE.getValue();
		 // If it is a boundary point, evaluate six different cutting alternatives
		 if(isBoundary(att, central, val, cls)){
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
	 }
  }
  
  private void deleteExample(int att, Instance instance) {
	 int cls = (int) instance.classValue();
	 float val = (float) instance.value(att);
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
			 central.removePoint(att, val, cls);
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
  
  private void checkUnfoundPoint(int att, float value) {
	  boolean found = false;
	  for (Iterator<Interval> iterator = allIntervals[att].values().iterator(); iterator.hasNext();) {
			Interval interval = iterator.next();
			if(interval.histogram.containsKey(value)){
				found = true;
			}	
	  }
	  if(!found)
		  System.err.println("Not found");
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
  
  private boolean isBoundary(int att, Interval ceiling, float value, int clas){
	  Entry<Float, int[]> following = ceiling.histogram.ceilingEntry(value);
	  boolean boundary = false;
	  // The previous point is in another interval (interval with a single point)
	  if(following == null) {		  
		  Entry<Float, Interval> higherE = allIntervals[att].higherEntry(ceiling.end);
		  if(higherE == null) {
			  boundary = true; // no more points at the right side
		  } else {
			  following = higherE.getValue().histogram.ceilingEntry(value);
			  int[] cd1 = following.getValue();		
			  int[] cd2 = new int[cd1.length];
			  cd2[clas]++;
			  boundary = isBoundary(cd1, cd2);
		  }
	  } else {
		  // If the point already exists before, evaluate if it is now a boundary
		  if(following.getKey() == value) {
			  Entry<Float, int[]> nextnext = ceiling.histogram.higherEntry(value);
			  if(nextnext != null) {
				  int[] cd1 = following.getValue();				  
				  cd1[clas]++;
				  boundary = isBoundary(cd1, nextnext.getValue());
				  cd1[clas]--;
			  } else {
				  // Last point in the interval, it does not make sense to split
				  boundary = false;
			  }
		  } else {
			  int[] cd1 = following.getValue();		
			  int[] cd2 = new int[cd1.length];
			  cd2[clas]++;
			  boundary = isBoundary(cd1, cd2);
		  }
	  }
	  return boundary;
	 
  }
  
  private boolean isBoundary(int[] cd1, int[] cd2){
	  int count = 0;
	  for (int i = 0; i < cd1.length; i++) {
		  if(cd1[i] + cd2[i] > 0) {
			  if(++count > 1) {
				  return true;
			  }						  
		  }
	  }
	  return false;
  }
  
  private void printIntervals(int att, Collection<Interval> intervals){
	  System.out.println("Atributo: " + att);
	  int sum = 0;
	  for (Iterator<Interval> iterator = intervals.iterator(); iterator.hasNext();) {
		Interval interval = (Interval) iterator.next();
		for (int i = 0; i < interval.cd.length; i++) {
			sum += interval.cd[i];
		}
		System.out.println(interval.label + "-" + interval.end + "," + sum);
	  }	  
  }
  
  private void batchFusinter() {
	  // TODO Auto-generated method stub
	  float[][] sorted = new float[numAttributes][sample.length];
	  final int[] classData = new int[sample.length];
	  int nvalid = 0;
	  for (int j = 0; j < sample.length && sample[j] != null; j++) {
		  classData[j] = (int) sample[j].classValue();
		  nvalid++;
	  }
	  
	  for (int i = numAttributes - 1; i >= 0; i--) {
		  if ((m_DiscretizeCols.isInRange(i))
				  && (sample[0].attribute(i).isNumeric())
				  && (sample[0].classIndex() != i)) {
			  Integer[] idx = new Integer[nvalid];
			  for (int j = 0; j < idx.length; j++) {
				  idx[j] = j;
				  sorted[i][j] = (float) sample[j].value(i);
			  }
			  final float[] data = sorted[i];
			  
			  // Order by feature value and class
			  Arrays.sort(idx, new Comparator<Integer>() {
			      @Override 
			      public int compare(final Integer o1, final Integer o2) {
			    	  int cmp_value = Float.compare(data[o1], data[o2]);
			    	  if(cmp_value == 0) 
			    		  cmp_value = Integer.compare(classData[o1], classData[o2]);
			          return cmp_value;
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
			float newGlobalCrit = globalCrits[att] - intervalList.get(i).crit - intervalList.get(i+1).crit
					+ newLocalCrit;
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
		LinkedList<Tuple<Float, int[]>> distinctPoints = new LinkedList<Tuple<Float, int[]>>();
		float valueAnt = (float) sample[idx[0]].value(att);
		int classAnt = (int) sample[idx[0]].classValue();
		int[] cd = new int[numClasses];
		cd[classAnt]++;
		
		for(int i = 1; i < idx.length;i++) {
			float val = (float) sample[idx[i]].value(att);
			int clas = (int) sample[idx[i]].classValue();
			if(val == valueAnt) {
				cd[clas]++;
			} else {
				distinctPoints.add(new Tuple<Float, int[]>(valueAnt, cd));
				cd = new int[numClasses];
				cd[clas]++;
				valueAnt = val;
			}
			
		}		
		distinctPoints.add(new Tuple<Float, int[]>(valueAnt, cd));
		
		Interval interval = new Interval(labelsToUse[att].poll());
		Tuple<Float, int[]> t1 = distinctPoints.get(0);
		interval.addPoint(t1);
		
		for (int i = 1; i < distinctPoints.size(); i++) {
			Tuple<Float, int[]> t2 = distinctPoints.get(i);
			if(isBoundary(t1.y, t2.y)){
				intervals.put(t1.x, interval);
				interval = new Interval(labelsToUse[att].poll());	
			}
			interval.addPoint(t2);
			t1 = t2;
		}
		intervals.put(t1.x, interval);		
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
		
		public Interval(int _label) {
			label = _label;
			end = -1;
			oldPoints = new LinkedList<Float>();
			histogram = new TreeMap<Float, int[]>();
			cd = new int[numClasses];
			//histogram.put(_end, cd.clone());
			crit = Float.MIN_VALUE;
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
		
		public void addPoint(float value, int cd[]){
			int[] prevd = histogram.get(value);
			if(prevd == null) {
				prevd = new int[numClasses];
			} else {
				for (int i = 0; i < cd.length; i++) {
					prevd[i] += cd[i];
				}
			}
			histogram.put(value, prevd);
			
			for (int i = 0; i < cd.length; i++) {
				this.cd[i] += cd[i];
			}
			
			if(value > end) 
				end = value;
		}
		
		public void addPoint(Tuple<Float, int[]> point){
			addPoint(point.x, point.y);
		}
		
		public void removePoint(int att, float value, int cls) {
			checkHistogramIntervals(this);
			int[] pd = histogram.get(value);
			if(pd != null) {
				if(pd[cls] > 0) {
					pd[cls]--;
					cd[cls]--;
				} else {
					System.err.println("Bad histogram.");
				}
				boolean delete = true;
				// If all values are equal to zero, remove the point from the map
				for (int i = 0; i < pd.length; i++) {
					if(pd[i] > 0){
						delete = false;
						break;
					}						
				}
				if(delete){
					histogram.remove(value);
				}
			} else {
				// Error, no point in this range.
				System.err.println("Point not found in the given range.");
				checkUnfoundPoint(att, value);
			}
			// Find a new maximum if the point removed is the maximum
			if(value == end) {
				Float newend = histogram.floorKey(value); // get the new maximum
				if(newend != null)
					end = newend;
			}
			checkHistogramIntervals(this);
		}
		
		public Interval splitByPrevMerge(int att) {			
			Float value = oldPoints.pollLast();
			if(value != null)
				return splitInterval(att, value);
			return null;
		}
		
		public Interval splitInterval(int att, float value) {
			
			checkHistogramIntervals(this);
			
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
			// Label management
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
			
			// Testings
			checkHistogramIntervals(this);
			checkHistogramIntervals(nInterval);
			
			return nInterval;
		}
		
		public int mergeIntervals(Interval interv2){
			// The interval with less elements lose its label
			//label = (this.label > interv2.label) ? this.label : interv2.label;
			checkHistogramIntervals(this);
			checkHistogramIntervals(interv2);
			
			int s1 = 0, s2 = 0;
			for (int i = 0; i < cd.length; i++) {
				s1 += cd[i];
				s2 += interv2.cd[i];
			}			
			
			int oldlab = interv2.label;
			if(s1 < s2) {
				oldlab = this.label;
				this.label = interv2.label;
			}
			
			// Merge old splitting points
			float innerpoint = (this.end < interv2.end) ? this.end : interv2.end;
			if(oldPoints.size() >= MAX_OLD)
				oldPoints.pollFirst(); // Remove the oldest element
			oldPoints.add(innerpoint);
			
			// Set the new end
			//end = (this.end > interv2.end) ? this.end : interv2.end;
			end = (interv2.end + end) / 2.0f;
			
			// Merge histograms and class distributions
			for (int i = 0; i < cd.length; i++) {
				cd[i] += interv2.cd[i];
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
			
			// Testings
			checkHistogramIntervals(this);
			
			updateCriterion();
			return oldlab;
		}
		
		public void setEnd(float end) {
			this.end = end;
		}
		
		public void setCrit(float crit) {
			this.crit = crit;
		}
		
		public void updateCriterion(){
			this.crit = evalInterval(cd);
		}
		
	}
	
	public class Tuple<X, Y> { 
		  public final X x; 
		  public final Y y; 
		  public Tuple(X x, Y y) { 
		    this.x = x; 
		    this.y = y; 
		  } 
		} 

}
