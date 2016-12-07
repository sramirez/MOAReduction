package moa.reduction.bayes;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

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
	List<Interval>[] allIntervals;
	Instance[] sample;
	boolean init = false;
	private float lambda, alpha, globalCrit = 0f, globalDiff = 0f;
	private static int MAX_OLD = 5;
	
	public REBdiscretize() {
		// TODO Auto-generated constructor stub
		setAttributeIndices("first-last");
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
	
	class Interval {
		/**
		 * <p>
		 * Interval class.
		 * </p>
		 */	
		int label;
		float end;
		int [] cd;
		LinkedList<Float> oldPoints; // Implementation of an evicted stack
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
		  
	  /*for (int i = instance.numAttributes() - 1; i >= 0; i--) {
		  if ((m_DiscretizeCols.isInRange(i))
				  && (instance.attribute(i).isNumeric())
				  && (instance.classIndex() != i)) {
			  //updateLayer1(instance, i);
    	  
		  }
	  }*/
    
	  if(totalCount > sample.length){
		  if(init) {
			  
			 // Do nothing meanwhile 
		  } else {
			  batchFUSINTER();
			  for (int i = 0; i < allIntervals.length; i++) {
				  printIntervals(i, allIntervals[i]);
			  }
		  }
	  } else {
		  sample[totalCount - 1] = instance;
	  }
  }
  
  private void printIntervals(int att, List<Interval> intervals){
	  System.out.println("Atributo: " + att);
	  for (Iterator<Interval> iterator = intervals.iterator(); iterator.hasNext();) {
		Interval interval = (Interval) iterator.next();
		System.out.println(interval.label + "-" + interval.end + ",");
	  }
	  
  }
  
  private void batchFUSINTER() {
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
			  printIntervals(i, allIntervals[i]);
			  fusinter(i, allIntervals[i]);
			  printIntervals(i, allIntervals[i]);
		  }
	  }
  }
  
  private void fusinter(int att, List<Interval> intervals) {
	int posMin = 0;
	float difference;
	globalDiff = Float.MIN_VALUE;
	globalCrit = evalIntervals(intervals);
	float newMaxCrit = 0;
	
	while(intervals.size() > 1) {
		globalDiff = 0;
		for(int i = 0; i < intervals.size() - 1; i++) {
			float newCrit = evaluteMerge(globalCrit, intervals.get(i), intervals.get(i+1));
			difference = globalCrit - newCrit;
			if(difference > globalDiff){
				posMin = i;
				globalDiff = difference;
				newMaxCrit = newCrit;
			}
		}
	
		if(globalDiff > 0) {
			globalCrit = newMaxCrit;
			Interval int1 = intervals.get(posMin);
			Interval int2 = intervals.remove(posMin+1);
			int1.mergeIntervals(int2);
		} else {
			break;
		}
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
  
  private List<Interval> initIntervals(int att, Integer[] idx) {
		List <Interval> intervals = new LinkedList<Interval> ();
		double valueAnt = sample[idx[0]].value(att);
		int classAnt = (int) sample[idx[0]].classValue();
		Interval lastInterval =  new Interval(1, (float) valueAnt, classAnt);
		intervals.add(lastInterval);
	
		for(int i = 1; i < sample.length;i++) {
			double val = sample[idx[i]].value(att);
			int clas = (int) sample[idx[i]].classValue();
			if(val != valueAnt && clas != classAnt) {
				lastInterval = new Interval(i + 1, (float) val, clas);
				intervals.add(lastInterval);
				valueAnt = val;
				classAnt = clas;
			} else {
				lastInterval.addPoint((float) val, clas);
			}
		}
		return intervals;
	}
  
  	private float evalIntervals (List<Interval> intervals) {
  		return evalIntervals(intervals, Integer.MIN_VALUE);
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

	private float evalIntervals (List<Interval> intervals, int merged) {
		int i, j;
		int Nj;
		float suma, factor, total = 0;
		
		for (i = 0; i < intervals.size(); i++) {
			if (i == merged) {
				Nj = 0;
				for (j=0; j< numClasses; j++) {
					Nj += intervals.get(i).cd[j];
					Nj += intervals.get(i+1).cd[j];
				}
				suma = 0;
				for (j=0; j< numClasses; j++) {
					factor = (intervals.get(i).cd[j] + intervals.get(i+1).cd[j] + lambda) / (Nj + numClasses * lambda);
					suma += factor * (1 - factor);
				}
				total += (alpha * ((float)Nj / totalCount) * suma) + ((1 - alpha) * (((float) numClasses * lambda) / Nj));
			} else if (i != merged + 1) {
				Nj = 0;
				for (j=0; j < numClasses; j++) {
					Nj += intervals.get(i).cd[j];
				}
				suma = 0;
				for (j=0; j < numClasses; j++) {
					factor = (intervals.get(i).cd[j] + lambda) / (Nj + numClasses * lambda);
					suma += factor * (1 - factor);
				}
				float crit = (alpha * ((float) Nj / totalCount) * suma) + ((1 - alpha) * (((float) numClasses * lambda) / Nj));
				intervals.get(i).setCrit(crit);
				total += crit;
			}
		}		
		return total;
	}
  
  
  private void initializeLayers(Instance inst) {
	  m_DiscretizeCols.setUpper(inst.numAttributes() - 1);
	  numClasses = inst.numClasses();
	  numAttributes = inst.numAttributes();
	  allIntervals = new LinkedList[numAttributes];
	  for (int i = 0; i < inst.numAttributes(); i++) {
		  allIntervals[i] = new LinkedList<Interval>();
	  }  
  }
	
}
