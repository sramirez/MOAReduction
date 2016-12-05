package moa.reduction.bayes;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.TreeMap;
import java.util.Vector;

import moa.reduction.core.MOADiscretize;
import moa.reduction.core.FUSINTER.Interval;

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
	private float lambda, alpha;
	
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
		LinkedList<Double> oldPoints;
		TreeMap<Float, List<Integer>> histogram;
		
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
			oldPoints = new LinkedList<Double>();
			// Initialize histogram and class distribution
			histogram = new TreeMap<Float, List<Integer>>();
			List<Integer> l = new LinkedList<Integer>();
			l.add(_class);
			histogram.put(_end, l);
			cd = new int[numClasses];
			cd[_class] = 1;
		}
		
		public void addPoint(float value, int cls){
			List<Integer> l = histogram.get(value);
			if(l != null) {
				l.add(cls);
				histogram.replace(value, l);
			} else {
				l = new LinkedList<Integer>();
				l.add(cls);
				histogram.put(value, l);
			}
			// Update values
			cd[cls]++;
			label++;
			if(value > end) 
				end = value;
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
		  
	  for (int i = instance.numAttributes() - 1; i >= 0; i--) {
		  if ((m_DiscretizeCols.isInRange(i))
				  && (instance.attribute(i).isNumeric())
				  && (instance.classIndex() != i)) {
			  //updateLayer1(instance, i);
    	  
		  }
	  }
    
	  if(totalCount > sample.length){
		  if(init) {
			  
		  } else {
			  batchFUSINTER();
		  }
	  } else {
		  sample[totalCount - 1] = instance;
	  }
  }
  
  private void batchFUSINTER() {
	  // TODO Auto-generated method stub
	  float[][] sorted = new float[numAttributes][sample.length];
	  
	  for (int i = numAttributes - 1; i >= 0; i--) {
		  if ((m_DiscretizeCols.isInRange(i))
				  && (sample[0].attribute(i).isNumeric())
				  && (sample[0].classIndex() != i)) {
			  for (int j = 0; j < sample.length; j++) {
			  //updateLayer1(instance, i);
				  sorted[i][j] = (float) sample[j].value(i);
			  }
			  Arrays.sort(sorted[i]);
			  allIntervals[i] = initIntervals(i, sorted[i]);
		  }
	  }  
  }
  
  private void FUSINTER(int att, List<Interval> intervals) {
	boolean exit=false;
	double criterion;
	
	while(intervals.size() > 1 && !exit) {
		int posMin=-1;
		double maxCri=0;
		double eval = evalIntervals(intervals, -2);
		for(int i = 0; i < intervals.size() - 1; i++) {
			criterion = eval - evalIntervals(intervals, i);
			if(posMin==-1) {
				posMin = i;
				maxCri = criterion;
			} else {
				if(criterion > maxCri) {
					posMin=i;
					maxCri = criterion;
				}
			}
		}
	
		if(maxCri > 0) {
			Interval int1 = (Interval)intervals.elementAt(posMin);
			Interval int2 = (Interval)intervals.elementAt(posMin+1);
			int1.enlargeInterval(int2.end);
			intervals.removeElementAt(posMin+1);
		} else {
			exit=true;
		}
	}
  }
  
  private List<Interval> initIntervals(int att, float[] values) {
		List <Interval> intervals = new LinkedList<Interval> ();
		double valueAnt = sample[0].value(att);
		int classAnt = (int) sample[0].classValue();
		Interval lastInterval =  new Interval(1, (float) valueAnt, classAnt);
	
		for(int i= 1; i < sample.length;i++) {
			double val = sample[i].value(att);
			int clas = (int) sample[i].classValue();
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
  
  	private double evalIntervals (List<Interval> intervals) {
  		evalIntervals(intervals, Integer.MIN_VALUE);
	}

	private double evalIntervals (List<Interval> intervals, int merged) {
		int i, j;
		int Nj;
		double suma;
		double factor;
		double total = 0;
		
		for (i=0; i<intervals.size(); i++) {
			if (i == merged) {
				Nj = 0;
				for (j=0; j< numClasses; j++) {
					Nj += intervals.get(i).cd[j];
					Nj += intervals.get(i+1).cd[j];
				}
				suma = 0;
				for (j=0; j< numClasses; j++) {
					factor = (intervals.get(i).cd[j] + intervals.get(i+1).cd[j] + lambda) / (Nj + numClasses*lambda);
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
				total += (alpha * ((float)Nj / totalCount) * suma) + ((1 - alpha) * (((float) numClasses * lambda) / Nj));
			}
		}		
		return total;
	}
  
  
  private void initializeLayers(Instance inst){
	  m_DiscretizeCols.setUpper(inst.numAttributes() - 1);
	  numClasses = inst.numClasses();
	  numAttributes = inst.numAttributes();
	  for (int i = 0; i < inst.numAttributes(); i++) {
		  allIntervals[i] = new LinkedList<Interval>();
	  }  
  }
	
}
