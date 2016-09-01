package moa.classifiers.meta;

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;
import moa.options.ClassOption;
import moa.options.FloatOption;
import moa.options.IntOption;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Learning in non-stationary environments.
 *
 * <p>
 * I. Žliobaite. Combining similarity in time and space for training set formation under concept drift, 
 * Intell. Data Anal. 15 (4) (2011) 589–611.
 * </p>
 *
 * @author Sergio Ramírez (sramirez at decsai dot ugr dot es)
 *
 */
@SuppressWarnings("serial")
public class FISH extends AbstractClassifier {

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "bayes.NaiveBayes");

    public IntOption periodOption = new IntOption("period", 'p',
            "Size of the environments.", 200, 100, Integer.MAX_VALUE);

    public FloatOption distancePropOption = new FloatOption(
            "distanceProportion",
            'd',
            "Distance proportion between space and time varibles.",
            1, Float.MIN_VALUE, Float.MAX_VALUE);
    public IntOption kOption = new IntOption("k", 'k',
            "Number of neighbors.", 3, 1, 50);

    protected Classifier testClassifier;
    protected List<STInstance> buffer;
    protected long index = 0;
    protected double distanceProp;
    protected int pruning, k, period;
  

    @Override
    public void resetLearningImpl() {
        this.index = 0;
        this.buffer = null;
        this.period = this.periodOption.getValue();
        this.distanceProp = this.distancePropOption.getValue();
        this.k = this.kOption.getValue();
        this.testClassifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption));
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {

        // Store instance in the buffer
        if (this.buffer == null) {
            this.buffer = new LinkedList<STInstance>();
        }

    	this.buffer.add(new STInstance(inst, index));
        if (this.index < this.periodOption.getValue()) { 
        	trainClassifier(testClassifier, buffer);
        }       
        
        this.index++;
    }
    
    private void trainClassifier(Classifier c, List<STInstance> instances) {
    	Instances trai = new Instances(instances.get(0).getInstance().dataset());
    	for(int z = 1; z < instances.size(); z++) trai.add(instances.get(z).getInstance());
		try {
			c.buildClassifier(trai);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
    
    private double euclideanDistance(Instance i1, Instance i2) throws Exception {
    	if(i1.numAttributes() != i2.numAttributes())
			throw new Exception("Euclidean distance performed on uneven instances");
		
		double accum = 0;
		for(int i = 0; i < i1.numAttributes(); i++) {
			if(i != i1.classIndex()) {
				accum += Math.pow(i1.value(i) - i2.value(i), 2);
			}
		}
		
		return Math.sqrt(accum);
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
    	
    	if (this.index >= this.periodOption.getValue()) {
        	// The specified size is gotten, now the online process is started
            Classifier classifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption));
            
            // Compute distances (space/time)
            //EuclideanDistance distancer = new EuclideanDistance();            
        	for(STInstance sti : buffer) {            	
            	double distance = Double.MIN_VALUE;
				try {
					distance = euclideanDistance(sti.getInstance(), inst) + 
							distanceProp * sti.getTime();
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
            	sti.setDistance(distance);
            }
            // Sort instances by time/space distance (ascending)
            Collections.sort(buffer);
            
            List<Double> measurements = new LinkedList<Double>();
			List<STInstance> val = buffer.subList(0, k);
            // Apply leave-one-out cross validation (validation set formed by top-k nearest neighbors)
            for (int i = k; i < buffer.size(); i++){
            	try {
	            	List<STInstance> tra = buffer.subList(0, i);
					Instances instancestr = new Instances(tra.get(0).getInstance().dataset());
            		for(int z = 1; z < tra.size(); z++) {
            			instancestr.add(tra.get(z).getInstance());
            		}
            		
	            	double errors = 0;
	            	for(int j = 0; j < k ; j++) {
		            	// Validation instance is removed from the training set for this run
	            		Instance removed = instancestr.remove(j);
	            		// Train for the tuple i, j
	            		classifier.buildClassifier(instancestr);
		            	
	                	//validate on instance j (from val set)
	            		int pred = (int) classifier.classifyInstance(val.get(j).getInstance());
	            		if (pred != val.get(j).getInstance().classValue()) 
							errors++;
	                	
	                	// We add the skipped instance again
	                	instancestr.add(j, removed);
	            	}
                	measurements.add(errors / k);
	            	
            	} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
            }           
            
            int min = 0;
            double minval = Double.MAX_VALUE;
            for(int i = 0; i < measurements.size(); i++) {
            	if(measurements.get(i) < minval) {
            		min = i;
            		minval = measurements.get(i);
            	}
            }
            //System.out.println("New chosen size: " + (min + k));
            List<STInstance> lastTrain = buffer.subList(0, min + k);
            trainClassifier(testClassifier, lastTrain);
            //buffer = lastTrain;
    	}
    	
    	int pred = 0;
		try {
			pred = (int) testClassifier.classifyInstance(inst);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	double[] v = new double[pred + 1];
    	v[pred] = 1.0;
    	return v;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }
    
    class STInstance implements Comparable<STInstance>{
    	private Instance instance;
    	private double distance;
    	private long time;
    	
    	public STInstance(Instance instance, long time) {
			// TODO Auto-generated constructor stub
    		this.instance = instance;
    		this.time = time;
    		distance = Double.MIN_VALUE;
		}
    	
    	public double getDistance() {
			return distance;
		}
    	
    	public void setDistance(double distance) {
			this.distance = distance;
		}
    	
    	public Instance getInstance() {
			return instance;
		}
    	
    	public void setInstance(Instance instance) {
			this.instance = instance;
		}
    	
    	public long getTime() {
			return time;
		}
    	
    	public int compareTo(STInstance other){
	       // your logic here
		   if(this.distance < other.getDistance()){
		        return -1;
		    }else if(this.distance > other.getDistance()){
		        return 1;
		    }		   
		    return 0;
    	}
	 
    }
}