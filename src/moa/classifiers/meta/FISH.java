package moa.classifiers.meta;

import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;

import moa.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import moa.classifiers.lazy.neighboursearch.EuclideanDistance;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.options.*;

/**
 * Learning in non-stationary environments.
 *
 * <p>
 * Ryan Elwell and Robi Polikar. Incremental learning of concept drift in
 * non-stationary environments. IEEE Transactions on Neural Networks,
 * 22(10):1517-1531, October 2011. ISSN 1045-9227. URL
 * http://dx.doi.org/10.1109/TNN.2011.2160459.
 * </p>
 *
 * @author Paulo Goncalves (paulomgj at gmail dot com)
 * @author Dariusz Brzezinski
 *
 * @version 0.4 (Corrected instance weights in classifier training)
 *
 */
public class FISH extends AbstractClassifier {

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "bayes.NaiveBayes");

    public IntOption periodOption = new IntOption("period", 'p',
            "Size of the environments.", 200, 1, Integer.MAX_VALUE);

    public FloatOption distancePropOption = new FloatOption(
            "distanceProportion",
            'd',
            "Distance proportion between space and time varibles.",
            0.5, 0, 1);
    public IntOption kOption = new IntOption("k", 'k',
            "Number of neighbors.", 3, 1, 100);

    protected List<Classifier> ensemble;
    protected List<Double> ensembleWeights;
    protected List<ArrayList<Double>> bkts, wkts;
    protected List<STInstance> buffer;
    protected long index;
    protected double distanceProp;
    protected int pruning, k, period;
  

    @Override
    public void resetLearningImpl() {
        this.ensemble = new ArrayList<>();
        this.ensembleWeights = new ArrayList<>();
        this.bkts = new ArrayList<>();
        this.wkts = new ArrayList<>();
        this.index = 0;
        this.buffer = null;
        this.period = this.periodOption.getValue();
        this.distanceProp = this.distancePropOption.getValue();
        this.k = this.kOption.getValue();
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {

        // Store instance in the buffer
        if (this.buffer == null) {
            this.buffer = new LinkedList<STInstance>();
        }
        this.buffer.add(new STInstance(inst, index));

        if (this.index > this.periodOption.getValue()) {
            //this.index = 0;
            double mt = this.buffer.size();
            Classifier classifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption));
            
            // Compute distances (space/time)
            //double dist[] = new double[buffer.size()];
            EuclideanDistance distancer = new EuclideanDistance();
            
        	List<STInstance> ordered = new LinkedList<STInstance>();
            for(STInstance sti : buffer) {            	
            	double distance = distancer.distance(sti.getInstance(), inst) + 
            			distanceProp * sti.getTime();
            	sti.setDistance(distance);
            }
            
            Collections.sort(ordered);
            
            for (int i = k; i < index + 1 ; i++){
            	List<STInstance> tra = ordered.subList(0, i);
            	//train on tra
            	Instances trai = new Instances(tra.get(0).getInstance().dataset());
            	for(int z = 1; z < tra.size(); z++) trai.add(tra.get(z).getInstance());
            	classifier.buildClassifier(trai);
            	
            	double errors = 0;
            	for(int j = 0; j < k ; j++) {
            		List<STInstance> val = ordered.subList(0, k);
                	STInstance removed = val.remove(j);
                	//validate on val  
                	for(STInstance vinst : val) {
                		if (classifier.classifyInstance(vinst.getInstance()) != vinst.getInstance().classIndex())
                			errors++;
                	}                		
                	val.add(j, removed);
            	}
            	double error = errors / k;
            }
        }
        
        this.index++;
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        DoubleVector combinedVote = new DoubleVector();
        if (this.trainingWeightSeenByModel > 0.0) {
            for (int i = 0; i < this.ensemble.size(); i++) {
                if (this.ensembleWeights.get(i) > 0.0) {
                    DoubleVector vote = new DoubleVector(this.ensemble.get(i)
                            .getVotesForInstance(inst));
                    if (vote.sumOfValues() > 0.0) {
                        vote.normalize();
                        vote.scaleValues(this.ensembleWeights.get(i));
                        combinedVote.addValues(vote);
                    }
                }
            }
        }
        return combinedVote.getArrayRef();
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