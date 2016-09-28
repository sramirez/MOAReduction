package moa.classifiers;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

import com.github.javacliparser.IntOption;

import jcolibri.casebase.CachedLinealCaseBase;
import jcolibri.cbrcore.Attribute;
import jcolibri.cbrcore.CBRCase;
import jcolibri.cbrcore.CBRCaseBase;
import jcolibri.cbrcore.CaseComponent;
import jcolibri.connector.WekaConnector;
import jcolibri.exception.InitializingException;
import jcolibri.extensions.classification.ClassificationSolution;
import jcolibri.method.maintenance.TwoStepCaseBaseEditMethod;
import jcolibri.method.retrieve.RetrievalResult;
import jcolibri.method.retrieve.NNretrieval.NNScoringMethod;
import jcolibri.method.retrieve.NNretrieval.similarity.global.Average;
import jcolibri.method.retrieve.NNretrieval.similarity.local.EuclideanDistance;
import jcolibri.method.retrieve.selection.SelectCases;
import jcolibri.method.reuse.classification.KNNClassificationConfig;
import jcolibri.method.reuse.classification.KNNClassificationMethod;
import jcolibri.method.reuse.classification.MajorityVotingMethod;
import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;

import com.yahoo.labs.samoa.instances.Instance;


/**
 * A case-base maintained by a maintenance method composed of two methods:
 * one for noise removal and another for redundancy reduction.
 *
 * @author Sergio Ramirez (sramirez at decsai dot ugr dot es)
 *
 * @version 0.1
 *
 */
@SuppressWarnings("serial")
public class MaintainedCB extends AbstractClassifier {

	public IntOption kOption = new IntOption("neighbors", 'k',
            "Number of neighbors used in search.", 3, 1, Integer.MAX_VALUE);
    public IntOption periodOption = new IntOption("period", 'p',
            "Size of the environments.", 500, 1, Integer.MAX_VALUE);

    protected long index = 0;
    protected boolean initialized = false;
    protected KNNClassificationConfig wekaSimConfig;
    TwoStepCaseBaseEditMethod maintenanceMethod;
    
	WekaConnector _connector;
	CBRCaseBase _caseBase;
	protected weka.core.Instances emptyDataset;
    
    public void initCaseBase(List<weka.core.Instance> init){
		try {
			_connector = new WekaConnector(init);
			_caseBase  = new CachedLinealCaseBase();
			_caseBase.init(_connector);
			_connector.retrieveAllCases();
		} catch (InitializingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
		
        // Configure KNN
   		wekaSimConfig = new KNNClassificationConfig();   		
   		wekaSimConfig.setDescriptionSimFunction(new Average());
   		try {
   			wekaSimConfig.addMapping(new Attribute("instance", Class.forName("jcolibri.connector.WekaInstanceDescription")), 
   					new EuclideanDistance());
   		} catch (ClassNotFoundException e) {
   			// TODO Auto-generated catch block
   			e.printStackTrace();
   		}			
   		wekaSimConfig.setClassificationMethod(new MajorityVotingMethod());
   		wekaSimConfig.setK(kOption.getValue());
    }

    @Override
    public void resetLearningImpl() {
        this.index = 0;
    }
    
    protected CBRCase createCase(weka.core.Instance instance) {	
    	CBRCase _case = new CBRCase();
		try {			
			CaseComponent description = (CaseComponent)_connector.descriptionClass.newInstance();
			Attribute idAttribute = description.getIdAttribute();
			idAttribute.setValue(description, index + "");
			Attribute att = new Attribute("instance", _connector.descriptionClass);
			att.setValue(description, instance);
			_case.setDescription(description);
			
			CaseComponent solution = (CaseComponent) _connector.solutionClass.newInstance();
			att = new Attribute("type", _connector.solutionClass);
			att.setValue(solution, instance.classValue());
			_case.setSolution(solution);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return(_case); 
    }
    
    TwoStepCaseBaseEditMethod getMaintenanceMethod() {
		return null;
	}

    protected void initializeDataset(Instance inst) {
    	ArrayList<weka.core.Attribute> list = new ArrayList<weka.core.Attribute>();
		for(int i = 0; i < inst.numAttributes(); i++) 
			list.add(new weka.core.Attribute("att" + i, i));
	  	emptyDataset = new weka.core.Instances ("single", list, 1);
	  	emptyDataset.setClassIndex(inst.classIndex());  
    }
    
    @Override
    public void trainOnInstanceImpl(Instance inst) {
    	weka.core.Instance winst = new weka.core.DenseInstance(inst.weight(), inst.toDoubleArray());
    	
        //Store immediately the instance in the case-base    	
        if(initialized) {
        	List<CBRCase> l = new LinkedList<CBRCase>();
        	winst.setDataset(emptyDataset);
        	l.add(createCase(winst));
        	_caseBase.learnCases(l);
        } else {
        	initialized = true;
        	initializeDataset(inst);
        	winst.setDataset(emptyDataset);
        	
        	List<weka.core.Instance> l = new LinkedList<weka.core.Instance>();
        	l.add(winst);
        	initCaseBase(l);
        }

        if (index >= periodOption.getValue() && index % periodOption.getValue() == 0) {
        	System.out.println("Tamaño de casebase: " + _caseBase.getCases().size());
        	TwoStepCaseBaseEditMethod maintenance = getMaintenanceMethod();
    		Collection<CBRCase> deleted = maintenance.retrieveCasesToDelete(_caseBase.getCases(), wekaSimConfig);		
    		System.out.println("\nNum Cases deleted by Alg: " + deleted.size());
    		//System.out.println("Cases deleted by Alg: ");
    		/*for(CBRCase c: deleted){	
    			System.out.println(c.getID());
    		}*/    		
    		_caseBase.forgetCases(deleted);
        }
        index++;
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }
    
    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
	public double[] getVotesForInstance(Instance instance) {
		Double solution = new Double(0);
		if(initialized){
			weka.core.Instance winst = new weka.core.DenseInstance(instance.weight(), instance.toDoubleArray());
			winst.setDataset(emptyDataset);
			CBRCase query = createCase(winst);
			//BasicClassificationOracle oracle = new BasicClassificationOracle();
			Collection<CBRCase> cases = _caseBase.getCases();
			Collection<RetrievalResult> knn = NNScoringMethod.evaluateSimilarity(cases, query, wekaSimConfig);
			knn = SelectCases.selectTopKRR(knn, wekaSimConfig.getK());
			KNNClassificationMethod classifier = wekaSimConfig.getClassificationMethod();
			ClassificationSolution predictedSolution = classifier.getPredictedSolution(knn);
			solution = (Double) predictedSolution.getClassification();
		}

		double weights[] = new double[solution.intValue() + 1];
		weights[solution.intValue()] = 1.0;
		return weights;
	}
}