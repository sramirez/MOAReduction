package moa.classifiers.competence;

import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

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
import moa.options.IntOption;
import weka.core.Instance;

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
public abstract class MaintainedCB extends AbstractClassifier {

	public IntOption kOption = new IntOption("neighbors", 'k',
            "Number of neighbors used in search.", 5, 1, Integer.MAX_VALUE);
    public IntOption periodOption = new IntOption("period", 'p',
            "Size of the environments.", 100, 1, Integer.MAX_VALUE);

    protected long index;
    protected boolean initialized = false;
    protected KNNClassificationConfig wekaSimConfig;
    TwoStepCaseBaseEditMethod maintenanceMethod;
    
	WekaConnector _connector;
	CBRCaseBase _caseBase;
    
    public void initCaseBase(List<Instance> init){
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
    
    protected CBRCase createCase(Instance instance) {	
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
    
    abstract TwoStepCaseBaseEditMethod getMaintenanceMethod();

    @Override
    public void trainOnInstanceImpl(Instance inst) {
    	this.index++;
        
        //Store immediately the instance in the case-base    	
        if(initialized) {
        	List<CBRCase> l = new LinkedList<CBRCase>();
        	l.add(createCase(inst));
        	_caseBase.learnCases(l);
        } else {
        	initialized = true;
        	List<Instance> l = new LinkedList<Instance>();
        	l.add(inst);
        	initCaseBase(l);
        }

        if (this.index % this.periodOption.getValue() == 0) {
        	resetLearningImpl();

        	TwoStepCaseBaseEditMethod maintenance = getMaintenanceMethod();
    		Collection<CBRCase> deleted = maintenance.retrieveCasesToDelete(_caseBase.getCases(), wekaSimConfig);		
    		System.out.println("\nNum Cases deleted by Alg: " + deleted.size());
    		//System.out.println("Cases deleted by Alg: ");
    		/*for(CBRCase c: deleted){	
    			System.out.println(c.getID());
    		}*/    		
    		_caseBase.forgetCases(deleted);           
        }
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
			CBRCase query = createCase(instance);
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