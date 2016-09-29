package moa.classifiers.competence;

import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

import jcolibri.cbrcore.CBRCase;
import jcolibri.method.maintenance.TwoStepCaseBaseEditMethod;
import jcolibri.method.retrieve.NNretrieval.similarity.global.Average;
import jcolibri.method.retrieve.NNretrieval.similarity.local.EuclideanDistance;
import jcolibri.method.reuse.classification.KNNClassificationConfig;
import jcolibri.method.reuse.classification.MajorityVotingMethod;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
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
public class NEFCSSRR extends MaintainedCB {

	public IntOption lOption = new IntOption("l", 'l',
            "Number of neighbors used in search.", 10, 1, Integer.MAX_VALUE);
	public FloatOption pminOption = new FloatOption("pmin", 'i',
            "p-min for confidence test in NEFCS", 0.5f, 0.0f, 1.0f);
	public FloatOption pmaxOption = new FloatOption("pman", 'a',
            "p-man for confidence test in NEFCS", 0.5f, 0.0f, 1.0f);
    public IntOption limitOption = new IntOption("limit", 's',
            "Size limit for case-base (used by SRR).", 1000, 1, Integer.MAX_VALUE);
    protected List<CBRCase> newWindow = new LinkedList<CBRCase>();
    protected List<CBRCase> oldWindow = new LinkedList<CBRCase>();	
	private NEFCSNoiseReduction nefcs;
	private SRRRedundancyRemoval srr;
	
	@Override
	TwoStepCaseBaseEditMethod getMaintenanceMethod() {
		// TODO Auto-generated method stub
		return null;
	}
	
	
	public NEFCSSRR() {
		// TODO Auto-generated constructor stub
		super();
        // Configure KNN
   		wekaSimConfig = new KNNClassificationConfig();   		
   		wekaSimConfig.setDescriptionSimFunction(new Average());
   		try {
   			wekaSimConfig.addMapping(new jcolibri.cbrcore.Attribute(
   					"instance", Class.forName("jcolibri.connector.WekaInstanceDescription")), 
   					new EuclideanDistance());
   		} catch (ClassNotFoundException e) {
   			// TODO Auto-generated catch block
   			e.printStackTrace();
   		}			
   		wekaSimConfig.setClassificationMethod(new MajorityVotingMethod());
   		wekaSimConfig.setK(kOption.getValue());
		nefcs = new NEFCSNoiseReduction(wekaSimConfig, lOption.getValue(), periodOption.getValue(), 
				(float) pminOption.getValue(), (float) pmaxOption.getValue());
		srr = new SRRRedundancyRemoval(wekaSimConfig, limitOption.getValue());
	}

    @Override
    public void resetLearningImpl() {
    	oldWindow.addAll(newWindow);
    	newWindow.clear();
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
    	weka.core.Instance winst = new weka.core.DenseInstance(inst.weight(), inst.toDoubleArray());
    	if(initialized)
    		winst.setDataset(emptyDataset);
    	
        //Store immediately the instance in the case-base during the first round  
    	if(index < periodOption.getValue()) {
    		if(!initialized) {
    			initialized = true;
    			initializeDataset(inst); 
            	winst.setDataset(emptyDataset);
            	
            	List<weka.core.Instance> l = new LinkedList<weka.core.Instance>();
            	l.add(winst);
            	initCaseBase(l);	
            	
            } else {            	            	
        		List<CBRCase> l = new LinkedList<CBRCase>();
        		CBRCase ncase = createCase(winst);
            	l.add(ncase);
            	_caseBase.learnCases(l);
        		oldWindow.add(ncase);
        		      		
            }    		
    	} else {
    		newWindow.add(createCase(winst));
    	}
        
    	Collection<CBRCase> nc = new LinkedList<CBRCase>();
        if (index >= periodOption.getValue() && index % periodOption.getValue() == 0) {
        	if(!newWindow.isEmpty()) { // Second window
        		nc = nefcs.applyMaintenance(newWindow, oldWindow, _caseBase.getCases());
        		if(nc.size() > limitOption.getValue()){
        			Collection<CBRCase> toRemove = srr.retrieveCasesToDelete(nc);
                	nc.removeAll(toRemove);
                	nefcs.removeFromDeactivated(toRemove);
        		}        			
            	_caseBase.forgetAllCases();
            	_caseBase.learnCases(nc);
        	}
        		
        	System.out.println("\nNum Cases in the new casebase: " + _caseBase.getCases().size());
    		resetLearningImpl();
        }
        index++;
    }
}