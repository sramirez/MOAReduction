package moa.classifiers.competence;

import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

import jcolibri.cbrcore.CBRCase;
import moa.options.FloatOption;
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
public abstract class NEFCSSRR extends MaintainedCB {

	public IntOption lOption = new IntOption("last predictions", 'l',
            "Number of neighbors used in search.", 10, 1, Integer.MAX_VALUE);
	public FloatOption pminOption = new FloatOption("p-min", 'i',
            "p-min for confidence test in NEFCS", 0.5f, 0.0f, 1.0f);
	public FloatOption pmaxOption = new FloatOption("p-man", 'a',
            "p-man for confidence test in NEFCS", 0.5f, 0.0f, 1.0f);
    public IntOption limitOption = new IntOption("size limit", 's',
            "Size limit for case-base (used by SRR).", 250, 1, Integer.MAX_VALUE);
    protected List<CBRCase> newWindow = new LinkedList<CBRCase>();
    protected List<CBRCase> oldWindow = new LinkedList<CBRCase>();	
	private NEFCSNoiseReduction nefcs;
	private SRRRedundancyRemoval srr;
	
	
	public NEFCSSRR() {
		// TODO Auto-generated constructor stub
		super();
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
    	this.index++;
        
        //Store immediately the instance in the case-base during the first round  
    	if(index < periodOption.getValue()) {
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
    		oldWindow.add(createCase(inst));
    	} else {
    		newWindow.add(createCase(inst));
    	}
        
    	Collection<CBRCase> nc = new LinkedList<CBRCase>();
        if (this.index % this.periodOption.getValue() == 0) {
        	
        	if(!newWindow.isEmpty()) { // Second window
        		nc = nefcs.applyMaintenance(newWindow, oldWindow, _caseBase.getCases());
        		if(nc.size() > limitOption.getValue()){
        			Collection<CBRCase> toRemove = srr.retrieveCasesToDelete(nc);
                	nc.removeAll(toRemove);
        		}        			
            	_caseBase.forgetAllCases();
            	_caseBase.learnCases(nc);
        	}
        		
        	System.out.println("\nNum Cases in the new casebase: " + _caseBase.getCases().size());
    		resetLearningImpl();
        }
    }
}