package moa.classifiers.competence;

import jcolibri.method.maintenance.TwoStepCaseBaseEditMethod;
import jcolibri.method.maintenance.algorithms.BBNRNoiseReduction;
import jcolibri.method.maintenance.algorithms.CRRFull;
import jcolibri.method.maintenance.algorithms.CRRRedundancyRemoval;

/**
 * A case-base maintained by a maintenance method composed of two methods:
 * BBNR and CRR.
 *
 * @author Sergio Ramirez (sramirez at decsai dot ugr dot es)
 *
 * @version 0.1
 *
 */
@SuppressWarnings("serial")
public class NEFCSFullCB extends MaintainedCB {

	@Override
	TwoStepCaseBaseEditMethod getMaintenanceMethod() {
		// TODO Auto-generated method stub
		return new CRRFull(new BBNRNoiseReduction(), new CRRRedundancyRemoval());
	}

}
