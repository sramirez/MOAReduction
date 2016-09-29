package moa.classifiers.competence;

import jcolibri.method.maintenance.TwoStepCaseBaseEditMethod;
import jcolibri.method.maintenance.algorithms.ICFRedundancyRemoval;
import jcolibri.method.maintenance.algorithms.ICFFull;
import jcolibri.method.maintenance.algorithms.RENNNoiseReduction;

/**
 * A case-base maintained by a maintenance method composed of two methods:
 * RENN and ICF.
 *
 * @author Sergio Ramirez (sramirez at decsai dot ugr dot es)
 *
 * @version 0.1
 *
 */
@SuppressWarnings("serial")
public class ICFFullCB extends MaintainedCB {

	@Override
	TwoStepCaseBaseEditMethod getMaintenanceMethod() {
		// TODO Auto-generated method stub
		return new ICFFull(new RENNNoiseReduction(), new ICFRedundancyRemoval());
	}

}
