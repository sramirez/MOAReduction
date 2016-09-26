package moa.reduction.core;

import com.yahoo.labs.samoa.instances.Instance;


public interface MOADiscretize {
	
	  public void updateEvaluator(Instance inst);
	  public Instance applyDiscretization(Instance inst);

}
