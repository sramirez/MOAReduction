package moa.reduction.core;

import com.yahoo.labs.samoa.instances.Instance;

public interface MOAAttributeEvaluator {
	
	  public void updateEvaluator(Instance inst) throws Exception;
	  public void applySelection();
	  public boolean isUpdated();

}
