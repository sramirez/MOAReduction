package moa.reduction.core;

import weka.core.Instance;

public interface MOAAttributeEvaluator {
	
	  public void updateEvaluator(Instance inst) throws Exception;
	  public void applySelection();
	  public boolean isUpdated();

}
