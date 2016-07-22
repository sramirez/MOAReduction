package moa.classifiers.bayes;

import weka.core.Instance;

public interface MOAAttributeEvaluator {
	
	  public void updateEvaluator(Instance inst) throws Exception;
	  public void updatePrediction();

}
