package moa.classifiers.bayes;

import weka.core.Instance;

public interface MOADiscretize {
	
	  public void updateEvaluator(Instance inst);
	  public Instance applyDiscretization(Instance inst);

}
