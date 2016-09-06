package moa.classifiers.competence;

import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

import jcolibri.cbrcore.CBRCase;
import jcolibri.exception.InitializingException;
import jcolibri.extensions.classification.ClassificationSolution;
import jcolibri.method.maintenance.CaseResult;
import jcolibri.method.maintenance.CompetenceModel;
import jcolibri.method.maintenance.solvesFunctions.CBESolvesFunction;
import jcolibri.method.retrieve.RetrievalResult;
import jcolibri.method.retrieve.NNretrieval.NNScoringMethod;
import jcolibri.method.retrieve.selection.SelectCases;
import jcolibri.method.reuse.classification.KNNClassificationConfig;
import jcolibri.method.reuse.classification.KNNClassificationMethod;
import jcolibri.method.revise.classification.BasicClassificationOracle;
import jcolibri.method.revise.classification.ClassificationOracle;

/**
 * Provides the ability to run the SRR case base condensation algorithm 
 * on a case base to eliminate redundancy.
 * 
 * @author Sergio Ramírez
 * 02/09/16
 */
public class SRRRedundancyRemoval {
	
	/**
	 * Simulates the RC case base editing algorithm, returning the cases
	 * that would be deleted by the algorithm.
	 * @param cases The group of cases on which to perform editing.
	 * @param simConfig The similarity configuration for these cases.
	 * @return the list of cases that would be deleted by the 
	 * RC algorithm.
	 */
	protected KNNClassificationConfig simConfig;
	protected int sizeLimit = 250;
	
	public SRRRedundancyRemoval(KNNClassificationConfig simConfig, int sizeLimit) {
		// TODO Auto-generated constructor stub
		this.simConfig = simConfig;
		this.sizeLimit = sizeLimit;
	}
	
	@SuppressWarnings("unchecked")
	public Collection<CBRCase> retrieveCasesToDelete(Collection<CBRCase> cases) {
		
    	jcolibri.util.ProgressController.init(this.getClass(),"RC Redundancy Removal",jcolibri.util.ProgressController.UNKNOWN_STEPS);
		List<CBRCase> localCases = new LinkedList<CBRCase>();
		for(CBRCase c: cases)
		{	localCases.add(c);
		}
			
		CompetenceModel sc = new CompetenceModel();		
		sc.computeCompetenceModel(new CBESolvesFunction(), simConfig, localCases);
		
		List<CaseResult> caseReachabilitySetSizes = new LinkedList<CaseResult>();
		for(CBRCase c : localCases){	
			Collection<CBRCase> currSet = null;
			try {	
				currSet = sc.getReachabilitySet(c);
			} catch (InitializingException e) {	
				e.printStackTrace();
			}
			int reachabilitySetSize = 0;

			if(currSet != null) {	
				reachabilitySetSize = currSet.size();
			}
		
			caseReachabilitySetSizes.add(new CaseResult(c, reachabilitySetSize));
			jcolibri.util.ProgressController.step(this.getClass());
		}
		caseReachabilitySetSizes = CaseResult.sortResults(caseReachabilitySetSizes, false);
		
		boolean[] preserved = new boolean[localCases.size()];
		LinkedList<CBRCase> locked = new LinkedList<CBRCase>();
		HashMap<CBRCase, LinkedList<CBRCase>> linked = new HashMap<CBRCase, LinkedList<CBRCase>>();
		List<CBRCase> allCasesToBeRemoved = new LinkedList<CBRCase>();
		
		int nlocked = 0;
		int npreserv = 0;
		int ninserted = localCases.size();
		int ti = 0, i = 0;
		int minLock = (simConfig.getK() + 1) / 2;
		
		do {
			locked.clear();
			while(ninserted - npreserv - nlocked > 0) {
				CBRCase x = caseReachabilitySetSizes.get(i).getCase();
				if(!preserved[i] && !locked.contains(x)) {
					// Check if x and kL of x can be solved without x in CB
					localCases.remove(x); ninserted--;
					allCasesToBeRemoved.add(x);
					
					boolean solvedK = true;
					for(CBRCase l: linked.get(i)) {
						if(!solves(localCases, l)){
							solvedK = false;
							break;
						}
					}
					
					if(solves(localCases, x) && solvedK) { // line 5
						try {
							for(CBRCase l: linked.get(i)) {					
								List<CBRCase> oldLinked = linked.getOrDefault(l, new LinkedList<CBRCase>());
								oldLinked.addAll(getkNN(sc.getReachabilitySet(l), l, minLock));
							}
							Collection<CBRCase> oldLinked = linked.getOrDefault(x, new LinkedList<CBRCase>());
							Collection<CBRCase> xreach = sc.getReachabilitySet(x);
							Collection<CBRCase> minSolve = getkNN(xreach, x, minLock);
							oldLinked.addAll(minSolve);	
							locked.addAll(minSolve); nlocked += minSolve.size();
						} catch (InitializingException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
					} else {
						preserved[i] = true; npreserv++;
						localCases.add(x);
						allCasesToBeRemoved.remove(x); ninserted++;
					}
				}			
				ti++;
				i = ti % caseReachabilitySetSizes.size();
			}			
		} while (ninserted > sizeLimit && locked.size() > 0);
		
		jcolibri.util.ProgressController.finish(this.getClass());
		return allCasesToBeRemoved;
	}
	
	private boolean solves(Collection<CBRCase> casebase, CBRCase query){

		Collection<RetrievalResult> knn = NNScoringMethod.evaluateSimilarity(casebase, query, simConfig);
		knn = SelectCases.selectTopKRR(knn, simConfig.getK());
		try{	
			KNNClassificationMethod classifier = ((KNNClassificationConfig) simConfig).getClassificationMethod();
			ClassificationSolution predictedSolution = classifier.getPredictedSolution(knn);
			ClassificationOracle oracle = new BasicClassificationOracle();
   			
			return oracle.isCorrectPrediction(predictedSolution, query);
		} catch(ClassCastException cce) {	
			cce.printStackTrace();
			System.exit(0);
		}
		return false;
		
	}
	
	private Collection<CBRCase> getkNN(Collection<CBRCase> casebase, CBRCase query, int k) {
		Collection<RetrievalResult> knn = NNScoringMethod.evaluateSimilarity(casebase, query, simConfig);
		return SelectCases.selectTopK(knn, k);
	}
}