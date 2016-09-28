package moa.classifiers;

import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

import jcolibri.cbrcore.CBRCase;
import jcolibri.exception.InitializingException;
import jcolibri.extensions.classification.ClassificationSolution;
import jcolibri.method.maintenance.CaseResult;
import jcolibri.method.maintenance.CompetenceModel;
import jcolibri.method.maintenance.solvesFunctions.ICFSolvesFunction;
import jcolibri.method.retrieve.RetrievalResult;
import jcolibri.method.retrieve.NNretrieval.NNScoringMethod;
import jcolibri.method.retrieve.selection.SelectCases;
import jcolibri.method.reuse.classification.KNNClassificationConfig;
import jcolibri.method.reuse.classification.KNNClassificationMethod;
import jcolibri.method.revise.classification.BasicClassificationOracle;
import jcolibri.method.revise.classification.ClassificationOracle;

/**
 * Provides the ability to run the Stepwise Redundancy Removal (SRR) algorithm 
 * on a case base to reduce redundancy.
 * 
 * @author Sergio Ramírez
 * 02/09/16
 */
public class SRRRedundancyRemoval {
	
	protected KNNClassificationConfig simConfig;
	protected int sizeLimit;
	protected int k;
	
	public SRRRedundancyRemoval(KNNClassificationConfig simConfig, int sizeLimit) {
		// TODO Auto-generated constructor stub
		this.simConfig = simConfig;
		this.sizeLimit = sizeLimit;
		this.k = simConfig.getK();
	}
	
	@SuppressWarnings("unchecked")
	/**
	 * Apply SRR on the current casebase. Only redundant models are removed from CB,
	 * competence model does not need to be rebuilt.
	 * @param cases the current model/CB.
	 * @return The new casebase.
	 */
	public Collection<CBRCase> retrieveCasesToDelete(Collection<CBRCase> cases) {
		
    	List<CBRCase> localCases = new LinkedList<CBRCase>();
		for(CBRCase c: cases) {	
			localCases.add(c);
		}
			
		CompetenceModel sc = new CompetenceModel();		
		sc.computeCompetenceModel(new ICFSolvesFunction(), simConfig, localCases);
		simConfig.setK(this.k); // Important!
		
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
		HashMap<CBRCase, List<CBRCase>> linked = new HashMap<CBRCase, List<CBRCase>>();
		List<CBRCase> allCasesToBeRemoved = new LinkedList<CBRCase>();
		
		int npreserv = 0;
		int ninserted = localCases.size();
		int ti = 0, i = 0;
		int minLock = (simConfig.getK() + 1) / 2;
		
		do {
			locked.clear();
			while(ninserted - npreserv - locked.size() > 0) {
				CBRCase x = caseReachabilitySetSizes.get(i).getCase();
				if(!preserved[i] && !locked.contains(x)) {
					// Check if x and kL of x can be solved without x in CB
					localCases.remove(x); ninserted--;
					allCasesToBeRemoved.add(x);
					
					boolean solvedK = true;
					for(CBRCase l: linked.getOrDefault(i, new LinkedList<CBRCase>())) {
						if(!solves(localCases, l)){
							solvedK = false;
							break;
						}
					}
					
					if(solves(localCases, x) && solvedK) { // line 5
						try {
							for(CBRCase l: linked.getOrDefault(i, new LinkedList<CBRCase>())) {					
								List<CBRCase> oldLinked = linked.getOrDefault(l, new LinkedList<CBRCase>());
								oldLinked.addAll(getkNN(sc.getReachabilitySet(l), l, minLock));
								linked.put(l, oldLinked);
							}
							List<CBRCase> oldLinked = linked.getOrDefault(x, new LinkedList<CBRCase>());
							Collection<CBRCase> minSolveX = getkNN(sc.getReachabilitySet(x), x, minLock);
							oldLinked.addAll(minSolveX);
							linked.put(x, oldLinked);
							locked.addAll(minSolveX);
						} catch (InitializingException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
					} else {
						preserved[i] = true; npreserv++;
						localCases.add(x); ninserted++;
						allCasesToBeRemoved.remove(x);
					}
				}			
				ti++;
				i = ti % caseReachabilitySetSizes.size();
			}			
		} while (ninserted > sizeLimit && locked.size() > 0);
		
		System.out.println("\nNum of cases removed by SRR: " + allCasesToBeRemoved.size());
		return allCasesToBeRemoved;
	}
	
	/**
	 * Evaluates if query element is solved by the current casebase/model.
	 * The element is considered to be solved if the majority voting of k-neighbors agrees
	 * with the class of query.
	 * @param casebase model to be considered
	 * @param query element to be queried
	 * @return 
	 */
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
	
	private List<CBRCase> getkNN(Collection<CBRCase> casebase, CBRCase query, int k) {
		List<CBRCase> neighbors = new LinkedList<CBRCase>();
		if(casebase != null) {
			Collection<RetrievalResult> knn = NNScoringMethod.evaluateSimilarity(casebase, query, simConfig);
			neighbors = new LinkedList<CBRCase>(SelectCases.selectTopK(knn, k));
		}
		return neighbors;
	}
}