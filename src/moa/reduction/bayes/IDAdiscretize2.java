/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    Discretize.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package moa.reduction.bayes;

import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.TreeSet;

import moa.reduction.core.MOADiscretize;
import weka.core.Range;

import com.yahoo.labs.samoa.instances.Instance;

/**
 <!-- globalinfo-start -->
 * Incremental Discretization Algorithm (IDA)
 * <br/>
 * For more information, see:<br/>
 * <br/>
 * Geoffrey I. Webb. 2014. Contrary to Popular Belief Incremental Discretization can be Sound, 
 * Computationally Efficient and Extremely Useful for Streaming Data. 
 * In Proceedings of the 2014 IEEE International Conference on Data Mining (ICDM '14). 
 * DOI=http://dx.doi.org/10.1109/ICDM.2014.123 
 * <br/>
 * @author Sergio Ramírez (sramirez at decsai dot ugr dot es)
 */
public class IDAdiscretize2 extends MOADiscretize {

	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	protected long seed = 517231741;

	protected long nElems = 0;
	
	protected IntervalHeap[][] bins;
	
	private int nBins;
	
	private TreeSet<Double>[] originalCP;
  
  protected Random rand;

  private int sampleSize;
  
  private boolean initialized = false;
  
  private int contInit = 0;

  /** Constructor - initialises the filter */
  public IDAdiscretize2() {
	  super();
	  setAttributeIndices("first-last");
	  this.rand = new Random();
	  this.sampleSize = 1000;
	  this.nBins = 100;
  }
  
  public IDAdiscretize2(long seed, int nBins, int sampleSize, boolean windowMode) {
	  this();
	  this.seed = seed;
	  this.nBins = nBins;
	  this.rand = new Random(seed);
	  this.sampleSize = sampleSize;
  }
  
  public Instance applyDiscretization(Instance inst) {
	  
	  if(initialized){
		  shiftLimits();
		  return convertInstance(inst);
	  }		  
	  return inst;
  }  
  
  public void updateEvaluator(Instance instance) {
	  
	  if(m_CutPoints == null) {
		  initialize(instance);
	  }
	  
	  nElems++;
	  
	  boolean insertion = rand.nextFloat() < ((float) sampleSize / nElems);
	  for (int i = instance.numAttributes() - 1; i >= 0; i--) {
		  if ((m_DiscretizeCols.isInRange(i))
				  && (instance.attribute(i).isNumeric())
				  && (instance.classIndex() != i)) {			  
			  
			  if(!isPreparedToInitialize(instance)){
				  originalCP[i].add(instance.value(i));				  
			  } else if (initialized) {
				  if(insertion)
					  update(instance.value(i), i, nElems > sampleSize); // replacement
			  } else {
				  initializeBins();
			  }				  
		  }
	  } 	  
  }
  
  private boolean isPreparedToInitialize(Instance instance){
	  for (int j = 0; j < originalCP.length; j++) {
		  if ((m_DiscretizeCols.isInRange(j))
				  && (instance.attribute(j).isNumeric())
				  && (instance.classIndex() != j)) {
			  if(originalCP[j].size() < nBins)
				  return false;
		  }
	  }
	  return true;
  }
  
  private void initializeBins(){
	  for (int i = 0; i < bins.length; i++) {
	    Double[] result = originalCP[i].toArray(new Double[originalCP[i].size()]);
		for (int j = 0; j < bins[i].length && result.length > 0; j++) {
			bins[i][j].add(result[j]);
		}
	  }
	  initialized = true;
  }
  
  // Update the limits according to the last movements
  private void shiftLimits(){
	  if(initialized)  {
		  // Transform intervals in list to a matrix
		  for (int i = 0; i < bins.length; i++) {
			  LinkedHashSet<Double> uniquePoints = new LinkedHashSet<Double>();
			  for (int j = 0; j < bins[i].length && bins[i][j].size() > 0; j++){				  
				 uniquePoints.add((double) bins[i][j].inspectMost());
			  }	
			  
			  m_CutPoints[i] = new double[uniquePoints.size()];
			  int j = 0;
			  for (Iterator<Double> iterator = uniquePoints.iterator(); iterator
					.hasNext();) {
				m_CutPoints[i][j] = (double) iterator.next();
				j++;
			  }
			  
		  } 			  
	  }	
  }
  
  private void update(double v, int index, boolean replacement) {
	  
	  // Look for the appropiate bin using binary search
	  double[] cutpoints = new double[bins[index].length];
	  for (int i = 0; i < bins[index].length; i++) {
		  cutpoints[i] = (double) bins[index][i].inspectMost();
	  }
	  int j = java.util.Arrays.binarySearch(cutpoints, v);
	  if(j < 0 || j == cutpoints.length){
		  j = Math.abs(j) - 1;
	  }
	  
	  // Randomly delete an element from the pool of interval heaps
	  /*if(replacement){
		  int rind = rand.nextInt(bins.length);
		  IntervalHeap2 interval = bins[index][rind];
		  int eind = rand.nextInt(interval.size);
		  interval.removeElement(eind);
	  }*/
	  
	  bins[index][j].add(v);
	  
	  int t = findTargetBin(j, index);
	  

	  // Insert the element into the appropiate bin
	  /*System.out.println("Index: " + index);
	  String str = "";
	  for (int i = 0; i < bins[index].length; i++) {
		str += bins[index][i].size() + "|";
		
	  }
	  System.out.println(str);*/
	  
	  
	  if(j < t){
		  for(int k = j; k < t; k++)
			  bins[index][k + 1].add(bins[index][k].getMost());
	  } else {
		  for(int k = t; k < j; k++)
			  bins[index][k].add(bins[index][k + 1].getLeast());
	  }
	  
	  /*System.out.println("Index: " + index);
	  String str2 = "";
	  for (int i = 0; i < bins[index].length; i++) {
		str2 += bins[index][i].size() + "|";
		
	  }
	  System.out.println(str2);*/
  }
  
  private int findTargetBin(int pivot, int index){
	  // Search to find the next bin that should increase in size
	  int minri = -1; 
	  int minrs = bins[index][pivot].size(); 		  
	  for (int i = pivot + 1; i < bins[index].length; i++) {
		  int size = bins[index][i].size();
		  if(size < minrs - 1){
			  minri = i;
			  minrs = size;
		  }		
	  }
	  
	  int minli = -1; 
	  int minls = bins[index][pivot].size(); 		  
	  for (int i = pivot - 1; i >= 0; i--) {
		  int size = bins[index][i].size();
		  if(size < minls - 1){
			  minli = i;
			  minls = size;
		  }	
	  }
	  
	  if(minri + minli == -2){
		  return pivot;
	  } else if (minrs < minls) {
		  return minri;
	  } else if(minrs > minls) {
		  return minli;
	  } else {
		  return Math.abs(minri - pivot) < Math.abs(minli - pivot) ? minri : minli ;
	  }
  }
  
  private void initialize(Instance inst){
	  m_DiscretizeCols.setUpper(inst.numAttributes() - 1);	  
	  bins = new IntervalHeap[inst.numAttributes()][nBins];	  
	  m_CutPoints = new double[inst.numAttributes()][nBins];
	  originalCP = new TreeSet[inst.numAttributes()];
	  for (int i = 0; i < inst.numAttributes(); i++){
		  originalCP[i] = new TreeSet<Double>();
		  for (int j = 0; j < bins[i].length; j++) {
			  bins[i][j] = new IntervalHeap();
		}		  
	  }		
  }
  

}
