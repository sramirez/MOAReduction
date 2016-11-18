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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import com.yahoo.labs.samoa.instances.Instance;

import moa.reduction.core.MOADiscretize;
import weka.core.ContingencyTables;
import weka.core.Range;
import weka.core.Utils;

/**
 <!-- globalinfo-start -->
 * Partition Incremental Discretization (PiD)
 * <br/>
 * For more information, see:<br/>
 * <br/>
 * João Gama and Carlos Pinto. 2006. Discretization from data streams: applications to histograms and data mining. 
 * In Proceedings of the 2006 ACM symposium on Applied computing (SAC '06). ACM, New York, NY, USA, 662-667. DOI=http://dx.doi.org/10.1145/1141277.1141429
 * <br/>
 * @author Sergio Ramírez (sramirez at decsai dot ugr dot es)
 */
public class IDAdiscretize extends MOADiscretize {

	
	protected int nBins = 100;
	
	protected long seed = 517231741;
	
	protected int sampleSize = 0;

	protected long nElems = 0;
	
	protected IntervalHeap[][] bins;
	
	protected boolean windowMode = false;
	
  /** Stores which columns to Discretize */  
  protected Range m_DiscretizeCols = new Range();

  /** Store the current cutpoints */  
  protected List<List<Double>> m_CutPointsL1 = null;
  
  protected List<List<Float>> m_Counts = null;
  
  protected List<List<Map<Integer, Float>>> m_Distrib = null;
  
  protected double[][] m_CutPointsL2 = null;
  
  protected double step;

  /** Output binary attributes for discretized attributes. */
  protected boolean m_MakeBinary = false;

  /** Use bin numbers rather than ranges for discretized attributes. */
  protected boolean m_UseBinNumbers = false;

  /** Use better encoding of split point for MDL. */
  protected boolean m_UseBetterEncoding = false;

  /** Precision for bin range labels */
  protected int m_BinRangePrecision = 6;
  
  protected Random rand;

  /** Constructor - initialises the filter */
  public IDAdiscretize() {
	  setAttributeIndices("first-last");
  }
  
  public IDAdiscretize(long seed, int initialElements, int initialBinsL1, int min, int max, int alpha, int l2UpdateExamples) {
	  this();
	  this.seed = seed;
	  this.rand = new Random(seed);
  }

  
  public Instance applyDiscretization(Instance inst) {
	  
	  if(m_CutPoints != null){
		  updateCP();
		  return convertInstance(inst);
	  }
		  
	  return inst;
  }
  
  
  public void updateEvaluator(Instance instance) {
	  
	  if(m_CutPoints == null) {
		  initialize(instance);
	  }
	  
	  nElems++;
	  
	  if(rand.nextFloat() < sampleSize / nElems) {
		  for (int i = instance.numAttributes() - 1; i >= 0; i--) {
			  if ((m_DiscretizeCols.isInRange(i))
					  && (instance.attribute(i).isNumeric())
					  && (instance.classIndex() != i)) {
				  
				  if(bins[i].length == sampleSize)
					  removeRandomValue(i);
				  insertValue(instance.value(i), i);				  
			  }
		  } 
	  }
  }
  
  
  private void insertValue(double v, int index){
	  IntervalHeap[] intervals = bins[index];
	  int t = intervals.length % nBins;
	  int j = 0;
	  while(j < intervals.length && (double) intervals[j].getMost() < v)
		  j++;
	  intervals[j].add(v);
	  if(j < t){
		  for(int k = j; j < t; k++)
			  intervals[k + 1].add(intervals[k].getMost());
	  } else {
		  for(int k = t; t < j; k++)
			  intervals[k].add(intervals[k + 1].getLeast());
	  }
  }
  
  private void removeRandomValue(int index) {
	  int rind = rand.nextInt(bins.length);
	  IntervalHeap interval = bins[index][rind];
	  rind = rand.nextInt(interval.size);
	  interval.removeElement(rind);
  }
  
  
  private void initialize(Instance inst){
	  m_DiscretizeCols.setUpper(inst.numAttributes() - 1);	  
	  bins = new IntervalHeap[inst.numAttributes()][sampleSize];
	  m_Counts = new ArrayList<List<Float>>(inst.numAttributes());
  }
  
  /**
   * Convert a single instance over. The converted instance is added to the end
   * of the output queue.
   * 
   * @param instance the instance to convert
   */
  protected Instance convertInstance(Instance instance) {

    int index = 0;
    double[] vals = new double[instance.numAttributes()];
    // Copy and convert the values
    for (int i = 0; i < instance.numAttributes(); i++) {
      if (m_DiscretizeCols.isInRange(i)
        && instance.attribute(i).isNumeric()) {
        int j;
        double currentVal = instance.value(i);
        if (m_CutPointsL2[i] == null) {
          if (instance.isMissing(i)) {
            vals[index] = Utils.missingValue();
            instance.setValue(index, Utils.missingValue());
          } else {
            vals[index] = 0;
            instance.setValue(index, 0);
          }
          index++;
        } else {
          if (!m_MakeBinary) {
            if (instance.isMissing(i)) {
              vals[index] = Utils.missingValue();
              instance.setValue(index, Utils.missingValue());
            } else {
              for (j = 0; j < m_CutPointsL2[i].length; j++) {
                if (currentVal <= m_CutPointsL2[i][j]) {
                  break;
                }
              }
              vals[index] = j;
              instance.setValue(index, j);
            }
            index++;
          } else {
            for (j = 0; j < m_CutPointsL2[i].length; j++) {
              if (instance.isMissing(i)) {
                vals[index] = Utils.missingValue();
                instance.setValue(index, Utils.missingValue());
              } else if (currentVal <= m_CutPointsL2[i][j]) {
                vals[index] = 0;
                instance.setValue(index, 0);
              } else {
                vals[index] = 1;
                instance.setValue(index, 1);
              }
              index++;
            }
          }
        }
      } else {
        vals[index] = instance.value(i);
        index++;
      }
    }
    
    return(instance);
  }
  
	@Override
	public int getNumberIntervals() {
		// TODO Auto-generated method stub
		if(m_CutPointsL2 != null) {
			int ni = 0;
			for(double[] cp: m_CutPointsL2){
				if(cp != null)
					ni += (cp.length + 1);
			}
			return ni;	
		}
		return 0;
	}


}
