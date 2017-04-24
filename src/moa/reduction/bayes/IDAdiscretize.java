/* 
** Class for a discretisation filter for instance streams
** Copyright (C) 2016 Germain Forestier, Geoffrey I Webb
**
** This program is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
** 
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
** GNU General Public License for more details.
** 
** You should have received a copy of the GNU General Public License
** along with this program. If not, see <http://www.gnu.org/licenses/>.
**
** Please report any bugs to Sergio Ramírez <sramirez@decsai.ugr.es> and
** Germain Forestier <germain.forestier@uha.fr>
** 
*/
package moa.reduction.bayes;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;

import moa.reduction.core.MOADiscretize;

import com.yahoo.labs.samoa.instances.Instance;

public class IDAdiscretize extends MOADiscretize {
	private static final long serialVersionUID = 1L;
	
	// number of bins for each numerical attributes
	protected int nBins;
	// number of samples (i.e. size of the window)
	protected int sampleSize;
	// number of instances seen so far
	protected int nbSeenInstances;
	// number of attributes
	protected int nbAttributes;
	// number of attributes
	protected int nbNumericalAttributes;
	// sample reservoir, one for each numerical attribute
	protected IntervalHeap[] sReservoirs;
	

	private LinkedList<Float> labels = new LinkedList<Float>();
	
	// type of IDA
	protected IDAType type;
	public enum IDAType {
		IDA,
		IDAW
	}
	
	// has been init
	protected boolean init = false;

	  public IDAdiscretize() {
		  super();
		  setAttributeIndices("first-last");
		  this.type = IDAType.IDAW;
		  this.sampleSize = 1000;
		  this.nBins = 5;
	  }	
	
	/**
	 * Create an IDA filter
	 * @param nBins number of bins
	 * @param sampleSize number of samples
	 * @param window or random
	 */
	public IDAdiscretize(int nBins, int sampleSize, IDAType type) {
		this();
		this.nBins = nBins;
		this.sampleSize = sampleSize;
		this.type = type;
	}
	
	@Override
	public Instance applyDiscretization(Instance inst) {
		  
		  if(init && nbSeenInstances > nBins){
			  int rind = 0;
			  for (int i = 0; i < this.nbAttributes; i++) {
					 // if numeric and not missing, discretize
					 if(inst.attribute(i).isNumeric() && !inst.isMissing(i)) {
						 m_CutPoints[i] = sReservoirs[rind++].getBoundaries();
					 }
				 }
			  return convertInstance(inst);
		  }		  
		  return inst;
	  }
	
	@Override
	public void updateEvaluator(Instance inst) {
		if(!init)
			init(inst);
		// TODO Auto-generated method stub
		nbSeenInstances++;
		
		if(type.equals(IDAType.IDA)) { // random sample
			updateRandomSample(inst); 
		} else if(type.equals(IDAType.IDAW)) { // window sample
			updateWindowSample(inst);
		}

		 if(nbSeenInstances % 101 == 0) 
			 writeCPointsToFile(1, 2, nbSeenInstances, "IDA");
	}
	
	/**
	 * Window sample (IDAW)
	 * @param inst the new instance
	 */
	protected void updateWindowSample(Instance inst) {
		int nbNumericalAttributesCount = 0;
		for (int i = 0; i < this.nbAttributes; i++) {
			double v = inst.value(i);
			// if the value is not missing, then add it to the pool
			if(inst.attribute(i).isNumeric() && !inst.isMissing(i)) {
				this.sReservoirs[nbNumericalAttributesCount].insertWithWindow(v);
				if(!this.sReservoirs[nbNumericalAttributesCount].checkValueInQueues(v)) {
					System.err.println("Value not added.");
				}
				if(labels.size() >= sampleSize)
					labels.poll();
				labels.add((float) inst.classValue());
			}
			if(inst.attribute(i).isNumeric()) { 
				nbNumericalAttributesCount++;
			}
		}
	}
	
	/**
	 * Random sample (IDA)
	 * @param inst the new instance
	 */
	protected void updateRandomSample(Instance inst) {
		int nbNumericalAttributesCount = 0;
		for (int i = 0; i < this.nbAttributes; i++) {
			double v = inst.value(i);
			// if the value is not missing, then add it to the pool
			if(inst.attribute(i).isNumeric() && !inst.isMissing(i)) {
				if(sReservoirs[nbNumericalAttributesCount].getNbSamples() < sampleSize) {
					this.sReservoirs[nbNumericalAttributesCount].insertValue(v);
				} else {
					double rValue = Math.random();
					if(rValue <= (double)sampleSize/(double)nbSeenInstances) {
						int randval = new Random().nextInt(sampleSize);
						this.sReservoirs[nbNumericalAttributesCount].replace(randval,v);
					}
				}
			}
			if(inst.attribute(i).isNumeric()) { 
				nbNumericalAttributesCount++;
			}
		}
	}
	
	/**
	 * Init the stream
	 */
	public void init(Instance inst) {
		this.init = true;
		//generateNewHeader();
		//this.nbAttributes = inst.numAttributes() - 1;
		this.nbAttributes = inst.numAttributes();
		m_DiscretizeCols.setUpper(nbAttributes);	  
		m_CutPoints = new double[inst.numAttributes()][this.nBins];
		// minus one for the class
		//this.nbAttributes = inst.numAttributes() - 1;
		for (int i = 0; i < this.nbAttributes; i++) {
			// if the value is not missing, then add it to the pool
			if(inst.attribute(i).isNumeric() && !inst.isMissing(i)) 
				nbNumericalAttributes++;
		}
		this.sReservoirs =  new IntervalHeap[nbNumericalAttributes];
		for (int i = 0; i < nbNumericalAttributes; i++) {
			sReservoirs[i] = new IntervalHeap(this.nBins, this.sampleSize, i);
		}
	}

	@Override
	public int getAttValGivenClass(int attI, double rVal, int dVal, int classVal) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public Float condProbGivenClass(int attI, double rVal, int dVal,
			int classVal, float classProb) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Float condProbGivenClass(int attI, double rVal, int dVal,
			int classVal, int classCount) {
		// TODO Auto-generated method stub
		return null;
	}
}
