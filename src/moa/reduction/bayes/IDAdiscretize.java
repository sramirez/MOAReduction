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
	

	private LinkedList<Integer> labels = new LinkedList<Integer>();
	
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
			  writeToFilePartial(3, 4, nbSeenInstances);
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
				labels.add(inst.classIndex());
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
	
	private void writeToFilePartial(int att1, int att2, int iteration){
		  FileWriter data = null;
		  FileWriter cpoints1 = null;
		  FileWriter cpoints2 = null;
			try {
				data = new FileWriter("IDA-data" + "-" + iteration + ".dat");
				cpoints1 = new FileWriter("IDA-cpoints1" + "-" + iteration + ".dat");
				cpoints2 = new FileWriter("IDA-cpoints2" + "-" + iteration + ".dat");
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		  PrintWriter dataout = new PrintWriter(data);
		  PrintWriter cpout1 = new PrintWriter(cpoints1);
		  PrintWriter cpout2 = new PrintWriter(cpoints2);
		  
		  for (int i = 0; i < sReservoirs[att1].windowValues.size(); i++) {
			  dataout.print(sReservoirs[att1].windowValues.get(i) + "," + 
					  sReservoirs[att2].windowValues.get(i) + "," + 
					  labels.get(i) + "\n");
		  }
		  
		  if(m_CutPoints != null && m_CutPoints[att1] != null) {
			  for (int i = 0; i < m_CutPoints[att1].length; i++) {
				  cpout1.println(m_CutPoints[att1][i]);
			  }
		  }
		  
		  if(m_CutPoints != null && m_CutPoints[att2] != null) {
			  for (int i = 0; i < m_CutPoints[att2].length; i++) {
				  cpout2.println(m_CutPoints[att2][i]);
			  }
		  }
		  //Flush the output to the file
		  dataout.flush();
		  cpout1.flush();
		  cpout2.flush();
		       
		   //Close the Print Writer
		  dataout.close();
		  cpout1.close();
		  cpout2.close();
		       
		   //Close the File Writer
		   try {
			data.close();
			cpoints1.close();
			cpoints2.close();
		   } catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		   }    
		  
	  }
}
