package moa.reduction.core;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import moa.core.Example;
import moa.core.FastVector;
import moa.core.InstanceExample;
import moa.streams.InstanceStream;
import moa.streams.filters.AbstractStreamFilter;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import weka.core.Range;
import weka.core.Utils;

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import com.yahoo.labs.samoa.instances.SparseInstance;


public abstract class MOADiscretize extends AbstractStreamFilter {

	/** for serialization */
	static final long serialVersionUID = -3141006402280129097L;
	
	/** Stores which columns to Discretize */
	protected Range m_DiscretizeCols = new Range();
	
	/** Store the current cutpoints */
	protected double[][] m_CutPoints = null;
	
	protected String[][] m_Labels = null;
	
	/** Precision for bin range labels */
	protected int m_BinRangePrecision = 6;
	
	/** new header with discretized attributes */
	protected InstancesHeader discretizedHeader;
	
	// number of instances seen so far
	protected int nbSeenInstances;
	
	// number of attributes
	protected int nbAttributes;
	

	// has been init
	protected boolean init = false;
	
	/** Constructor - initialises the filter */
	public MOADiscretize() {	
	  setAttributeIndices("first-last");
	}

	/**
	 * Set the precision for bin boundaries. Only affects the boundary values used
	 * in the labels for the converted attributes; internal cutpoints are at full
	 * double precision.
	 * 
	 * @param p the precision for bin boundaries
	 */
	public void setBinRangePrecision(int p) {
	  m_BinRangePrecision = p;
	}
	
	/**
	 * Get the precision for bin boundaries. Only affects the boundary values used
	 * in the labels for the converted attributes; internal cutpoints are at full
	 * double precision.
	 * 
	 * @return the precision for bin boundaries
	 */
	public int getBinRangePrecision() {
	  return m_BinRangePrecision;
	}
	
	/**
	 * Gets whether the supplied columns are to be removed or kept
	 * 
	 * @return true if the supplied columns will be kept
	 */
	public boolean getInvertSelection() {
	
	  return m_DiscretizeCols.getInvert();
	}
	
	/**
	 * Sets whether selected columns should be removed or kept. If true the
	 * selected columns are kept and unselected columns are deleted. If false
	 * selected columns are deleted and unselected columns are kept.
	 * 
	 * @param invert the new invert setting
	 */
	public void setInvertSelection(boolean invert) {
	
	  m_DiscretizeCols.setInvert(invert);
	}
	
	/**
	 * Gets the current range selection
	 * 
	 * @return a string containing a comma separated list of ranges
	 */
	public String getAttributeIndices() {
	
	  return m_DiscretizeCols.getRanges();
	}
	
	/**
	 * Sets which attributes are to be Discretized (only numeric attributes among
	 * the selection will be Discretized).
	 * 
	 * @param rangeList a string representing the list of attributes. Since the
	 *          string will typically come from a user, attributes are indexed
	 *          from 1. <br>
	 *          eg: first-3,5,6-last
	 * @throws IllegalArgumentException if an invalid range list is supplied
	 */
	public void setAttributeIndices(String rangeList) {
	
	  m_DiscretizeCols.setRanges(rangeList);
	}
	
	/**
	 * Sets which attributes are to be Discretized (only numeric attributes among
	 * the selection will be Discretized).
	 * 
	 * @param attributes an array containing indexes of attributes to Discretize.
	 *          Since the array will typically come from a program, attributes are
	 *          indexed from 0.
	 * @throws IllegalArgumentException if an invalid set of ranges is supplied
	 */
	public void setAttributeIndicesArray(int[] attributes) {
	
	  setAttributeIndices(Range.indicesToRangeList(attributes));
	}
	
	/**
	 * Gets the cut points for an attribute
	 * 
	 * @param attributeIndex the index (from 0) of the attribute to get the cut
	 *          points of
	 * @return an array containing the cutpoints (or null if the attribute
	 *         requested isn't being Discretized
	 */
	public double[] getCutPoints(int attributeIndex) {
	
	  if (m_CutPoints == null) {
	    return null;
	  }
	  return m_CutPoints[attributeIndex];
	}
	
	/**
	 * Gets the bin ranges string for an attribute
	 * 
	 * @param attributeIndex the index (from 0) of the attribute to get the bin
	 *          ranges string of
	 * @return the bin ranges string (or null if the attribute requested has been
	 *         discretized into only one interval.)
	 */
	public String getBinRangesString(int attributeIndex) {
	
	  if (m_CutPoints == null) {
	    return null;
	  }
	
	  double[] cutPoints = m_CutPoints[attributeIndex];
	
	  if (cutPoints == null) {
	    return "All";
	  }
	
	  StringBuilder sb = new StringBuilder();
	  boolean first = true;
	
	  for (int j = 0, n = cutPoints.length; j <= n; ++j) {
	    if (first) {
	      first = false;
	    } else {
	      sb.append(',');
	    }
	
	    sb.append(binRangeString(cutPoints, j, m_BinRangePrecision));
	  }
	
	  return sb.toString();
	}
	
	/**
	 * Get a bin range string for a specified bin of some attribute's cut points.
	 * 
	 * @param cutPoints The attribute's cut points; never null.
	 * @param j The bin number (zero based); never out of range.
	 * @param precision the precision for the range values
	 * 
	 * @return The bin range string.
	 */
	private static String binRangeString(double[] cutPoints, int j, int precision) {
	  assert cutPoints != null;
	
	  int n = cutPoints.length;
	  assert 0 <= j && j <= n;
	
	  return j == 0 ? "" + "(" + "-inf" + "-"
	    + Utils.doubleToString(cutPoints[0], precision) + "]" : j == n ? "" + "("
	    + Utils.doubleToString(cutPoints[n - 1], precision) + "-" + "inf" + ")"
	    : "" + "(" + Utils.doubleToString(cutPoints[j - 1], precision) + "-"
	      + Utils.doubleToString(cutPoints[j], precision) + "]";
	}
	
	/**
	 * Convert a single instance over. The converted instance is added to the end
	 * of the output queue.
	 * 
	 * @param instance the instance to convert
	 */
	protected InstanceExample discretize(Instance instance) {
		int index = 0;
	    double[] vals = new double[instance.numAttributes()];
	    // Copy and convert the values
	    for (int i = 0; i < instance.numAttributes(); i++) {
	      if (m_DiscretizeCols.isInRange(i)
	        && instance.attribute(i).isNumeric()) {
	        int j;
	        double currentVal = instance.value(i);
	        if (m_CutPoints[i] == null) {
	          if (instance.isMissing(i)) {
	            vals[index] = Utils.missingValue();
	          } else {
	            vals[index] = 0;
	          }
	          index++;
	        } else {
	            if (instance.isMissing(i)) {
	              vals[index] = Utils.missingValue();
	            } else {
	              for (j = 0; j < m_CutPoints[i].length; j++) {
	                if (currentVal <= m_CutPoints[i][j]) {
	                  break;
	                }
	              }
	              if(m_Labels != null) {
	            	  if(m_Labels[i] != null) {
	            		  if(j < m_Labels[i].length)
	            			  vals[index] = Float.parseFloat(m_Labels[i][j]);
	            		  else 
	            			  vals[index] = Integer.MAX_VALUE;
	            	  } else {
	            		  vals[index] = j;
	            	  }
	              } else {
	            	  vals[index] = j;
	              }
	            }
	            index++;
	        }
	      } else {
	        vals[index] = instance.value(i);
	        index++;
	      }
	    }
	    
	    Instance discretizedInstance = null;
	    if (instance instanceof SparseInstance) {
	    	discretizedInstance = new SparseInstance(instance.weight(), vals);
	    } else {
	    	discretizedInstance = new DenseInstance(instance.weight(), vals);
	    }
	    
	    generateNewHeader();
	    discretizedInstance.setDataset(discretizedHeader);
		 
	    return new InstanceExample(discretizedInstance);
	}
	
	public int getNumberIntervals() {
		// TODO Auto-generated method stub
		if(m_CutPoints != null) {
			int ni = 0;
			for(double[] cp: m_CutPoints){
				if(cp != null)
					ni += (cp.length + 1);
			}
			return ni;	
		}
		return 0;
	}
	
	@Override
	public InstancesHeader getHeader() {
		return this.discretizedHeader;
	}

	@Override
	public void getDescription(StringBuilder sb, int indent) {
		throw new NotImplementedException();
	}

	@Override
	protected void restartImpl() {
		this.init = false;
	}
	
	/**
	 * Init the stream
	 */
	public void init() {
		// minus one for the class
		this.nbAttributes = inputStream.getHeader().numAttributes() - 1;
		this.m_DiscretizeCols.setUpper(nbAttributes - 1); // Important (class is removed from discretization)
		//generateNewHeader();
		this.init = true;
	}

	@Override
	public InstanceExample nextInstance() {
		if(!init) {
			init();
		}
		// one more seen instance
		nbSeenInstances++;
		
		Example oldEx = this.inputStream.nextInstance();
		InstanceExample discretizedInstance = (InstanceExample) oldEx.copy();
		Instance inst = discretizedInstance.getData();
		if(m_CutPoints != null)
			discretizedInstance = discretize(inst);
		
		// We update the model with the prior instance
		updateEvaluator(inst);
		 
		return discretizedInstance;
	}
	
	protected void generateNewHeader() {
		InstancesHeader streamHeader = this.inputStream.getHeader();
		int nbAttributes = streamHeader.numAttributes();
		FastVector attributes = new FastVector();
		for (int i = 0; i < nbAttributes; i++) {
			com.yahoo.labs.samoa.instances.Attribute attr = streamHeader.attribute(i);
			// create a new categorical attribute
			if (attr.isNumeric() && m_DiscretizeCols.isInRange(i)) {
				//Set<String> cutPointsCheck = new HashSet<String>();
		      	double[] cutPoints = m_CutPoints[i];
		      	FastVector newAttrLabels = new FastVector();
		        if (cutPoints == null) {
		          newAttrLabels.add("'All'");
		        } else {
		          boolean predefinedLabels = false;
		          if(m_Labels != null) {
		        	  if(m_Labels[i].length == m_CutPoints[i].length)
		        		  predefinedLabels = true;
		          }
		          if(predefinedLabels) {
		        	  for (int j = 0; j < m_Labels[i].length; j++) {
			              newAttrLabels.add(m_Labels[i][j]);
		        	  }  
		          } else {
		            for (int j = 0, n = cutPoints.length; j <= n; ++j) {
		              String newBinRangeString = binRangeString(cutPoints, j,
		                m_BinRangePrecision);
		              /*if (cutPointsCheck.contains(newBinRangeString)) {
		                throw new IllegalArgumentException(
		                  "A duplicate bin range was detected. "
		                    + "Try increasing the bin range precision.");
		              }*/
		              newAttrLabels.add("'" + newBinRangeString + "'");
		            }
		          }
		        }

				attributes.addElement(new Attribute(attr.name(), newAttrLabels));

			} else {
				attributes.addElement(attr);
			}
		}
		discretizedHeader = new InstancesHeader(new Instances(getCLICreationString(InstanceStream.class), attributes, 0));
		// TODO better handle class attribute
		discretizedHeader.setClassIndex(discretizedHeader.numAttributes() - 1);
	}
	

	
	protected void writeCPointsToFile(int att1, int att2, int iteration, String method){
		  FileWriter cpoints1 = null;
		  FileWriter cpoints2 = null;
			try {
				cpoints1 = new FileWriter(method + "-cpoints1" + "-" + iteration + ".dat");
				cpoints2 = new FileWriter(method + "-cpoints2" + "-" + iteration + ".dat");
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		  PrintWriter cpout1 = new PrintWriter(cpoints1);
		  PrintWriter cpout2 = new PrintWriter(cpoints2);		  
		  
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
		  //Flush the output to the file;
		  cpout1.flush();
		  cpout2.flush();
		       
		   //Close the Print Writer
		  cpout1.close();
		  cpout2.close();
		       
		   //Close the File Writer
		   try {
			cpoints1.close();
			cpoints2.close();
		   } catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		   }    
		  
	  }
	

	/**
	 * Update the discretization model without updating 
	 * @param inst
	 */
	public abstract void updateEvaluator(Instance inst);

}