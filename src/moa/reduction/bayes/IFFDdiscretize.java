package moa.reduction.bayes;

/*
 * DiscretizeDynamic.java
 *
 * Created on 2005Äê10ÔÂ12ÈÕ, ÏÂÎç4:34
 *
 * To change this template, choose Tools | Options and locate the template under
 * the Source Creation and Management node. Right-click the template and choose
 * Open. You can then make changes to the template in the Source Editor.
 */

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.Vector;

import moa.reduction.core.MOADiscretize;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Range;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.StreamableFilter;
import weka.filters.unsupervised.attribute.Discretize;
/**
 *
 * @author jlu
 */
public class IFFDdiscretize extends Discretize
        implements StreamableFilter, MOADiscretize{
    
    /**
	 * 
	 */
	/** Stores which columns to Discretize */
	protected Range m_DiscretizeCols = new Range();
	
	protected Set<Integer> classes = new TreeSet<Integer>();
	
	private static final long serialVersionUID = 1L;
	/** number of discrete values in each bin (default is 30) */
    protected int m_BinSize = 45;
    protected int m_MinBinSize=30;
    protected int m_MaxBinSize=60;
    
    //private Instances m_OriginalInstances=null;
    protected int [] m_ChangedAttributes=null;
    protected int[][] m_leftDistribution=null;
    protected int[][] m_rightDistribution=null;
    protected long m_InstanceNum=0;
    protected List<SortedMap<Double, List<Integer>>> m_AttributeClassPairs = null;
    
    
    /** do you want debugging info printed out? */
    protected boolean m_Debug = false;
    /** Store frequency of every interval of every attribute*/
    protected int [][] m_IntervalFrequency=null;
    
    
    /** Creates a new instance of DiscretizeDynamic */
    public IFFDdiscretize() {
        
        
    }
    
    /**
     * Sets the format of the input instances.
     *
     * @param instanceInfo an Instances object containing the input instance
     * structure (any instances contained in the object are ignored - only the
     * structure is required).
     * @return true if the outputFormat may be collected immediately
     * @exception Exception if the input format can't be set successfully
     */
    public boolean setInputFormat(Instances instanceInfo) throws Exception {
        
        // alter child behaviour to do what we want
        m_FindNumBins = true;
        
        return super.setInputFormat(instanceInfo);
    }
    protected void calculateCutPoints() {
        int numAttributes=getInputFormat().numAttributes();
        m_InstanceNum=getInputFormat().numInstances();
        
        /* PD setting begin*/
        /*
         if((int)(Math.sqrt(m_InstanceNum))/2>30){
        m_MinBinSize=(int)(Math.sqrt(m_InstanceNum))/2;
        m_MaxBinSize=m_MinBinSize*2;
        m_BinSize=(m_MaxBinSize+m_MinBinSize)/2;
         }
         */
        /* PD setting end*/
        
        
        m_IntervalFrequency = new int [numAttributes] [];
        m_leftDistribution= new int [numAttributes] [];
        m_rightDistribution = new int [numAttributes] [];
        m_AttributeClassPairs= new ArrayList<SortedMap<Double, List<Integer>>> (numAttributes);
        Instances data = new Instances(getInputFormat());
        
        for(int i = numAttributes - 1; i >= 0; i--) {
            if ((m_DiscretizeCols.isInRange(i)) &&
                    (getInputFormat().attribute(i).isNumeric()) &&
                    (getInputFormat().classIndex() != i)) {
                TreeMap<Double, List<Integer>> map = new TreeMap<Double, List<Integer>>();
                m_AttributeClassPairs.add(map);
            }
        }
        
        super.calculateCutPoints();       
        
    }
    /* update the discretization if there is some interval split,return true, else return false*/
    public void updateEvaluator(Instance instance) {
    	
    	if(m_AttributeClassPairs == null) {
    		m_DiscretizeCols.setUpper(instance.numAttributes() - 1);
    		//setAttributeIndices("first-last");
    		m_IntervalFrequency = new int [instance.numAttributes()] [];
            m_leftDistribution= new int [instance.numAttributes()] [];
            m_rightDistribution = new int [instance.numAttributes()] [];
            m_AttributeClassPairs= new ArrayList<SortedMap<Double, List<Integer>>> (instance.numAttributes());
            
            for(int i = instance.numAttributes() - 1; i >= 0; i--) {
                if ((m_DiscretizeCols.isInRange(i)) &&
                        (instance.attribute(i).isNumeric()) &&
                        (instance.classIndex() != i)) {
                    TreeMap<Double, List<Integer>> map = new TreeMap<Double, List<Integer>>();
                    m_AttributeClassPairs.add(map);
                }
            }
    	}
        
        /*if (getInputFormat() == null) {
            throw new IllegalStateException("No input instance format defined");
        }*/
        
        /*if (m_NewBatch) {
            resetQueue();
            m_NewBatch = false;
        }*/
        //m_OriginalInstances.add(instance);
        classes.add((int) instance.classValue());
        
        int numAttributes=instance.numAttributes();
        for(int i = numAttributes - 1; i >= 0; i--) {
            if ((m_DiscretizeCols.isInRange(i)) &&
                    (instance.attribute(i).isNumeric()) &&
                    (instance.classIndex() != i)) {
                List<Integer> l = m_AttributeClassPairs.get(i).getOrDefault(instance.value(i), new ArrayList<Integer>());
                l.add((int) instance.classValue());
                m_AttributeClassPairs.get(i).put(instance.value(i), l);
            }
        }
        
        if (m_CutPoints != null || m_IntervalFrequency!=null) {
            updateCutpoints(instance);
            //super.resetQueue();
        }
        
        //bufferInput(instance);
        
    }
    
    public Instance applyDiscretization(Instance inst) {
  	  return convertInstance2(inst);
    }   
    
    /**
     * Convert a single instance over. The converted instance is added to the end
     * of the output queue.
     * 
     * @param instance the instance to convert
     */
    protected Instance convertInstance2(Instance instance) {

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
                for (j = 0; j < m_CutPoints[i].length; j++) {
                  if (currentVal <= m_CutPoints[i][j]) {
                    break;
                  }
                }
                vals[index] = j;
                instance.setValue(index, j);
              }
              index++;
            } else {
              for (j = 0; j < m_CutPoints[i].length; j++) {
                if (instance.isMissing(i)) {
                  vals[index] = Utils.missingValue();
                  instance.setValue(index, Utils.missingValue());
                } else if (currentVal <= m_CutPoints[i][j]) {
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

      /*Instance inst = null;
      if (instance instanceof SparseInstance) {
        inst = new SparseInstance(instance.weight(), vals);
      } else {
        inst = new DenseInstance(instance.weight(), vals);
      }

      copyValues(inst, false, instance.dataset(), outputFormatPeek());
      */
      //push(inst); // No need to copy instance
      
      return(instance);
    }

    
    protected void updateCutpoints(Instance instance) {
        /* PD setting begin*/
        /*
        m_InstanceNum++;
        int sqrt;
        sqrt=(int)(Math.sqrt(m_InstanceNum))/2;
        if(sqrt>30 && sqrt!=m_BinSize){
            m_MinBinSize=sqrt;
            m_MaxBinSize=m_MinBinSize*2;
            m_BinSize=(m_MaxBinSize+m_MinBinSize)/2;
        }
         */
        /* PD setting end*/
        int index=0;
        
        double [] vals = new double [instance.numAttributes()];
        //long minBinSize = Math.round(m_BinSize * m_MinPercentSize);
        long minBinSize,maxBinSize;
        minBinSize=m_MinBinSize;
        //long maxBinSize = Math.round(m_BinSize + (m_BinSize * m_MaxPercentSize));
        maxBinSize=m_MaxBinSize;
        
        //Instances data = new Instances(getInputFormat());
        //data.sort(index);

        
        for(index=0;index<instance.numAttributes();index++) {
            if (m_DiscretizeCols.isInRange(index) &&
                    instance.attribute(index).isNumeric() &&
                    (instance.classIndex() != index)){
                int j;
                double currentVal = instance.value(index);
                if (m_CutPoints[index] == null) {
                    if (instance.isMissing(index)) {
                        //vals[index] = Instance.missingValue();
                    } else {
                        /*
                        if(m_IntervalFrequency[index][0]>=maxBinSize) {
                            splitInterval(index, 0);
                        } else {
                            vals[index] = 0;
                            m_IntervalFrequency[index][0]++;
                        }*/
                        splitInterval(index, 0);
                    }
                    
                } else {
                    if(!m_MakeBinary) {
                        
                        if (instance.isMissing(index)) {
                            //vals[index] = Instance.missingValue();
                        } else {
                            for ( j = 0; j < m_CutPoints[index].length; j++) {
                                if (currentVal <= m_CutPoints[index][j]) {
                                    break;
                                }
                            }
                            if(m_IntervalFrequency[index][j]<maxBinSize)
                                m_IntervalFrequency[index][j]++;
                            else
                                splitInterval(index, j);
                        }
                        
                    }
                }
            }
        }
        
    }
    
    protected void splitInterval(int index,int splitinterval) {
        double newcutpoint=0;
        int i,j;
        int start=0,end,count;
        //int numInstances=m_AttributeClassPairs.get(index).size();
        int numAttribute=m_AttributeClassPairs.size();
        for(i=0;i<splitinterval;i++){
            start += m_IntervalFrequency[index][i];
        }
        //if(m_IntervalFrequency[index].length==1)
        //    end=numInstances;
        // else
        end = start+m_IntervalFrequency[index][splitinterval]+1;
        
        //data.delete(data.numInstances()-1);
        Set<Map.Entry<Double, List<Integer>>> attbuteClassPairs = m_AttributeClassPairs.get(index).entrySet();
        Map.Entry<Double, List<Integer>>[] pairs = (Entry<Double, List<Integer>>[]) attbuteClassPairs.toArray();
        if(pairs[start].getKey() ==
        		pairs[end - 1].getKey()) {
            if(m_Debug)
                System.err.println("This interval can not split,"
                        +"all instances in the interval have the same value for attribute");
            m_IntervalFrequency[index][splitinterval]++;
            return;
        }
        
        
        count=0;
        int lastcount=0,lasti=-1;
        int leftIntervalFrequency=0;
        
        
        double sumOfWeights = 0;
        int numOfInstances = 0;
        
        
        int [] classInstance=new int[classes.size()];
        /*
        for(i=0;i<numClass;i++)
            classInstance[i]=0;
         */
        for( i = start; i < end; i++) {
            //if(data.instance(i).isMissing(index))
            //if(attbuteClassPairs.attributeClassPair(i).isMissing())
            //    break;
        	List<Integer> classes = pairs[start].getValue();
        	for (Integer clas : classes) {
        		classInstance[clas]++;
                sumOfWeights ++;
                numOfInstances++;
			}            
        }
        /*
        int maxInstance=0;
        for(i=0;i<numClass;i++)
            if(maxInstance<classInstance[i])
                maxInstance=classInstance[i];
        if(maxInstance/((double)(end-start))>0.8)
        {      m_IntervalFrequency[index][splitinterval]++;
            return;
        }
         */
        int middlecount = (int)(sumOfWeights / 2);
        
        
        int middle = 0;
        int mcount = start;
        while(middle < middlecount) {
            middle ++;
            mcount++;
        }
        double midval = pairs[mcount].getKey();
        
        int n = -1;
        double nweight = 0;
        // if middle val same as last there's no place above middle to make cut
        if(midval != pairs[end - 1].getKey()){
            
            // find the first different value to the right   ******
            //for(n = middle + 1; n < numOfInstances; n++) {
            for(n = mcount + 1; n < end; n++) {
                nweight ++;
                if(midval != pairs[n].getKey())
                    break;  // found place for cut
            }
        }
        
        if(n == -1) { // there's no different vals to the right of the middle
            n = numOfInstances;
            nweight = Double.MAX_VALUE;
        } else {
            // n = n - middle; // use nweight, you idiot   ******
            
            n = n - mcount;
        }
        // n now tells us how many places to the right of middle we'll cut at
        
        int m = start-1, mcut = -1;
        double mweight = 0;
        // find first diff val to left only if midval and first val aren't same
        // && n is more than one away from midpoint (if n is 1 we go with that
        // so don't bother finding m)
        if((midval != pairs[start].getKey()) && n > 1) {
            // find the first different value to the left
            //for(m = middle - 1; m > -1; m--) { ******
            //for(m = mcount - 1; m > -1; m--) {
            for(m = mcount - 1; m > start-1; m--) {
                mweight ++;
                if(midval != pairs[m].getKey()) {
                    // mcut = (int)data.instance(m).value(index);
                    break;  // found place for cut
                }
            }
        }
        
        // find which of m and n is the better cut (closer to middle)
        if(m == start-1 || nweight < mweight ){
            newcutpoint = midval;
            //newcutpoint=attbuteClassPairs.attributeClassPair(mcount+n-1).getAttributeValue();
        } else{
            newcutpoint = pairs[m].getKey();
        }
        
        
        // set cutpoint array for this attribute (with a single cut)
        
        for(j=start; pairs[j].getKey() <= newcutpoint;j++)
            leftIntervalFrequency++;
        
        if((leftIntervalFrequency<m_MinBinSize || end-start-leftIntervalFrequency<m_MinBinSize) && m_CutPoints[index]!=null){
            //if(leftIntervalFrequency<m_MinBinSize || end-start-leftIntervalFrequency<m_MinBinSize) {
            m_IntervalFrequency[index][splitinterval]++;
            return;
        }
        
        /*
         if(m_CutPoints[index]!=null && m_CutPoints[index].length>1 ){
            if(newcutpoint==m_CutPoints[index][splitinterval-1]) {
                int test=0;
         
            }
        }
         */
        
        m_leftDistribution[index]=new int [classes.size()];
        m_rightDistribution[index]=new int [classes.size()];
        /*
        for(i=0;i<numClass;i++){
            m_leftDistribution[index][i]=0;
            m_rightDistribution[index][i]=0;
        }
         */
        for(i=start;i<start+leftIntervalFrequency;i++){
        	List<Integer> classes = pairs[i].getValue();
        	for (Integer clas : classes) {
        		m_leftDistribution[index][clas]++;
			} 
            
        }
        
        for(;i<end;i++){
        	List<Integer> classes = pairs[i].getValue();
        	for (Integer clas : classes) {
        		m_rightDistribution[index][clas]++;
			}
        }
        
        
        
        if(m_CutPoints[index]!=null) {
            double []cutpoints;
            int []intervalfrequency;
            cutpoints=new double [m_CutPoints[index].length+1];
            intervalfrequency=new int [m_IntervalFrequency[index].length+1];
            
            if(splitinterval!=0)
                System.arraycopy(m_CutPoints[index], 0, cutpoints, 0, splitinterval);
            cutpoints[splitinterval]=newcutpoint;
            if(m_CutPoints[index].length!=splitinterval)
                System.arraycopy(m_CutPoints[index], splitinterval, cutpoints,
                        splitinterval+1, m_CutPoints[index].length-splitinterval);
            
            if(splitinterval!=0)
                System.arraycopy(m_IntervalFrequency[index], 0, intervalfrequency, 0, splitinterval);
            
            intervalfrequency[splitinterval]=leftIntervalFrequency;
            intervalfrequency[splitinterval+1]=end-start-leftIntervalFrequency;
            if(m_CutPoints[index].length!=splitinterval)
                System.arraycopy(m_IntervalFrequency[index], splitinterval+1, intervalfrequency,
                        splitinterval+2, m_CutPoints[index].length-splitinterval);
            
            
            m_CutPoints[index]=cutpoints;
            m_IntervalFrequency[index]=intervalfrequency;
        } else{
            m_CutPoints[index]=new double [1];
            m_CutPoints[index][0]=newcutpoint;
            
            m_IntervalFrequency[index]=new int [2];
            m_IntervalFrequency[index][0]=leftIntervalFrequency;
            m_IntervalFrequency[index][1]=end-start-leftIntervalFrequency;
        }
        
        if (m_ChangedAttributes==null){
            m_ChangedAttributes=new int [numAttribute-1];
            for(i=0;i<numAttribute-1;i++)
                m_ChangedAttributes[i]=-1;
        }
        m_ChangedAttributes[index]=splitinterval;
        //updateOutputFormat(index);
        
    }
    
    
    
    /*public void updateOutputFormat(int index)    {
        FastVector Attriutes=null;
        Attriutes=super.m_OutputFormat.getAttributesInformation();
        Attriutes.removeElementAt(index);
        int i=index;
        
        FastVector attribValues = new FastVector(1);
        if (m_CutPoints[i] == null) {
            attribValues.addElement("'All'");
        } else {
            for(int j = 0; j <= m_CutPoints[i].length; j++) {
                if (j == 0) {
                    attribValues.addElement("'(-inf-"
                            + Utils.doubleToString(m_CutPoints[i][j], 6) + "]'");
                } else if (j == m_CutPoints[i].length) {
                    attribValues.addElement("'("
                            + Utils.doubleToString(m_CutPoints[i][j - 1], 6)
                            + "-inf)'");
                } else {
                    attribValues.addElement("'("
                            + Utils.doubleToString(m_CutPoints[i][j - 1], 6) + "-"
                            + Utils.doubleToString(m_CutPoints[i][j], 6) + "]'");
                }
            }
        }
        Attriutes.insertElementAt(new Attribute(instance.
                attribute(i).name(),
                attribValues,index),index);
        m_OutputFormat.setAttributesInformation(Attriutes);
    }*/
    
    public int [] getLeftDistribution(int index){
        return m_leftDistribution[index];
    }
    public int [] getRightDistribution(int index){
        return m_rightDistribution[index];
    }
    /*
    public Instances getOriginalInstances() {
        return m_OriginalInstances;
    }
     */
    public int [] getChangedAttributes(){
        return m_ChangedAttributes;
    }
    
    public void resetChanagedAttributes(){
        m_ChangedAttributes=null;
    }
    
    public int checkChangedAttribute(int attIndex){
        return m_ChangedAttributes[attIndex];
        
    }
    
    public boolean input(Instance instance) {
        if (m_NewBatch) {
            m_ChangedAttributes=null;
        }
        return super.input(instance);
    }
    /**
     * Finds the number of bins to use and creates the cut points.
     *
     * @param index the index of the attribute to be discretized
     */
    protected void findNumBins(int index) {
        
        // create copy of the data and sort it
        Instances data = new Instances(getInputFormat());
        
        
        data.sort(index);
        //m_OriginalInstances=new Instances(getInputFormat());
        double sumOfWeights = 0;
        int numOfInstances = 0;
        int temp;
        for(int i = 0; i < data.numInstances(); i++) {
            if(data.instance(i).isMissing(index))
                break;
            sumOfWeights += data.instance(i).weight();
            numOfInstances++;
        }
        
        // bail out now if there's no non-missing for this attribute
        if(sumOfWeights == 0) {
            if(m_Debug)
                System.err.println("there are no instances that have valid "
                        + "values for attribute " + index);
            m_CutPoints[index] = null;
            m_IntervalFrequency[index]=new int [1];
            m_IntervalFrequency[index][0]=numOfInstances;
            return;
        }
        
        if(m_Debug)
            System.out.println("sum of weights == " + sumOfWeights);
        
        // return null cutpoints for this att if all instances have same value
        // if(datasetAttVals[0] == datasetAttVals[numOfInstances - 1])
        if(data.instance(0).value(index)
        == data.instance(numOfInstances - 1).value(index)) {
            if(m_Debug)
                System.err.println("all instances have the same value for attribute "
                        + index);
            m_CutPoints[index] = null;
            m_IntervalFrequency[index]=new int [1];
            m_IntervalFrequency[index][0]=numOfInstances;
            return;
        }
        
        // work out the min and max deviations from the binsize that are allowed
        //long minBinSize = Math.round(m_BinSize * m_MinPercentSize);
        long minBinSize,maxBinSize;
        minBinSize=m_MinBinSize;
        // long maxBinSize = Math.round(m_BinSize + (m_BinSize * m_MaxPercentSize));
        maxBinSize=m_MaxBinSize;
        
        // allocate space to array as largest possible number of cutpoints
        // double [] cutPoints = new double[(int)(sumOfWeights/m_BinSize + 1)];
        double [] cutPoints = new double[(int)(sumOfWeights/minBinSize + 1)];
        
        
        if(m_Debug) {
            System.out.println("cutPoints[] size is: "
                    + (sumOfWeights / m_BinSize + 1));
            // + (numOfInstances / m_BinSize + 1));
        }
        
        int cpIndex = 0;  // tracks index in cutPoints[] of most recent cut
        int count = 0;    // number of instances in the current bin
        // vars for a cutpoint that's at least 1/2 m_BinSize, but < m_BinSize
        int lastcount = 0, lasti = -1;
        
        // handle differently if weight of all instances is < 2 times binsize
        // if(numOfInstances < (2 * m_BinSize))
        if(sumOfWeights < (2 * m_BinSize)) {
            
            if(m_Debug)
                System.err.println("number of instances is too small for "
                        + "specified bin size.");
            
            // find the index for the instance at the "middle" point
            int middlecount = (int)(sumOfWeights / 2);
            // double middle = 0;
            int middle = 0;
            int mcount = 0;
            while(middle < middlecount)
                middle += data.instance(mcount++).weight();
            double midval = (double)data.instance(mcount).value(index);
            
            int n = -1;
            double nweight = 0;
            // if middle val same as last there's no place above middle to make cut
            if(midval != data.instance(numOfInstances - 1).value(index)) {
                // find the first different value to the right
                //for(n = middle + 1; n < numOfInstances; n++) {  ******
                for(n = mcount + 1; n < numOfInstances; n++) {
                    nweight += data.instance(n).weight();
                    if(midval != data.instance(n).value(index))
                        break;  // found place for cut
                }
            }
            
            if(n == -1) { // there's no different vals to the right of the middle
                n = numOfInstances;
                nweight = Double.MAX_VALUE;
            } else {
                //n = n - middle; // use nweight, you idiot ******
                n = n - mcount; // use nweight, you idiot
            }
            // n now tells us how many places to the right of middle we'll cut at
            
            int m = -1, mcut = -1;
            double mweight = 0;
            // find first diff val to left only if midval and first val aren't same
            // && n is more than one away from midpoint (if n is 1 we go with that
            // so don't bother finding m)
            if((midval != data.instance(0).value(index)) && n > 1) {
                // find the first different value to the left
                //for(m = middle - 1; m > -1; m--) {  ******
                for(m = mcount - 1; m > -1; m--) {
                    mweight += data.instance(m).weight();
                    if(midval != data.instance(m).value(index)) {
                        // mcut = (int)data.instance(m).value(index);
                        break;  // found place for cut
                    }
                }
            }
            
            // find which of m and n is the better cut (closer to middle)
            if(m == -1 || nweight < mweight){
                cutPoints[cpIndex++] = midval;
            }else{
                cutPoints[cpIndex++] = data.instance(m).value(index);
            }
            
            // set cutpoint array for this attribute (with a single cut)
            double [] cp = new double[1];
            cp[0] = cutPoints[0];
            m_CutPoints[index] = cp;
            /*calculate the interval frequency*/
            int j=0;
            m_IntervalFrequency[index]=new int [2];
            count=0;
            for(;data.instance(j).value(index)<=m_CutPoints[index][0];j++) {
                count++;
            }
            m_IntervalFrequency[index][0]=count;
            
            m_IntervalFrequency[index][1]=data.numInstances()-j;
            return;
        }
        
        if(m_Debug) {
            System.out.println("binsize=" + m_BinSize + "; minsize=" + minBinSize
                    + "; maxsize=" + maxBinSize);
        }
        
        for(int i = 0; i < numOfInstances - 1; i++) {
            
            // count++;
            count+= data.instance(i).weight();
            
            // don't even look at cutpoints until we're over the minimum size
            if(count < minBinSize)
                continue;
            
            // potential cutpoint?
            if(data.instance(i).value(index) < data.instance(i + 1).value(index)) {
                
                if(count < m_BinSize) {
                    
                    // this will be a potential cut between minbinsize and m_BinSize
                    
                    lastcount = count;   // remember info for a < m_BinSize cut
                    lasti = i;
                    
                } else {
                    
                    // this will be a cut on or above m_BinSize
                    
                    // is this cut better than the one below m_BinSize???
                    
                    if(lastcount == 0 || count <= maxBinSize) {
                        
                        // go with this one
                        if(m_Debug)
                            System.out.println("cutting: val="
                                    + data.instance(i).value(index) + "; count="
                                    + count + "; i=" + i);
                        cutPoints[cpIndex++] = data.instance(i).value(index);
                        
                        // cutPoints[cpIndex++] = data.instance(i+1).value(index);
                        // and reset the counter variables
                        count = 0;
                        lastcount = 0;
                        lasti = -1;
                        
                    } else {
                        
                        // go with the lower
                        cutPoints[cpIndex++] = data.instance(lasti).value(index);
                        
                        // cutPoints[cpIndex++] = data.instance(lasti+1).value(index);
                        count -= lastcount;
                        // can we make another cutpoint already?
                        if(count >= m_BinSize) {
                            
                            //cutPoints[cpIndex++] = data.instance(i+1).value(index);
                            cutPoints[cpIndex++] = data.instance(i).value(index);
                            count = 0;
                        }
                        // set last* counters to appropriate value
                        if(count >= minBinSize) {
                            lastcount = count;
                            lasti = i;
                        } else {
                            lastcount = 0;
                            lasti = -1;
                        }
                    }
                }
            }
        }
        
        // remove last cutPoint if the last bin is too small
        // if(count < minBinSize) need to + 1, b/c we looped through numInstances-1
        if(count + 1 < minBinSize) {
            cpIndex--;
            if(m_Debug)
                System.out.println("removing last cut, count=" + count);
        }
        
        // copy the cutpoints info to m_CutPoints
        if(cpIndex < 1) {
            // no cutpoints made
            m_CutPoints[index] = null;
            m_IntervalFrequency[index]=new int [1];
            m_IntervalFrequency[index][0]=data.numInstances();
            
            if(m_Debug)
                System.out.println("no cutpoints.");
            
            return;
        } else {
            double [] cp = new double[cpIndex];
            for(int x = 0; x < cpIndex; x++) {
                cp[x] = cutPoints[x];
            }
            m_CutPoints[index] = cp;
            
            if(m_Debug) {
                System.out.print("CUTPOINTS at:\n\t");
                for(int x = 0; x < cpIndex; x++)
                    System.out.print(cutPoints[x] + ", ");
                System.out.println();
            }
        }
        /*calculate the interval frequency*/
        int i,j;
        
        m_IntervalFrequency[index]=new int [m_CutPoints[index].length + 1];
        for( i=0,j=0;i<m_CutPoints[index].length;i++) {
            count=0;
            for(;data.instance(j).value(index)<=m_CutPoints[index][i];j++) {
                count++;
            }
            m_IntervalFrequency[index][i]=count;
        }
        m_IntervalFrequency[index][i]=data.numInstances()-j;
        
    }
    
    
    public String toString() {
        
        StringBuffer text = new StringBuffer();
        
        text.append("Discretization interval:\n\n");
        int numAttribute=getInputFormat().numAttributes();
        try{
            
            
            Enumeration enumAtts = getInputFormat().enumerateAttributes();
            int attIndex = 0;
            while (enumAtts.hasMoreElements()) {
                Attribute attribute = (Attribute) enumAtts.nextElement();
                if(m_IntervalFrequency[attIndex]!=null){
                    text.append("\n"+attribute.name() + ":  ");
                    for(int i=0;i<m_IntervalFrequency[attIndex].length;i++)
                        text.append(m_IntervalFrequency[attIndex][i]+";");
                }
                attIndex++;
            }
        } catch (Exception ex) {
            text.append(ex.getMessage());
        }
        
        
        return text.toString();
    }
    
    
    /**
     * Gets an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    public Enumeration listOptions() {
        
        Vector newVector = new Vector(6);
        /*Specifies the bin size information*/
        newVector.addElement(new Option(
                "\tSpecifies the desired minimum weight of instances of each bin"
                + " to divide each attributes into."
                + " the desired maximum weight of each bin will be 2 times of minimum bin size;"
                + " the desired weight of each bin will be the mean of minimum and maximum bin size.\n"
                + "\t(default = 3)",
                "S", 1, "-S <num>"));
        
        newVector.addElement(new Option(
                "\tSpecifies list of columns to Discretize. First"
                + " and last are valid indexes.\n"
                + "\t(default: first-last)",
                "R", 1, "-R <col1,col2-col4,...>"));
        
        return newVector.elements();
    }
    
    
    /**
     * Parses the options for this object. Valid options are: <p>
     *
     * -D <br>
     * Debugging information is printed.<p>
     *
     * -R col1,col2-col4,... <br>
     * Specifies list of columns to Discretize. First
     * and last are valid indexes. (default none) <p>
     *
     * -S <br>
     * The bin size (desired weight of instances per bin). (default 30) <p>
     *
     * -N <br>
     * The minimum weight allowed for a bin (as a percentage of the bin size) <p>
     *
     * -M <br>
     * The maximum weight allowed for a bin (as a percentage of the bin size) <p>
     *
     * @param options the list of options as an array of strings
     * @exception Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {
        
        m_Debug = Utils.getFlag('D', options);
        
        String minsizeBins = Utils.getOption('S', options);
        if(minsizeBins.length() != 0) {
            m_MinBinSize = Integer.parseInt(minsizeBins);
            m_MaxBinSize = m_MinBinSize*2;
            m_BinSize = (int)((m_MinBinSize+m_MaxBinSize)/2);
        }
        
        
        String convertList = Utils.getOption('R', options);
        if (convertList.length() != 0) {
            setAttributeIndices(convertList);
        } else {
            setAttributeIndices("first-last");
        }
        
        if (getInputFormat() != null) {
            setInputFormat(getInputFormat());
        }
    }
    
    
    /**
     * Gets the current settings of the filter.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    public String [] getOptions() {
        
        String [] options = new String [8];
        int current = 0;
        
        if (!getAttributeIndices().equals("")) {
            options[current++] = "-R"; options[current++] = getAttributeIndices();
        }
        options[current++] = "-S"; options[current++] = "" + getBinSize();
        options[current++] = "-N"; options[current++] = "" + getMinBinSize();
        options[current++] = "-M"; options[current++] = "" + getMaxBinSize();
        
        while (current < options.length) {
            options[current++] = "";
        }
        return options;
    }
    
    /**
     * Returns a string describing this filter
     *
     * @return a description of the filter suitable for
     * displaying in the explorer/experimenter gui
     */
    public String globalInfo() {
        
        return "To discretize a numeric attribute, fixed frequency discretization"
                +" (FFD) sets a sufficient interval frequency, m. It then discretizes"
                +" the ascendingly sorted values into intervals of frequency m. Thus"
                +" each interval has approximately the same number m of training"
                +" instances with adjacent (possibly identical) values.<br>FFD has been"
                +" demonstrated appropriate for naïve-Bayes classifiers [1].  By"
                +" introducing m, FFD aims to ensure that in general the interval"
                +" frequency is sufficient so that there are enough training"
                +" instances in each interval to reliably estimate the naive-Bayes"
                +" probabilities. Thus FFD can prevent high discretization variance. "
                +" By not limiting the number of intervals formed, more intervals"
                +" can be formed as the training data size increases. This means that"
                +" FFD can make use of additional data to reduce discretization bias. "
                +" As a result, FFD can lead to lower classification errors.<br>It is"
                +" important to distinguish FFD from equal frequency discretization"
                +" (EFD) [2], both of which form intervals of equal frequency. EFD"
                +" fixes the interval number instead of controlling the interval"
                +" frequency. It arbitrarily chooses the interval number k and then"
                +" discretizes a numeric attribute into k intervals such that each"
                +" interval has the same number of training instances. EFD is not"
                +" good at reducing discretization bias or variance.<br>"
                +"[1] YING YANG, 2003. Discretization for naïve-Bayes learning. PhD"
                +" Thesis, School of Computer Science and Software Engineering, Monash"
                +" University, Australia.<br>[2] DOUGHERTY, J., KOHAVI, R., AND SAHAMI,"
                +" M. 1995. Supervised and unsupervised discretization of continuous"
                +" features.";
    }
    
    
    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String findNumBinsTipText() {
        
        return "Ignored.";
    }
    
    /**
     * Get the value of FindNumBins.
     *
     * @return Value of FindNumBins.
     */
    public boolean getFindNumBins() {
        
        return false;
    }
    
    /**
     * Set the value of FindNumBins.
     *
     * @param newFindNumBins Value to assign to FindNumBins.
     */
    public void setFindNumBins(boolean newFindNumBins) {
        
    }
    
    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String binSizeTipText() {
        
        return "the target number of items in each bin";
    }
    
    /**
     * Get the value of m_BinSize.
     *
     * @return Value of FindNumBins.
     */
    public int getBinSize() {
        
        return m_BinSize;
    }
    
    /**
     * Set the value of m_BinSize.
     *
     * @param binsize Value to assign to m_BinSize.
     */
    public void setBinSizeInformation(int minbinsize) {
        m_MinBinSize = minbinsize;
        m_MaxBinSize = 2*m_MinBinSize;
        m_BinSize = m_MinBinSize+m_MaxBinSize;
    }
    
    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String minBinSizeTipText() {
        
        return "The minimum target binSize allowed as a bin size";
    }
    
    /**
     * Get the value of MinPercentSize.
     *
     * @return Value of m_MinPercentSize.
     */
    public double getMinBinSize() {
        
        return m_MinBinSize;
    }
    
    
    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String maxBinSizeTipText() {
        
        return "The amount a bin can exceed target binSize.";
    }
    
    /**
     * Get the value of MaxPercentSize.
     *
     * @return Value of m_MaxPercentSize.
     */
    public int getMaxBinSize() {
        
        return m_MaxBinSize;
    }
    
    
    
    /**
     * Main method for testing this class.
     *
     * @param argv should contain arguments to the filter: use -h for help
     */
    public static void main(String [] argv) {
        
        try {
            if (Utils.getFlag('b', argv)) {
                Filter.batchFilterFile(new IFFDdiscretize(), argv);
            } else {
                Filter.filterFile(new IFFDdiscretize(), argv);
            }
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
        }
    }
}