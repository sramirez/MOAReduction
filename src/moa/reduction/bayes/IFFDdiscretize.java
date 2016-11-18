package moa.reduction.bayes;

/*
 * DiscretizeDynamic.java
 */

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.TreeSet;

import moa.reduction.core.MOADiscretize;
import weka.core.Range;

import com.yahoo.labs.samoa.instances.Instance;
/**
 * "To discretize a numeric attribute, fixed frequency discretization (FFD) sets a sufficient interval frequency, m. 
 * It then discretizes the ascendingly sorted values into intervals of frequency m. 
 * Thus each interval has approximately the same number m of training instances with adjacent (possibly identical) values.
 * <br>FFD has been demonstrated appropriate for naïve-Bayes classifiers [1]. 
 * By introducing m, FFD aims to ensure that in general the interval frequency is sufficient so that there are enough training 
 * instances in each interval to reliably estimate the naive-Bayes probabilities. 
 * Thus FFD can prevent high discretization variance. 
 * By not limiting the number of intervals formed, more intervals can be formed as the training data size increases. 
 * This means that FFD can make use of additional data to reduce discretization bias. 
 * As a result, FFD can lead to lower classification errors.
 * <br>It is important to distinguish FFD from equal frequency discretization (EFD) [2], 
 * both of which form intervals of equal frequency. 
 * EFD  fixes the interval number instead of controlling the interval frequency. 
 * It arbitrarily chooses the interval number k and then discretizes a numeric attribute into k 
 * intervals such that each interval has the same number of training instances. 
 * EFD is not good at reducing discretization bias or variance.<br> 
 * [1] YING YANG, 2003. Discretization for naïve-Bayes learning. PhD Thesis, School of Computer Science and Software Engineering, 
 * Monash University, Australia.<br>
 * [2] DOUGHERTY, J., KOHAVI, R., AND SAHAMI, M. 1995. Supervised and unsupervised discretization of continuous features.
 * 
 * @author Sergio Ramírez (sramirez at decsai dot ugr dot es)
 */
public class IFFDdiscretize extends MOADiscretize {
    
    /**
	 * 
	 */
	/** Stores which columns to Discretize */
	protected Range m_DiscretizeCols = new Range("first-last");
	
	protected Set<Integer> classes = new TreeSet<Integer>();
	
	private static final long serialVersionUID = 1L;
	/** number of discrete values in each bin (default is 30) */
    //protected int m_BinSize = 45;
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
    
    public IFFDdiscretize() {
		// TODO Auto-generated constructor stub
	}
    
    public IFFDdiscretize(int minSize, int maxSize) {
		// TODO Auto-generated constructor stub
    	m_MinBinSize = minSize;
    	m_MaxBinSize = maxSize;
	}
    
    /**
     * Update the discretization scheme using the new instance values.
     * It also initializes all variables if it is the first update.
     * 
     * @param instance the instance to consider
     */
    public void updateEvaluator(Instance instance) {
    	
    	if(m_AttributeClassPairs == null) {
    		m_DiscretizeCols.setUpper(instance.numAttributes() - 1);
    		//setAttributeIndices("first-last");
    		m_IntervalFrequency = new int [instance.numAttributes()] [];
            m_leftDistribution= new int [instance.numAttributes()] [];
            m_rightDistribution = new int [instance.numAttributes()] [];
            m_CutPoints = new double [instance.numAttributes()] []; 
            m_AttributeClassPairs= new ArrayList<SortedMap<Double, List<Integer>>> (instance.numAttributes());
            
            for(int i = instance.numAttributes() - 1; i >= 0; i--) {
                if ((m_DiscretizeCols.isInRange(i)) &&
                        (instance.attribute(i).isNumeric()) &&
                        (instance.classIndex() != i)) {
                    TreeMap<Double, List<Integer>> map = new TreeMap<Double, List<Integer>>();
                    m_AttributeClassPairs.add(map);
                    m_IntervalFrequency[i] = new int[1];
                }
            }
    	}
    	
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
        }        
    }
    
    /**
     * Apply to the instance the discretization scheme derived from previous updates.
     *  
     * @param instance instance to discretize
     */
    public Instance applyDiscretization(Instance inst) {
    	if(m_CutPoints != null)
    		return convertInstance(inst);
    	return inst;
    }

    /**
     * Update the discretization scheme using the new instance. 
     * It makes several calls to the splitInterval function.
     * 
     * @param instance the instance to consider
     */
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
        
        //long minBinSize = Math.round(m_BinSize * m_MinPercentSize);
        long maxBinSize;
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
    
    /**
     * Split the given interval into two equal-size intervals.
     * It also update the distribution variables.
     * 
     * @param index Attribute index
     * @param splitinterval Interval index in the class attribute pairs.
     */
    protected void splitInterval(int index,int splitinterval) {
        double newcutpoint=0;
        int i,j;
        int start=0,end;
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
        LinkedList<Map.Entry<Double, List<Integer>>> pairs = new LinkedList<Map.Entry<Double, List<Integer>>>();
        //Map.Entry<Double, List<Integer>>[] pairs = (Entry<Double, List<Integer>>[]) attbuteClassPairs.toArray();
        
        for(Map.Entry<Double, List<Integer>> e: m_AttributeClassPairs.get(index).entrySet())
        	pairs.add(e);
        
        
        
        if(pairs.get(start).getKey() ==
        		pairs.get(end - 1).getKey()) {
            if(m_Debug)
                System.err.println("This interval can not split,"
                        +"all instances in the interval have the same value for attribute");
            m_IntervalFrequency[index][splitinterval]++;
            return;
        }
        
        int leftIntervalFrequency=0;
        double sumOfWeights = 0;
        int numOfInstances = 0;
        
        
        int max = Integer.MIN_VALUE;

        for (int d : classes) {
           if (d > max) max = d;
        }

        int [] classInstance=new int[max + 1];
        
        for( i = start; i < end; i++) {
            //if(data.instance(i).isMissing(index))
            //if(attbuteClassPairs.attributeClassPair(i).isMissing())
            //    break;
        	List<Integer> classes = pairs.get(start).getValue();
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
        double midval = pairs.get(mcount).getKey();
        
        int n = -1;
        double nweight = 0;
        // if middle val same as last there's no place above middle to make cut
        if(midval != pairs.get(end - 1).getKey()){
            
            // find the first different value to the right   ******
            //for(n = middle + 1; n < numOfInstances; n++) {
            for(n = mcount + 1; n < end; n++) {
                nweight ++;
                if(midval != pairs.get(n).getKey())
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
        
        int m = start-1;
        double mweight = 0;
        // find first diff val to left only if midval and first val aren't same
        // && n is more than one away from midpoint (if n is 1 we go with that
        // so don't bother finding m)
        if((midval != pairs.get(start).getKey()) && n > 1) {
            // find the first different value to the left
            //for(m = middle - 1; m > -1; m--) { ******
            //for(m = mcount - 1; m > -1; m--) {
            for(m = mcount - 1; m > start-1; m--) {
                mweight ++;
                if(midval != pairs.get(m).getKey()) {
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
            newcutpoint = pairs.get(m).getKey();
        }
        
        
        // set cutpoint array for this attribute (with a single cut)
        
        for(j=start; pairs.get(j).getKey() <= newcutpoint;j++)
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
        m_leftDistribution[index]=new int [max + 1];
        m_rightDistribution[index]=new int [max + 1];
        /*
        for(i=0;i<numClass;i++){
            m_leftDistribution[index][i]=0;
            m_rightDistribution[index][i]=0;
        }
         */
        for(i=start;i<start+leftIntervalFrequency;i++){
        	List<Integer> classes = pairs.get(i).getValue();
        	for (Integer clas : classes) {
        		m_leftDistribution[index][clas]++;
			} 
            
        }
        
        for(;i<end;i++){
        	List<Integer> classes = pairs.get(i).getValue();
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
            m_ChangedAttributes=new int [numAttribute];
            for(i=0;i<numAttribute;i++)
                m_ChangedAttributes[i]=-1;
        }
        m_ChangedAttributes[index]=splitinterval;
        //updateOutputFormat(index);
        
    }
}