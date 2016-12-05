# MOAReduction (Data reduction on MOA)

## Abstract

MOAReduction is an extension for MOA, which allows to perform data reduction on data streams (w/o drift). It includes several reduction methods for different reduction tasks, such as: discretization, instance selection, and feature selection. The complete list of methods is described below:

## Installation and requirements

In order to use the library and its reduction algorithms, please download MOAReduction.jar, jCOLIBRI2.jar (both modified), guava.jar and weka.jar from the lib directory in this repository. Notice that only weka.jar and guava (v20.0) can be downloaded from the official repository (the version used of weka is detailed in the NOTICE file). Then add all these libraries to the classpath when lauching MOA.

Example (Windows):

> java -cp .;lib/MOAReduction.jar;moa.jar;lib/guava-20.0.jar;lib/weka.jar;lib/jCOLIBRI2.jar -javaagent:sizeofag-1.0.0.jar moa.gui.GUI

Example (Linux/mac):

> java -cp lib/MOAReduction.jar:moa.jar:lib/guava-20.0.jar:lib/weka.jar:lib/jCOLIBRI2.jar -javaagent:sizeofag-1.0.0.jar moa.gui.GUI

You can also unzip MOA's any of the JAR files, add the source code for the algorithms desired to MOA and recompile it.

Requirements: Java 8 and MOA (v2016.04).

## Documentation

### Instance Selection

### Noise-Enhanced Fast Context Switch + Stepwise Redundancy Removal (NEFCS-SRR) -> moa.classifiers.competence.NEFCSSRR

NEFCS-SRR is a new case-base editing technique which takes both competence enhancement and competence
preservation into consideration. Noise-Enhanced Fast Context Switch (NEFCS)
algorithm prevents noise from being included during case retention and to speed the context switch process in the face of concept drift. By taking advantage of a concept drift detection technique, NEFCS minimizes the risk of discarding novel concepts. For competence preservation, Stepwise Redundancy Removal (SRR) algorithm  uniformly removes superfluous cases without loss of case-base competence. 

Params:
* -k neighbors: Number of neighbors to used in searches (default = 3)
* -p period: batch size. Algorithm is activated in each batch arrival, also the concept drift detector (default = 500)
* -l L: number of elements in the windows of predictions used in the confidence interval test (default = 10) 
* -i pmin: activation threshold for the confidence interval test (default = 0.5)
* -a pmax: deactivation threshold for the confidence interval test (default = 0.5)
* -s sizeLimit: when this limit is overcome, SRR is applied (default = 10,000)

*N. Lu, J. Lu, G. Zhang, R. L. de Mantaras, A concept drift-tolerant case-base editing technique, Artificial Intelligence 230 (2016) 108 – 133.*

### Competence-Based Editing (CBE) -> moa.classifiers.competence.BBNRFullCB

CBE has two stages, a noise reduction phase called BBNR and a redundancy elimination phase
called CRR. BBNR focuses on the damage that certain cases are causing in classifications. This model includes the notion of blame or liability. A measure is introduced for a case of how often it is the cause of, or contributes to, other cases being incorrectly classified. redundancy reduction process (CRR) focuses on a more conservative reduction of the case-base than its competitors. It uses the competence characteristics of the case-base to identify and retain border cases.

Params:
* -k neighbors: Number of neighbors to used in searches (default = 3)
* -p period: batch size. Algorithm is activated in each batch arrival, also the concept drift detector (default = 500)

*S. J. Delany, P. Cunningham, A. Tsymbal, L. Coyle, A case-based technique for tracking concept drift in spam filtering, Knowledge-Based Systems 18 (45) (2005) 187 – 195.*

### Iterative Case Filtering Algorithm (ICF) -> moa.classifiers.competence.ICFFullCB

ICF is focused on redundancy removal, only removes those cases with a coverage set size smaller than
its reachability set. Repeated Edited-NN is included to remove noise around the borders.

Params:
* -k neighbors: Number of neighbors to used in searches (default = 3)
* -p period: batch size. Algorithm is activated in each batch arrival, also the concept drift detector (default = 500)

*H. Brighton, C. Mellish, Advances in instance selection for instance-based learning algorithms, Data Mining and Knowledge Discovery 6 (2) (2002) 153–172.*

### uniFied Instance Selection algoritHm (FISH) -> moa.classifiers.meta.FISH

FISH combines distance in time and space to select instances. Training instances
are systematically selected at each time step. The methods can be used with different base classifiers.
The family includes three modifications: FISH1, FISH2 and FISH3. In FISH1 the size of a training set is fixed and set in advance, the extension FISH2 operates using variable training set size. In FISH2 the proportion of time and space distances in the final distance measure are fixed in advance as a design choice. FISH2 is considered to be the central in the family, and this version is implemented here.

Params:
* -k neighbors: Number of neighbors to used in searches (default = 3)
* -p period: batch size. Algorithm is activated in each batch arrival, also the concept drift detector (default = 500)
* -l baseLearner: base learner to evaluate examples (default = lazy.IBk (weka))

*I. Zliobaite, Combining similarity in time and space for training set formation under concept drift, Intelligent Data Analysis 15 (4) (2011) 589–611.*

### NaiveBayesReduction (Naive Bayes + Feature Selection and/or Discretization) -> moa.reduction.core.NaiveBayesReduction
Params: 
* -w winSize: batch size (default = 1)
* -f numFeatures: number of features to select (default = 10)
* -m fsMethod: feature selection method to apply. Options: 0. No method. 1. InfoGain 2. Symmetrical Uncertainty 3. OFS (default = 0)
* -d discMethod: feature selection method to apply. Options: 0. No method. 1. PiD 2. IFFD 3. Online Chi-Merge 4. IDA (default = 0)
* -c numClasses: maximum number of classes involved in the problem (default = 100)

The above parameters are common to the following discretization and feature selection methods: 

### Discretization

### Partition Incremental Discretization algorithm (PiD)

PiD performs incremental discretization. The basic idea is to perform the task in two layers. The first layer receives the sequence of input data and keeps some statistics on the data using many more intervals than required. Based on the statistics stored by the first layer, the second layer creates the final discretization. The proposed architecture processes streaming exam
ples in a single scan, in constant time and space even for infinite sequences of examples.

*J. Gama, C. Pinto, Discretization from data streams: Applications to histograms and data mining, in: Proceedings of the 2006 ACM Sympo sium on Applied Computing, SAC ’06, 2006, pp. 662–667.*

### Online ChiMerge (OC)

OC is an online version of popular ChiMerge algorithm. ChiMerge is a univariate approach in which only one attribute is examined at a time without regard to the values of the other attributes. The interval combinations need to be examined in the order of their goodness as indicated by the X^2 statistics.

*P. Lehtinen, M. Saarela, T. Elomaa, Data Mining: Foundations and Intelligent Paradigms: Volume 2: Statistical, Bayesian, Time Series and other Theoretical Aspects, Springer Berlin Heidelberg, Berlin, Heidelberg, 2012, Ch. Online ChiMerge Algorithm, pp. 199–216.*

### Incremental Flexible Frequency Discretization (IFFD)

IFFD produces intervals with flexible sizes, stipulated by a lower bound and an upper bound. An interval is allowed to accept new values until its size reaches the upper bound. 
An interval whose size exceeds the upper bound is allowed to split if the resulting smaller intervals each have a size no smaller than the lower bound. 
Accordingly IFFD is able to incrementally adjust discretized intervals, effectively update associated statistics and efficiently synchronize with NB's incremental learning.

*J. Lu, Y. Yang, G. I. Webb, Incremental discretization for Naive-bayes classifier, in: Proceedings of the Second International Conference on Advanced Data Mining and Applications, ADMA’06, 2006, pp. 223–
238.*

### Incremental Discretization Algorithm (IDA)

Incremental Discretization Algorithm (IDA) approximates quantile-based discretization on the entire data stream
encountered to date by maintaining a random sample of the data which is used to calculate the cut points. IDA uses the reservoir sampling algorithm to maintain a sample drawn uniformly at random from the entire stream up until the current time.

*G. I. Webb. 2014. Contrary to Popular Belief Incremental Discretization can be Sound, Computationally Efficient and Extremely Useful for Streaming Data. In Proceedings of the 2014 IEEE International Conference on Data Mining (ICDM '14). IEEE Computer Society, Washington, DC, USA, 1031-1036.*

## Feature Selection:

### Katakis' FS

This FS scheme is formed by two steps: a) an incremental feature ranking method, and b) an incremental learning algorithm that can consider a subset of the features during prediction (Naive Bayes). 

*I. Katakis, G. Tsoumakas, I. Vlahavas, Advances in Informatics: 10th Panhellenic Conference on Informatics, PCI 2005, Springer Berlin Heidelberg, 2005, Ch. On the Utility of Incremental Feature Selection for the Classification of Textual Data Streams, pp. 338–348.*


### Fast Correlation-Based Filter (FCBF)

FCBF is a multivariate feature selection method where the class relevance and the dependency between each feature pair are taken into account. Based on information theory, FCBF uses symmetrical uncertainty to calculate dependencies of features and the class relevance. Starting with the full feature set, FCBF heuristically applies a backward selection technique with a sequential search strategy to remove irrelevant and redundant features. The algorithm stops when there are no features left to eliminate.

*H.-L. Nguyen, Y.-K. Woon, W.-K. Ng, L. Wan, Heterogeneous ensemble for feature drifts in data streams, in: Proceedings of the 16th Pacific-Asia Conference on Advances in Knowledge Discovery and Data Mining - Volume Part II, PAKDD’12, 2012, pp. 1–12.*

### Online Feature Selection (OFS)

OFS proposes an ε-greedy online feature selection method based on weights generated by an online classifier (neural networks) which makes a trade-off between exploration and exploitation of features.

*J. Wang, P. Zhao, S. Hoi, R. Jin, Online feature selection and its applications, IEEE Transactions on Knowledge and Data Engineering 26 (3) (2014) 698–710.*


## Contact:

Sergio Ramírez Gallego (sramirez@decsai.ugr.es) - Department of Computer Science and Artificial Intelligence, University of Granada.

