package moa.reduction.bayes;

/**
 * Mwg2DEPQ is an implementation of the double ended priority queue (DEPQ) interface
 * provided as part of the CS21100 (Program Design, Data Structures And Algorithms)
 * assignment at Aberystwyth University.
 *
 * DEPQ has been implemented using an interval heap which makes it very efficient.
 * Interval heap is similar to a traditional heap, however each node, represented by
 * the Interval inner class, contains two Comparable elements. Left element is
 * typically smaller (can be equal to) than the right one. If the total amount of
 * elements in the interval heap is odd, the last Interval will hold a second 'dummy'
 * element to make the implementation simpler. All left elements in the interval heap
 * define a min heap, whereas all right elements define a max heap. Therefore in the
 * root Interval the left element is the smallest one in the heap and the right one
 * is the largest one in the heap. Internally all elements are stored inside a
 * standard array which can be expanded if necessary. Size is being tracked in an
 * instance variable.
 *
 * TIME COMPLEXITY SUMMARY
 *
 * Each significant method has an associated complexity analysis in the JavaDoc,
 * however the summary of the efficiency of all methods is presented below.
 *
 * Method name    | Complexity 
 * ---------------+--------------- 
 * size()         | O(1) 
 * isEmpty()      | O(1) 
 * inspectLeast() | O(1) 
 * inspectMost()  | O(1) 
 * add()          | O(log n) 
 * getLeast()     | O(log n) 
 * getMost()      | O(log n)
 *
 * SOURCES
 *
 * In order to understand the concept of an interval heap and the way to implement
 * it, I extensively used this web site:
 * http://www.cise.ufl.edu/~sahni/dsaaj/enrich/c13/double.htm
 *
 * In order to see how the other developers tackled the problem I read through this
 * web site:
 * http://www.mhhe.com/engcs/compsci/sahni/enrich/c9/interval.pdf
 *
 * SELF EVALUATION
 *
 * I belive I have met all the marking criteria to receive a mark of around 80% in
 * each part of the assessment. My DEPQ is functioning correctly, has a very high
 * efficiency due to the way it has been implemented. Furthermore, the documentation
 * is clear an informative and each significant method has an associated complexity
 * analysis with the summary in the class-wide JavaDoc comment. Finally, I believe
 * that my code is clean an simple to understand, which means that I would hope to
 * receive an overall mark of around 80% for this assignment.
 *
 * @author Michal Goly, mwg2@aber.ac.uk
 * @version 1.0
 * @since 23/11/2015
 */
public class IntervalHeap2 {

   // An array of Intervals which contain two Comparable elements
   private Interval[] heap;

   // Number of Comparable objects in the interval heap
   private int size;

   /**
    * Creates the Mwg2DEPQ which is a double ended priority queue implemented using
    * an interval heap. This constructor sets the initial size of the queue to 0 and
    * initialises the empty heap array.
    */
   public IntervalHeap2() {
      heap = new Interval[1000];
      size = 0;
   }

   /**
    * Returns the smallest element in the DEPQ but does not remove it from the DEPQ.
    *
    * COMPLEXITY ANALYSIS
    *
    * Because we are only interested in the value of the smallest element, without
    * removing it, we only have to perform a single operation. Therefore, the time
    * complexity of inspecting the smallest element is constant O(1).
    *
    * @return Either the smallest element in the DEPQ, or null if one does not exist
    */
   
   public Comparable inspectLeast() {
      if (size == 0) {
         return null;
      } else {
         Interval root = heap[0];
         return root.getLeft();
      }
   }

   /**
    * Returns the largest element in the DEPQ but does not remove it from the DEPQ.
    *
    * COMPLEXITY ANALYSIS
    *
    * Because we are only interested in the value of the largest element, without
    * removing it, we only have to perform a single operation. Therefore, the time
    * complexity of inspecting the largest element is constant O(1).
    *
    * @return Either the largest element in the DEPQ, or null if one does not exist
    */
   
   public Comparable inspectMost() {
      if (size == 0) {
         return null;
      } else {
         Interval root = heap[0];
         return root.getRight();
      }
   }

   /**
    * Adds an element to the DEPQ.
    *
    * Method firstly checks if the size of the array which is used to store elements
    * is large enough to accommodate the new one. If it is not, the expand method
    * will be called to increase the capacity of the queue. Then, depending on the
    * current size of the queue, the Comparable element will either be placed in the
    * newly created root Interval (if size was 0), or it will be placed in the
    * appropriate position in the root Interval (if size was 1), otherwise its place
    * has to be calculated.
    *
    * We start by checking if the amount of elements in the queue is odd or even.
    *
    * If it is even, the new node needs to be created and its index has to be
    * calculated. Then the parent node of that new index has to be determined. Then,
    * we compare our Comparable element with the smaller element of its parent. If
    * Comparable is smaller than the smaller element of its parent, we know that it
    * should be placed somewhere in the min part of the interval heap. Therefore we
    * call the minHeapInsert method which will bubble the Comparable element into its
    * proper position, while shifting other elements if necessary. If Comparable is
    * larger than the smaller element of its parent, we call the maxHeapInsert method
    * to place the element into its correct position in the max heap part of the
    * interval heap. Finally, we duplicate the last element in the queue in order to
    * make other methods easier to implement.
    *
    * If it is odd, we do not have to create a new node, therefore we simply
    * calculate the last element, its parent, determine whether to put it in the min
    * or max part of the heap and call either minHeapInsert or maxHeapInsert
    * appropriately. We do not have to duplicate anything at the end, as the size
    * after the operation will be even.
    *
    * COMPLEXITY ANALYSIS
    *
    * In the best case scenario, the complexity is constant O(1) when the size of the
    * queue before the addition is either 0 or 1. Because we only need to perform a
    * single operation. In the worst case, we have to go through the whole heap from
    * bottom to the top. The height of the heap is equal to log(n), where n is the
    * amount of elements in the heap. Because the complexity of comparing elements
    * throughout the walk is constant, the worst case complexity of the add method is
    * O(log n).
    *
    * @param c The Comparable element to be inserted into the DEPQ
    */
   
   public void add(Comparable c) {
      if (size == heap.length) {
         expand();
      }

      if (size == 0) {
         heap[0] = new Interval(c, c);
         size++;
      } else if (size == 1) {
         if (c.compareTo(heap[0].getLeft()) < 0) {
            heap[0].setLeft(c);
         } else {
            heap[0].setRight(c);
         }
         size++;
      } else {
         // size >= 2
         if (size % 2 == 0) {
            int newNodeIndex = (size / 2) + (size % 2);
            int parentIndex = (newNodeIndex - 1) / 2;

            // initialize placeholder for that index in the array if used 1st time
            if (heap[newNodeIndex] == null) {
               heap[newNodeIndex] = new Interval();
            }

            // if c < parent.left
            if (c.compareTo(heap[parentIndex].getLeft()) < 0) {
               minHeapInsert(c, newNodeIndex);

               // duplicate the item in the last Interval
               heap[newNodeIndex].setRight(heap[newNodeIndex].getLeft());
            } else {
               maxHeapInsert(c, newNodeIndex);

               // duplicate the item in the last Interval
               heap[newNodeIndex].setLeft(heap[newNodeIndex].getRight());
            }
         } else {
            int lastNodeIndex = (size / 2) + (size % 2) - 1;

            if (c.compareTo(heap[lastNodeIndex].getLeft()) < 0) {
               minHeapInsert(c, lastNodeIndex);
            } else {
               maxHeapInsert(c, lastNodeIndex);
            }
         }
      }
   }

   /**
    * Removes the smallest element from the DEPQ and returns it.
    *
    * Method firstly checks if the queue is empty and returns null if it is. If the
    * queue holds only a single element, we return it and set the first Interval in
    * the heap to null and decrease the size appropriately. Otherwise we calculate an
    * index of the last Interval in the interval heap and compare it to 0. If the
    * last index is equal to 0, it means that the size is 2 and only the root
    * Interval exists. Therefore we return the left element of the root, which is the
    * smallest in the interval heap, assign root.right value to root.left and
    * decrease the size by 1. Finally if none of the above happened, our interval
    * heap holds more than 2 elements which means that we have to return the left
    * element of the root Interval (as it is the smallest in the DEPQ) and fix the
    * resulting heap.
    *
    * We start by saving the smallest element in the interval heap in a 'result'
    * variable, to return it back after the heap has been fixed. We then check if the
    * size of the heap is odd or even. If it is odd, we delete the last node by
    * setting it to null and decrease the 'size' and 'lastNodeIndex' variables by 1.
    * If it is even, we duplicate the right element in the last node and put it into
    * its left sibling, and decrease the 'size' by 1. We do not have to decrease the
    * last node index as we did not remove it in the first place. Finally, regardless
    * of whether the size was odd or even, we call the fixMinHeap method with passed
    * last index parameter. This method will fix the min heap part of the interval
    * heap and when it finishes, we can return the 'result' back to the caller of the
    * getLeast method.
    *
    * COMPLEXITY ANALYSIS
    *
    * Again, the removal itself of the smallest element has a constant complexity
    * O(1). However, we have to fix the resulting min heap in order to use it again
    * in the future. In the worst case, fixMinHeap method has to go through the whole
    * min heap from top to bottom, which has a height of log (n). Because the time
    * complexity of the comparisons it has to do along the way are constant, the
    * overall complexity of the getLeast method is O(log n).
    *
    * @return The smallest element in the DEPQ, or null if queue is empty
    */
   
   public Comparable getLeast() {
      Comparable result = null;

      if (isEmpty()) {
         return result;
      } else if (size == 1) {
         result = heap[0].getLeft();

         heap[0] = null;
         size--;

         return result;
      } else {
         result = heap[0].getLeft();

         int lastNodeIndex = (size / 2) + (size % 2) - 1;
         if (lastNodeIndex == 0) {
            // only root Interval exists
            heap[0].setLeft(heap[0].getRight());
            size--;
            return result;
         }

         // fix heap & start by moving the last element.left to the root.left
         heap[0].setLeft(heap[lastNodeIndex].getLeft());

         if (size % 2 == 0) {
            heap[lastNodeIndex].setLeft(heap[lastNodeIndex].getRight());
            size--;

            fixMinHeap(lastNodeIndex);
         } else {
            heap[lastNodeIndex] = null;
            size--;
            lastNodeIndex--;

            fixMinHeap(lastNodeIndex);
         }
         return result;
      }
   }

   /**
    * Removes the largest element from the DEPQ and returns it.
    *
    * Method firstly checks if the queue is empty and returns null if it is. If the
    * queue holds only a single element, we return it and set the first Interval in
    * the heap to null and decrease the size appropriately. Otherwise we calculate an
    * index of the last Interval in the interval heap and compare it to 0. If the
    * last index is equal to 0, it means that the size is 2 and only the root
    * Interval exists. Therefore we return the right element of the root, which is
    * the largest in the interval heap, assign root.left value to root.right and
    * decrease the size by 1. Finally if none of the above happened, our interval
    * heap holds more than 2 elements which means that we have to return the right
    * element of the root Interval (as it is the largest in the DEPQ) and fix the
    * resulting heap.
    *
    * We start by saving the largest element in the interval heap in a 'result'
    * variable, to return it back after the heap has been fixed. We then check if the
    * size of the heap is odd or even. If it is odd, we delete the last node by
    * setting it to null and decrease the 'size' and 'lastNodeIndex' variables by 1.
    * If it is even, we duplicate the left element in the last node and put it into
    * its right sibling, and decrease the 'size' by 1. We do not have to decrease the
    * last node index as we did not remove it in the first place. Finally, regardless
    * of whether the size was odd or even, we call the fixMaxHeap method with passed
    * last index parameter. This method will fix the max heap part of the interval
    * heap and when it finishes, we can return the 'result' back to the caller of the
    * getMost method.
    *
    * COMPLEXITY ANALYSIS
    *
    * Again, the removal itself of the largest element has a constant complexity
    * O(1). However, we have to fix the resulting max heap in order to use it again
    * in the future. In the worst case, fixMaxHeap method has to go through the whole
    * max heap from top to bottom, which has a height of log (n). Because the time
    * complexity of the comparisons it has to do along the way are constant, the
    * overall complexity of the getMost method is O(log n).
    *
    * @return The largest element in the DEPQ, or null if queue is empty
    */
   
   public Comparable getMost() {
      Comparable result = null;

      if (isEmpty()) {
         return result;
      } else if (size == 1) {
         result = heap[0].getLeft();

         heap[0] = null;
         size--;

         return result;
      } else {
         result = heap[0].getRight();

         int lastNodeIndex = (size / 2) + (size % 2) - 1;
         if (lastNodeIndex == 0) {
            // only root Interval exists
            heap[0].setRight(heap[0].getLeft());
            size--;
            return result;
         }

         // fix heap & start by moving the last element.right to the root.right
         heap[0].setRight(heap[lastNodeIndex].getRight());

         if (size % 2 == 0) {
            heap[lastNodeIndex].setRight(heap[lastNodeIndex].getLeft());
            size--;

            fixMaxHeap(lastNodeIndex);
         } else {
            heap[lastNodeIndex] = null;
            size--;
            lastNodeIndex--;

            fixMaxHeap(lastNodeIndex);
         }
      }

      return result;
   }

   /**
    * Checks if the DEPQ is empty.
    *
    * COMPLEXITY ANALYSIS
    *
    * We simply compare the current size of the queue with 0, which is a single
    * operation with a constant complexity O(1).
    *
    * @return true if the queue is empty, false otherwise
    */
   
   public boolean isEmpty() {
      return size == 0;
   }

   /**
    * Returns the size of the DEPQ.
    *
    * COMPLEXITY ANALYSIS
    *
    * We only return a single value which means that the time complexity of this
    * method is constant O(1).
    *
    * @return The number of Comparable elements currently in the DEPQ
    */
   
   public int size() {
      return size;
   }

   /**
    * This method will expand the capacity of the DEPQ by a factor of 2.
    */
   private void expand() {
      Interval[] newHeap = new Interval[heap.length * 2];
      System.arraycopy(heap, 0, newHeap, 0, heap.length);
      heap = newHeap;
   }

   /**
    * This method will bubble up the Comparable element into its proper position
    * within the min part of the interval heap, while shifting other elements if
    * necessary. It starts from the bottom and 'goes' upwards.
    *
    * @param c The Comparable element to be added into the queue
    * @param lastNodeIndex Index of the last Interval (node) in the queue
    */
   private void minHeapInsert(Comparable c, int lastNodeIndex) {

      int index = lastNodeIndex;
      while (index != 0 && c.compareTo(heap[(index - 1) / 2].getLeft()) < 0) {
         int parent = (index - 1) / 2;

         // move parent.left down (explicitly) and c up (implicitly)
         heap[index].setLeft(heap[parent].getLeft());
         index = parent;
      }

      heap[index].setLeft(c);
      size++;
   }

   /**
    * This method will bubble up the Comparable element into its proper position
    * within the max part of the interval heap, while shifting other elements if
    * necessary. It starts from the bottom and 'goes' upwards.
    *
    * @param c The Comparable element to be added into the queue
    * @param lastNodeIndex Index of the last Interval (node) in the queue
    */
   private void maxHeapInsert(Comparable c, int lastNodeIndex) {

      int index = lastNodeIndex;
      while (index != 0 && c.compareTo(heap[(index - 1) / 2].getRight()) > 0) {
         int parent = (index - 1) / 2;

         // move parent.right down (explicitly) and c up (implicitly)
         heap[index].setRight(heap[parent].getRight());
         index = parent;
      }

      heap[index].setRight(c);
      size++;
   }

   /**
    * This method will go through the min part of the interval heap and fix it.
    *
    * It starts at the root and compares left and right element to make sure that
    * smaller element is on the left side. If that is not the case, right and left
    * elements are swapped. Then the two children of the root node are calculated and
    * the smaller one is found. Then left element of the root is compared with the
    * left element of the smaller child. If the left element of the root is larger
    * than the left element of the child, they will be swapped. Finally the while
    * loop starts over, but this time the 'currentNode' became the value of the
    * previously smaller child index (so we effectively moved down the heap). Now the
    * algorithm will start over and fix the rest of the heap, until the current.left
    * has no children or their left elements are larger than the current.left.
    *
    * @param lastNodeIndex The last index of the Interval(node) in the heap
    */
   private void fixMinHeap(int lastNodeIndex) {
      int currentNode = 0;
      int smallerChildNode;
      while (currentNode <= lastNodeIndex) {

         // compare left and right and swap if left > right
         if (heap[currentNode].getLeft()
                 .compareTo(heap[currentNode].getRight()) > 0) {
            // swap
            Comparable temp = heap[currentNode].getLeft();
            heap[currentNode].setLeft(heap[currentNode].getRight());
            heap[currentNode].setRight(temp);
         }

         // find smaller child
         int leftChildNode = currentNode * 2 + 1;
         int rightChildNode = currentNode * 2 + 2;

         // stop if there are no children
         if (rightChildNode > lastNodeIndex && heap[leftChildNode] == null) {
            break;
         }

         // cover the special case when right child is null
         if (heap[rightChildNode] == null) {
            smallerChildNode = leftChildNode;
         } else if (heap[leftChildNode].getLeft()
                 .compareTo(heap[rightChildNode].getLeft()) <= 0) {
            smallerChildNode = leftChildNode;
         } else {
            smallerChildNode = rightChildNode;
         }

         // compare current.left with smaller.left
         if (heap[currentNode].getLeft()
                 .compareTo(heap[smallerChildNode].getLeft()) > 0) {
            // swap
            Comparable temp = heap[currentNode].getLeft();
            heap[currentNode].setLeft(heap[smallerChildNode].getLeft());
            heap[smallerChildNode].setLeft(temp);
         } else {
            break;
         }

         currentNode = smallerChildNode;
      }
   }

   /**
    * This method will go through the max part of the interval heap and fix it.
    *
    * It starts at the root and compares left and right element to make sure that
    * smaller element is on the left side. If that is not the case, right and left
    * elements are swapped. Then the two children of the root node are calculated and
    * the larger one is found. Then right element of the root is compared with the
    * right element of the larger child. If the right element of the root is smaller
    * than the right element of the child, they will be swapped. Finally the while
    * loop starts over, but this time the 'currentNode' became the value of the
    * previously larger child index (so we effectively moved down the heap). Now the
    * algorithm will start over and fix the rest of the heap, until the current.right
    * has no children or their right elements are smaller than the current.right.
    *
    * @param lastNodeIndex
    */
   private void fixMaxHeap(int lastNodeIndex) {
      int currentNode = 0;
      int largerChildNode;
      while (currentNode <= lastNodeIndex) {

         // compare left and right and swap if left > right
         if (heap[currentNode].getLeft()
                 .compareTo(heap[currentNode].getRight()) > 0) {
            // swap
            Comparable temp = heap[currentNode].getLeft();
            heap[currentNode].setLeft(heap[currentNode].getRight());
            heap[currentNode].setRight(temp);
         }

         // find smaller child
         int leftChildNode = currentNode * 2 + 1;
         int rightChildNode = currentNode * 2 + 2;

         // stop if there are no children
         if (rightChildNode > lastNodeIndex && heap[leftChildNode] == null) {
            break;
         }

         // take care of the special case when right child is null
         if (heap[rightChildNode] == null) {
            largerChildNode = leftChildNode;
         } else if (heap[leftChildNode].getRight()
                 .compareTo(heap[rightChildNode].getRight()) > 0) {
            largerChildNode = leftChildNode;
         } else {
            largerChildNode = rightChildNode;
         }

         // compare current.right with larger.right
         if (heap[currentNode].getRight()
                 .compareTo(heap[largerChildNode].getRight()) < 0) {
            // swap
            Comparable temp = heap[currentNode].getRight();
            heap[currentNode].setRight(heap[largerChildNode].getRight());
            heap[largerChildNode].setRight(temp);
         } else {
            break;
         }

         currentNode = largerChildNode;
      }
   }

   /**
    * Interval represents a single 'node' in the interval heap. It holds the
    * information about its two children. Left child should typically be smaller then
    * its right sibling. This should be enforced by the interval heap implementation.
    */
   private class Interval {

      // Left child in the interval
      private Comparable left;

      // Right child in the interval
      private Comparable right;

      /**
       * Creates a basic interval object with both of its children set to null
       */
      public Interval() {
      }

      /**
       * Creates the Interval object which represents a node in the interval heap.
       *
       * @param left The left child of the Interval (typically smaller)
       * @param right The right child of the Interval (typically larger)
       */
      public Interval(Comparable left, Comparable right) {
         this.left = left;
         this.right = right;
      }

      /**
       * @return The left child of the Interval
       */
      public Comparable getLeft() {
         return left;
      }

      /**
       * @return The right child of the Interval
       */
      public Comparable getRight() {
         return right;
      }

      /**
       * Assigns the new value to the left child
       *
       * @param left The new value to be assigned
       */
      public void setLeft(Comparable left) {
         this.left = left;
      }

      /**
       * Assigns the new value to the right child
       *
       * @param right The new value to be assigned
       */
      public void setRight(Comparable right) {
         this.right = right;
      }

   }

}
