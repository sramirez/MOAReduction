package moa.reduction.bayes;

import java.util.EmptyStackException;
import java.util.LinkedList;

public class IntervalHeap {

	
	LinkedList<Interval> container_;
	boolean isOdd_ = false;
	
	public IntervalHeap() {
		// TODO Auto-generated constructor stub
		container_ = new LinkedList<Interval>();
	}
	
	private boolean isOdd(int i) { return (i & 1) != 0;}
	private int leftChild(int parent) {
		return (parent << 1) + 1; // 2 * parent + 1
	}

	private int rightChild(int parent) {
		return (parent << 1) + 2; // 2 * parent + 2
	}	
	
	private int parent(int node) {
		return (node-1) / 2;
	}

	// container_.get(n).lower is empty.
	// v is to be inserted into the interval or a higher interval
	private void bubbleUpLower(int n, double v) {
		assert(n >= 0);
		while(n > 0) {
			int p = parent(n);
			if (v < container_.get(p).lower) {
	          container_.get(p).lower = container_.get(p).lower;
	          n = p;
	        } else {
	          break;
	        }
		}
	    // insert the value here
	    container_.get(n).lower = v;
	}
	
	
	  // container_.get(n).upper is empty.
	  // v is to be inserted into the interval or a higher interval
	  private void bubbleUpUpper(int n, double v) {
	    assert (n >= 0);

	    while (n > 0) {
	      int p = parent(n);
	      if (v > container_.get(p).upper) {
	        container_.get(n).upper = container_.get(p).upper;
	        n = p;
	      }
	      else {
	        break;
	      }
	    }
	    // insert the value here
	    container_.get(n).upper = v;
	  }	
	
	// container_.get(n).upper is empty.
	  // raise a value from below
	  private void bubbleDownUpper(int n) {
	    while (true) {
	      // loop until finished
	      int lc = leftChild(n);
	      if (lc >= container_.size()) {
	        // no children
	        if (n == container_.size() - 1) {
	          // the rightmost node
	          assert(!isOdd_); // cannot delete the upper bound from a single value interval!

	          // just invalidate the upper bound
	          isOdd_ = true;
	          return;
	        }
	        else {
	          // no children but not the rightmost node, so take a value from the rightmost node
	          double v;  // the value that is taken

	          if (isOdd_) {
	            v = container_.peekLast().lower;
	            container_.pollLast();
	            isOdd_ = false;
	          }
	          else {
	            v = container_.peekLast().upper;
	            isOdd_ = true;
	          }

	          // insert the borrowed value into this branch of the heap
	          if (v < container_.get(n).lower) {
	            // the upper value becomes the value of lower and the borrowed value becomes the new upper
	            // need to bubble this new upper value upwards
	        	  container_.get(n).upper = container_.get(n).lower;
	            bubbleUpLower(n, v);
	          }
	          else {
	            bubbleUpUpper(n, v);
	          }
	          return;
	        }
	      }
	      else {
	        int rc = rightChild(n);
	        if (rc >= container_.size()) {
	          // only the left child, which must be a leaf

	          if (lc == container_.size()-1 && isOdd_) {
	            // the child only has a lower bound so take it and delete the child node
	        	 container_.get(n).upper = container_.get(lc).lower;
	            container_.pollLast();
	            isOdd_ = false;
	            return;
	          }
	          else {
	            // take the child's upper bound and continue down to borrow the value from the last node
	            container_.get(n).upper = container_.get(lc).upper;
	            n = lc;
	          }
	        }
	        else {
	          // two children

	          // check whether the right child has only a lower bound
	          if (rc == container_.size()-1 && isOdd_) {
	            double rightLower = container_.get(rc).lower;

	            // the heap shrinks by disposing of the right node
	            container_.pollLast();
	            isOdd_ = false;

	            if (container_.get(lc).upper > rightLower) {
	              container_.get(n).upper = container_.get(lc).upper; // take the higher value

	              // insert the lower value into the left branch
	              if (container_.get(lc).lower > rightLower) {
	                // take value from right child and insert it as a lower bound in the left branch
	                container_.get(lc).upper = container_.get(lc).lower;

	                // insert the value
	                bubbleUpLower(lc, rightLower);
	                return;
	              }
	              else {
	                container_.get(lc).upper = rightLower;
	                return;
	              }
	            }
	            else {
	              container_.get(n).upper = rightLower;
	              return;
	            }
	          }
	          else {
	            // take the upper child's upper bound and continue down
	            if (container_.get(lc).upper > container_.get(rc).upper) {
	              container_.get(n).upper = container_.get(lc).upper;
	              n = lc;
	            }
	            else {
	              container_.get(n).upper = container_.get(rc).upper;
	              n = rc;
	            }
	          }
	        }
	      }
	    }
	  }
	  


	  // container_.get(n).lower is empty.
	  // raise a value from below
	  private void bubbleDownLower(int n) {
	    while (true) {
	      // loop until finished
	      
	      int lc = leftChild(n);
	      if (lc >= container_.size()) {
	        // no children
	        if (n == container_.size() - 1) {
	          // the rightmost node
	          if (isOdd_) {
	            // the only value in the node so simply delete it
	            container_.pollLast();
	            isOdd_ = false;
	            return;
	          }
	          else {
	            container_.get(n).lower = container_.get(n).upper;
	            isOdd_ = true;
	            return;
	          }
	        }
	        else {
	          // no children but not the rightmost node, so take a value from the rightmost node
	          double v = container_.peekLast().lower;

	          // insert the borrowed value into this branch of the heap
	          if (v > container_.get(n).upper) {
	            // the upper value becomes the value of lower and the borrowed value becomes the new upper
	            // need to bubble this new upper value upwards
	            container_.get(n).lower = container_.get(n).upper;
	            bubbleUpUpper(n, v);
	          }
	          else {
	            bubbleUpLower(n, v);
	          }

	          // remove the borrowed value from the end of the heap
	          if (isOdd_) {
	            // there is only one value in the rightnode so simply delete the node
	            container_.pollLast();
	            isOdd_ = false;
	          }
	          else {
	            container_.peekLast().lower = container_.peekLast().upper;
	            isOdd_ = true;
	          }

	          return;
	        }
	      }
	      else {
	        int rc = rightChild(n);
	        if (rc >= container_.size()) {
	          // only the left child

	          // take the child's lower bound and continue down
	          container_.get(n).lower = container_.get(lc).lower;
	          n = lc;
	        }
	        else {
	          // two children

	          // take the lower child's lower bound and continue down
	          if (container_.get(lc).lower < container_.get(rc).lower) {
	            container_.get(n).lower = container_.get(lc).lower;
	            n = lc;
	          }
	          else {
	            container_.get(n).lower = container_.get(rc).lower;
	            n = rc;
	          }
	        }
	      }
	    }
	  }
	  
	  // container_.get(n).upper has been decreased to v, so bubble down if necessary.
	  private void bubbleDownUpper(int n, double v) {
	    if (n == container_.size()-1 && isOdd_) {
	      // special case, we are inserting at a single value interval - just insert and we are done
	      container_.get(n).lower = v;
	      return;
	    }

	    //interval* current;

	    //while (v > (current = &container_.get(n))->lower) {
	    while (v > container_.get(n).lower) {
	      Interval current = container_.get(n);
	      int lc = leftChild(n);

	      if (lc >= container_.size()) {
	        // no children

	        // just insert the value and we are done
	    	current.upper = v;
	        return;
	      }
	      else {
	        int rc = rightChild(n);
	        if (rc >= container_.size()) {
	          // only the left child - insert in the correct place and we are done

	          if (lc == container_.size()-1 && isOdd_) {
	            // the child only has a lower bound
	            // insert in the appropriate place and we are done
	            //double leftChildLower = &container_.get(lc).lower;
	        	double leftChildLower = container_.get(lc).lower;  
	            if (v < leftChildLower) {
	              current.upper = leftChildLower;
	              leftChildLower = v;
	            } else {
	              current.upper = v;
	            }
	          }
	          else {
	            if (v < container_.get(lc).upper) {
	              // take the child's upper bound and check whether to insert as upper or lower
	              current.upper = container_.get(lc).upper;
	              if (container_.get(lc).lower > v) {
	                container_.get(lc).upper = container_.get(lc).lower;
	                container_.get(lc).lower = v;
	              }
	              else {
	                container_.get(lc).upper = v;
	              }
	            }
	            else {
	              // insert here and we are done
	              current.upper = v;
	            }
	          }

	          return;
	        }
	        else {
	          // two children
	          double leftUpper = container_.get(lc).upper;

	          if (rc == container_.size()-1 && isOdd_) {
	            // the right child has only a lower bound
	            double rightLower = container_.get(rc).lower;

	            // swap it if it is greater than v
	            if (rightLower > v) {
	              double tmp = rightLower;
	              container_.get(rc).lower = v;
	              v = tmp;
	            }

	            // then check the left branch
	            if (leftUpper > v) {
	              // swap the values - no need to recurse because there can't be any children
	              current.upper = leftUpper;
	              container_.get(lc).upper = v;
	            }
	            else {
	              // insert here
	              current.upper = v;
	            }

	            return;
	          }
	          else {
	            // check against the greater child
	            double rightUpper = container_.get(rc).upper;

	            if (leftUpper > rightUpper) {
	              if (leftUpper > v) {
	                // lift the value and bubble down
	                current.upper = leftUpper;
	                n = lc;
	              }
	              else {
	                // insert here and we are done
	                current.upper = v;
	                return;
	              }
	            }
	            else {
	              if (rightUpper > v) {
	                // lift the value and bubble down
	                current.upper = rightUpper;
	                n = rc;
	              }
	              else {
	                // insert here and we are done
	                current.upper = v;
	                return;
	              }
	            }
	          }
	        }
	      }
	    }

	    // v is less than the lower bound so rotate the lower bounds down to a root and the upper bounds up to here then insert v as the lower bound here
	    assert(v <= container_.get(n).lower);

	    cycleLowerToUpper(v, n);
	  }

	  // a new value is being inserted into current.lower and current.upper is empty
	  // shuffle values of lower down and values of upper up accordingly
	  private void cycleLowerToUpper(double v, int n) {
	    while (true) {
	      Interval current = container_.get(n);
	      int lc = leftChild(n);

	      if (lc >= container_.size()) {
	        // no children

	        // just insert the value and we are done
	        current.upper = current.lower;
	        current.lower = v;
	        return;
	      }
	      else {
	        int rc = rightChild(n);
	        if (rc >= container_.size()) {
	          // only the left child which must be a leaf so cycle through it and we are done

	          if (lc == container_.size()-1 && isOdd_) {
	            // the child only has a lower bound
	            // cycle to the appropriate places and we are done
	            double leftChildLower = container_.get(lc).lower;

	            current.upper = leftChildLower;
	            leftChildLower = current.lower;
	            current.lower = v;
	          }
	          else {
	            // cycle
	            current.upper = container_.get(lc).upper;
	            container_.get(lc).upper = container_.get(lc).lower;
	            container_.get(lc).lower = current.lower;
	            current.lower = v;
	          }

	          return;
	        }
	        else {
	          // two children

	          double leftUpper = container_.get(lc).upper;

	          if (rc == container_.size()-1 && isOdd_) {
	            // the right child has only a lower bound and both children are leaves
	            double rightLower = container_.get(rc).lower;

	            // cycle through the node with the greater value for upper and we are done
	            if (rightLower >= leftUpper) {
	              current.upper = rightLower;
	              container_.get(rc).lower = current.lower;
	            }
	            else {
	              current.upper = leftUpper;
	              container_.get(lc).upper = container_.get(lc).lower;
	              container_.get(lc).lower = current.lower;
	            }

	            current.lower = v;

	            return;
	          }
	          else {
	            // cycle through the child with the greater upper bound
	            double rightUpper = container_.get(rc).upper;
	            

	            if (leftUpper > rightUpper) {
	              current.upper = leftUpper;
	              n = lc;
	            }
	            else {
	              current.upper = rightUpper;
	              n = rc;
	            }

	            double tmp = current.lower;
	            current.lower = v;
	            v = tmp;
	          }
	        }
	      }
	    }
	  }

	  // container_.get(n).lower has been increased to v, so bubble down if necessary.
	  private void bubbleDownLower(int n, double v) {
	    if (n == container_.size() - 1 && isOdd_) {
	      // special case, we are inserting at a single value interval - just insert and we are done
	      container_.get(n).lower = v;
	      return;
	    }

	    Interval current;

	    while (v < (current = container_.get(n)).upper) { // cannot be a single element node
	      int lc = leftChild(n);

	      if (lc >= container_.size()) {
	        // no children

	        // just insert the value and we are done
	        current.lower = v;
	        return;
	      }
	      else {
	        int rc = rightChild(n);
	        if (rc >= container_.size()) {
	          // only the left child - insert in the correct place and we are done

	          if (lc == container_.size()-1 && isOdd_) {
	            // the child only has a lower bound
	            // insert in the appropriate place and we are done
	            double leftChildLower = container_.get(lc).lower;

	            if (v > leftChildLower) {
	              current.lower = leftChildLower;
	              leftChildLower = v;
	            }
	            else {
	              current.lower = v;
	            }
	          }
	          else {
	            if (v > container_.get(lc).lower) {
	              // take the child's upper bound and check whether to insert as upper or lower
	              current.lower = container_.get(lc).lower;
	              if (container_.get(lc).upper < v) {
	                container_.get(lc).lower = container_.get(lc).upper;
	                container_.get(lc).upper = v;
	              }
	              else {
	                container_.get(lc).lower = v;
	              }
	            }
	            else {
	              // insert here and we are done
	              current.lower = v;
	            }
	          }

	          return;
	        }
	        else {
	          // two children

	          double leftLower = container_.get(lc).lower;

	          if (rc == container_.size()-1 && isOdd_) {
	            // the right child has only a lower bound
	            double rightLower = container_.get(rc).lower;

	            // swap it if it is less than v
	            if (rightLower < v) {
	              double tmp = rightLower;
	              container_.get(rc).lower = v;
	              v = tmp;
	            }

	            // then check the left branch
	            if (leftLower < v) {
	              // swap the values - no need to recurse because there can't be any children
	              current.lower = leftLower;
	              container_.get(lc).lower = v;
	            }
	            else {
	              // insert here
	              current.lower = v;
	            }

	            return;
	          }
	          else {
	            // check against the lower child
	            double rightLower = container_.get(rc).lower;

	            if (leftLower < rightLower) {
	              if (leftLower < v) {
	                // lift the value and bubble down
	                current.lower = leftLower;
	                n = lc;
	              }
	              else {
	                // insert here and we are done
	                current.lower = v;
	                return;
	              }
	            }
	            else {
	              if (rightLower < v) {
	                // lift the value and bubble down
	                current.lower = rightLower;
	                n = rc;
	              }
	              else {
	                // insert here and we are done
	                current.lower = v;
	                return;
	              }
	            }
	          }
	        }
	      }
	    }

	    // v is greater than the upper bound so rotate the upper bound down to a leaf and the lower bound up to here then insert v as the auuper bound here
	    assert(v >= current.upper);

	    cycleUpperToLower(v, n);
	  }

	  // a new value is being inserted into current.lower and current.upper is empty
	  // shuffle values of lower down and values of upper up accordingly
	  private void cycleUpperToLower(double v, int n) {
	    while (true) {
	      Interval current = container_.get(n);
	      int lc = leftChild(n);

	      if (lc >= container_.size()) {
	        // no children

	        // just insert the value and we are done
	        current.lower = current.upper;
	        current.upper = v;
	        return;
	      }
	      else {
	        int rc = rightChild(n);
	        if (rc >= container_.size()) {
	          // only the left child which must be a leaf so cycle through it and we are done

	          if (lc == container_.size()-1 && isOdd_) {
	            // the child only has a lower bound
	            // cycle to the appropriate places and we are done
	            double leftChildLower = container_.get(lc).lower;

	            current.lower = leftChildLower;
	            leftChildLower = current.upper;
	            current.upper = v;
	          }
	          else {
	            // cycle
	            current.lower = container_.get(lc).lower;
	            container_.get(lc).lower = container_.get(lc).upper;
	            container_.get(lc).upper = current.upper;
	            current.upper = v;
	          }

	          return;
	        }
	        else {
	          // two children

	          double leftLower = container_.get(lc).lower;

	          if (rc == container_.size()-1 && isOdd_) {
	            // the right child has only a lower bound and both children are leaves
	            double rightLower = container_.get(rc).lower;

	            // cycle through the node with the lesser value for lower and we are done
	            if (rightLower <= leftLower) {
	              current.lower = rightLower;
	              container_.get(rc).lower = current.upper;
	            }
	            else {
	              current.lower = leftLower;
	              container_.get(lc).lower = container_.get(lc).upper;
	              container_.get(lc).upper = current.upper;
	            }

	            current.upper = v;

	            return;
	          }
	          else {
	            // cycle through the child with the lower lower bound
	            double rightLower = container_.get(rc).lower;	            

	            if (leftLower < rightLower) {
	              current.lower = leftLower;
	              n = lc;
	            }
	            else {
	              current.lower = rightLower;
	              n = rc;
	            }

	            double tmp = current.upper;
	            current.upper = v;
	            v = tmp;
	          }
	        }
	      }
	    }
	  }
	  
	// empty the queue
	  public void clear() {
	    container_.clear();
	    isOdd_ = false;
	  }
	  
	  public double inspectLeast() {
	    assert(!empty());

	    return container_.get(0).lower;
	  }
	  
	  public double inspectMost() {
	    assert(!empty());

	    if (size() == 1) return container_.get(0).lower;
	    else return container_.get(0).upper;
	  }

	  // add an element
	  public void add(double v) {
	    if (isOdd_) {
	      // insert v into the last interval which contains only one value
	      if (container_.peekLast().lower > v) {
	        container_.peekLast().upper = container_.peekLast().lower;
	        bubbleUpLower(container_.size()-1, v);
	      }
	      else {
	        bubbleUpUpper(container_.size()-1, v);
	      }
	      isOdd_ = false;
	    }
	    else {
	      // create a new interval containing only one element
	      int n = container_.size();          // the index of the new container

	      //container_.resize(n+1);
	      container_.addLast(new Interval());
	      isOdd_ = true;
	      if (container_.size() == 1) {
	        // container was empty so job done
	        container_.peekFirst().lower = v;
	      }
	      else {
	        // check whether we need to bubble up
	        int p = parent(n);

	        if (v < container_.get(p).lower) {
	          container_.get(n).lower = container_.get(p).lower;
	          bubbleUpLower(p, v);
	        }
	        else if (v > container_.get(p).upper) {
	          container_.get(n).lower = container_.get(p).upper;
	          bubbleUpUpper(p, v);
	        }
	        else {
	          container_.get(n).lower = v;
	        }
	      }
	    }
	  }

	  // remove the lowest value element
	  public double getLeast() {
	    if (empty()) throw new EmptyStackException();

	    double v = container_.get(0).lower;

	    bubbleDownLower(0);

	    return v;
	  }

	  // remove the highest value element
	  public double getMost() {
	    if (empty()) throw new EmptyStackException();

	    if (size() == 1) {
	      double v = container_.get(0).lower;

	      container_.clear();
	      isOdd_ = false;
	      return v;
	    }
	    else {
	      double v = container_.get(0).upper;

	      bubbleDownUpper(0);

	      return v;
	    }
	  }

	  // true iff the queue is empty
	  public boolean empty() {
	    return container_.isEmpty();
	  }

	  public int size() {
	    int size = 2 * container_.size();

	    if (isOdd_) return size - 1;
	    else return size;
	  }

	  // remove an element by index (intended to remove a random element where the random selection has been done externally
	  public void remove(int index) {
	    int interval = index / 2;
	    if (interval >= container_.size()) throw new IndexOutOfBoundsException();
	    if (interval == container_.size()-1) {
	      // last container - special case

	      if (isOdd_) {
	        // the very last element, so just remove it

	        if (isOdd(index)) throw new IndexOutOfBoundsException();

	        isOdd_ = false;
	        container_.pollLast();
	      }
	      else {
	        isOdd_ = true; // remove the upper bound
	        
	        if (!isOdd(index)) {
	          // need to remove the lower bound, so move the upper bound into its slot
	          container_.get(interval).lower = container_.get(interval).upper;
	        }
	      }
	    }
	    else {
	      // borrow a value from the last interval
	      double borrowed;

	      if (isOdd_) {
	        // last interval has only one value, so delete it
	        borrowed = container_.peekLast().lower;
	        isOdd_ = false;
	        container_.pollLast();
	      }
	      else {
	        // borrow the upper value
	        borrowed = container_.peekLast().upper;
	        isOdd_ = true;
	      }

	      if (isOdd(index)) {
	        // deleting the upper value
	        if (borrowed != container_.get(interval).upper) {
	          // no need to do anything if the borrowed value is the same as the removed value

	          if (borrowed > container_.get(interval).upper) {
	            // need to bubble up upper
	            bubbleUpUpper(interval, borrowed);
	          }
	          else if (borrowed >= container_.get(interval).lower) {
	            // bubble the borrowed value downwards
	            bubbleDownUpper(interval, borrowed);
	          }
	          else {
	            // need to make the lower value into the new upper and bubble down
	            bubbleDownUpper(interval, container_.get(interval).lower);
	            // and bubble up the new lower
	            bubbleUpLower(interval, borrowed);
	          }
	        }
	      }
	      else {
	        // deleting the lower value

	        if (borrowed != container_.get(interval).lower) {
	          // no need to do anything if the borrowed value is the same as the removed value
	          if (borrowed < container_.get(interval).lower) {
	            // need to bubble up lower
	            bubbleUpLower(interval, borrowed);
	          }
	          else if (borrowed <= container_.get(interval).upper) {
	            // bubble the borrowed value downwards
	            bubbleDownLower(interval, borrowed);
	          }
	          else {
	            // need to make the upper value into the new lower and bubble down
	            bubbleDownLower(interval, container_.get(interval).upper);
	            // and bubble up the new upper
	            bubbleUpUpper(interval, borrowed);
	          }
	        }
	      }
	    }
	  }

	  // replace an element by index (intended to replace a random element where the random selection has been done externally
	  public void replace(int index, double v) {
	    int interval = index / 2;
	    if (interval >= container_.size()) throw new IndexOutOfBoundsException();
	    if (interval == container_.size()-1 && isOdd_) {
	      // last container with 1 element - special case
	      if (isOdd(index)) throw new IndexOutOfBoundsException();

	      if (interval == 0) {
	        // replacing the only value
	        container_.get(0).lower = v;
	      }
	      else {
	        int p = parent(interval);
	        if (v > container_.get(p).upper) {
	          container_.get(interval).lower = container_.get(p).upper;
	          bubbleUpUpper(p, v);
	        }
	        else if (v < container_.get(p).lower) {
	          container_.get(interval).lower = container_.get(p).lower;
	          bubbleUpLower(p, v);
	        }
	        else {
	          container_.get(interval).lower = v;
	        }
	      }
	    }
	    else {
	      if (isOdd(index)) {
	        // deleting the upper value
	        if (v != container_.get(interval).upper) {
	          // no need to do anything if the value is the same as the removed value

	          if (v > container_.get(interval).upper) {
	            // need to bubble up upper
	            bubbleUpUpper(interval, v);
	          }
	          else if (v >= container_.get(interval).lower) {
	            // bubble the value downwards
	            bubbleDownUpper(interval, v);
	          }
	          else {
	            // need to make v the new lower and bubble up
	            double tmp = container_.get(interval).lower;
	            bubbleUpLower(interval, v);
	            // and make the old lower value into the new upper and bubble down
	            bubbleDownUpper(interval, tmp);
	          }
	        }
	      }
	      else {
	        // deleting the lower value

	        if (v != container_.get(interval).lower) {
	          // no need to do anything if the value is the same as the removed value
	          if (v < container_.get(interval).lower) {
	            // need to bubble up lower
	            bubbleUpLower(interval, v);
	          }
	          else if (v <= container_.get(interval).upper) {
	            // bubble the value downwards
	            bubbleDownLower(interval, v);
	          }
	          else {
	            // need to make v the nw upper and bubble up
	            double tmp = container_.get(interval).upper;
	            bubbleUpUpper(interval, v);
	            // and make the old upper value into the new lower and bubble down
	            bubbleDownLower(interval, tmp);
	          }
	        }
	      }
	    }
	  }

	  // replace the minimum value with the specified value
	  public void replaceMin(double v) { replace(0, v); }

	  // replace the maximum value with the specified value
	  public void replaceMax(double v) { replace(1, v); }

	 
	  // print the underlying vector - for debugging - printF is a function for printing one of the base type
	  /*public void dump(void (*printF)(_Ty)) const {
	    for (int i = 0; i < container_.size(); i++) {
	      if (i == container_.size()-1 && isOdd_) {
	        printF(container_[i].lower);
	      }
	      else {
	        printF(container_[i].lower);
	        printF(container_[i].upper);
	      }
	    }
	  }*/
	  
	  // test whether the DEPQ is well stuctured
	  public boolean test(Interval bounds, int node) {
	    if (node >= container_.size()) return true;
	    else if (node == container_.size()-1 &&
	             isOdd_ &&
	             (container_.get(node).lower < bounds.lower ||
	              bounds.upper < container_.get(node).lower)) {
	                return false;  // special case for last interval containing only one element
	    }
	    if (container_.get(node).lower < bounds.lower) {
	      return false;          // lower must not not be lower than the containing interval
	    }
	    if (bounds.upper < container_.get(node).upper) {
	      return false;         // the containing interval must not not be lower than upper
	    }
	    if (!test(container_.get(node), leftChild(node))) {
	      return false;              // the left child must be correct
	    }
	    return test(container_.get(node), rightChild(node));                         // the right child must be correct
	  }
	
	class Interval {
		double lower;
		double upper;

		
		public Interval() {}
		
		public Interval(double lower, double upper) {
			// TODO Auto-generated constructor stub
			this.lower = lower;
			this.upper = upper;
		}
		
		
	}
}
