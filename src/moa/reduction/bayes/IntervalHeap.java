package moa.reduction.bayes;

/**
 * <h1>DEPQ: Interval Heap</h1>
 * <p>
 * This program implements a DEPQ as an Interval Heap. An interval heap is a
 * data structure that allows for efficient removal of the minimum and maximum
 * values. It is possible to remove in both ascending and descending order, as
 * all elements have an inherent value.
 * <p>
 * In an interval heap, it acts as an embedded min-max heap, where each node
 * contains 2 elements. The "left" set of elements follows the rules of a min
 * heap, and the "right" set of elements follows the rules of a max heap. The
 * "left" element is always smaller then or equal to the right element. Each
 * node describes the range all children's values fall within.
 * <p>
 * Due to the nature of an interval heap, the farthest away any node can be from
 * the root is Log(n). This means that the program scales well with large
 * amounts of data, as in the worst case, the slowest operation with take Log(n)
 * time to complete. This worst case scenario of Log(n) for the slowest
 * operation makes this a very efficient deque when compared to other
 * implementations. Thanks to O(logn) being the worst case time complexity, the
 * average case is going to be even better that this, as not all operations will
 * need to go from the root to a leaf.
 * <p>
 * For this assignment, I believe I have achieved a grade of roughly 80%, as the
 * implementation I used is a strong one, with good complexities. My
 * implementation has no bugs to errors to the best of my knowledge, and is
 * robust. As far as the individual aspects go, I would mark them as such
 * <p>
 * Correct Functioning: 100%, as all methods implemented function and behave as
 * expected
 * <p>
 * Efficiency: 100%, as all my methods operate in the most efficient way
 * possible as listed in the spec "add, getMost and getLeast O(logN),
 * inspectLeast, inspectMost and getSize isEmpty O(1)".
 * <p>
 * Documentation based report: 80%, as I believe I have described my algorithm
 * in a large amount of depth. I believe my formatting to be correct in these
 * comments. For my comments I used the standards found at
 * http://www.oracle.com/
 * technetwork/java/javase/documentation/codeconventions-141999.html#385
 * <p>
 * Complexity Analysis: 70%, as I believe I have analysed my complexities
 * correctly for all methods as described and implemented. However, I am unsure
 * if I have gone into enough depth to warrant and outstanding mark for this
 * section.
 * <p>
 * Self evaluation: 100%, as I have estimated my mark for all sections of the
 * assignment mark scheme, and have commented on them all individually.
 * 
 * 
 * @author Shankly Richard Cragg (shc27)
 * @version 1.0
 * @since 2015-11-23
 */
public class IntervalHeap {

	int lastNode = 0;
	int spareSpace = 0;
	int numOfNodes = 1;
	Comparable[][] DEPQH;
	int size = 0;

	/**
	 * <h1>Constructor</h1> Constructor for my deque
	 */
	public IntervalHeap() {
		DEPQH = new Comparable[2][numOfNodes];
	}

	/**
	 * <h1>Expand</h1> Expand method to increase the size of the 2D array, on
	 * every increase it doubles the size of the DEPQ
	 */
	private void expand() {
		System.out.println("Expanding");

		/* Get's the total amount of elements in the array using both "rows" */
		int length = DEPQH[0].length + DEPQH[1].length;

		/* Create a new array which had double the number of nodes */
		Comparable[][] expanded = new Comparable[2][length];
		int j;
		int i;

		/* Moves all elements from current 2D array into larger 2D array */
		for (i = 0; i < length / 2; i++) {
			for (j = 0; j < 2; j++) {
				expanded[j][i] = DEPQH[j][i];
			}
		}
		DEPQH = expanded;
	}

	/**
	 * Updates the pointer looking at the last element. Always flips to spare
	 * bit as when something is removed it's space either becomes free, or it
	 * means a node is removed and there is no free space in the remaining node.
	 */
	public void removeElement(int Cl) {
		if (spareSpace == 1) {
			DEPQH[Cl][0] = DEPQH[spareSpace - 1][lastNode];
			DEPQH[spareSpace - 1][lastNode] = null;
		} else if (spareSpace == 0) {
			DEPQH[Cl][0] = DEPQH[spareSpace + 1][lastNode - 1];
			DEPQH[spareSpace + 1][lastNode - 1] = null;
		}

		/*
		 * If the last node has 2 elements, then lastNode needs to be pulled
		 * back to the node with now only 1 element
		 */
		if (spareSpace == 0) {
			lastNode = lastNode - 1;
			spareSpaceFlip();
		}

		/*
		 * If just 1 element in lastNode, then just there is no spareSpace is
		 * the new last node
		 */
		else if (spareSpace == 1) {
			spareSpaceFlip();
		}
	}

	/**
	 * Fetches the index of the child to the left of a parent node
	 * 
	 * @param parameter
	 *            The index of the parent node
	 * @return childLeftIndex The index of the left child
	 */
	public int getLeftChild(int parameter) {
		int parentIndex = parameter * 2;
		int childLeftIndex = parentIndex + 1;
		return childLeftIndex;
	}

	/**
	 * Fetches the index of the child to the right of a parent node
	 * 
	 * @param parameter
	 *            The index of the parent node
	 * @return childRightIndex The index of the right child
	 */
	public int getRightChild(int parameter) {
		int parentIndex = parameter * 2;
		int childRightIndex = parentIndex + 2;
		return childRightIndex;
	}

	/**
	 * Fetches the index of the parent of a a child node
	 * 
	 * @param parameter
	 *            The index of the parent node
	 * @return parent The index of the parent node
	 */
	public int getParent(int parameter) {
		int childIndex = parameter;
		int parent = (childIndex - 1) / 2;
		return parent;
	}

	/**
	 * Flips the value from a 1 to 0, or a 0 to a 1
	 */
	public void spareSpaceFlip() {
		if (spareSpace == 0) {
			spareSpace = 1;
		} else if (spareSpace == 1) {
			spareSpace = 0;
		}
	}

	/**
	 * <h1>Inspect Smallest Element</h1> Returns the smallest element in the
	 * DEPQ but does not remove it from the DEPQ
	 * <p>
	 * The time complexity for this method is O(1)
	 * 
	 * @return least Returns the smallest element in the DEPQ
	 */
	
	public Comparable inspectLeast() {
		Comparable least = DEPQH[0][0];
		System.out.println("Inspect Least is " + DEPQH[0][0]);
		return least;
	}

	/**
	 * <h1>Inspect Smallest Element</h1> Returns the largest element in the DEPQ
	 * but does not remove it from the DEPQ
	 * <p>
	 * The time complexity for this method is O(1)
	 * 
	 * @return most returns the largest element in the DEPQ
	 */
	
	public Comparable inspectMost() {
		Comparable most = DEPQH[1][0];
		//System.out.println("Inspect most is " + DEPQH[1][0]);

		/*
		 * If only 1 element in whole DEPQ (can tell if most is null), then the
		 * least is ALSO the most by definition
		 */
		if (most == null) {
			most = DEPQH[0][0];
		}
		return most;
	}

	/**
	 * <h1>Add Element</h1> Adds an element to the DEPQ. If there are an even
	 * number of elements, it is added to the "small" side of the last node. If
	 * there is an odd number of elements, it is added to the "large" side of
	 * the node. If this is the case, it was first compare to the "small"
	 * element in that node, and if it is smaller, they switch places before
	 * bubbling up the heap.
	 * <p>
	 * This has O(logn) time complexity in the worst case O(1) time complexity
	 * in the best case
	 * 
	 * @param c
	 *            the element to insert into the DEPQ
	 */
	
	public void add(Comparable c) {
		size++;

		/*
		 * If the size of the DEPQ is going to be will be larger than the 2D
		 * array can currently hold, expand the 2D array
		 */
		if (size > (DEPQH[0].length + DEPQH[1].length))
			expand();

		/* Use this to help find parent */
		int compareNode = lastNode;

		/* The node we are currently working in */
		int cr = lastNode;

		/* Whether we are looking in the "small" or "large" size of the array */
		int cl = spareSpace;
		Comparable swap = 0;
		Comparable num = c;
		// System.out.println("Add is " + num);

		/* Set the last item in 2D array to new item */
		DEPQH[spareSpace][lastNode] = num;

		/*
		 * If added to larger side of node, see if it needs to be moved to
		 * smaller side
		 */
		if (spareSpace == 1) {
			if ((DEPQH[cl][cr]).compareTo(DEPQH[0][compareNode]) < 0) {
				swap = DEPQH[cl][cr];
				DEPQH[cl][cr] = DEPQH[0][compareNode];
				DEPQH[0][compareNode] = swap;
				cl = 0;
			}

			/* If no more spare space (as it just filled), next node is end */
			lastNode++;
		}
		spareSpaceFlip();

		/* If we are in the parent node we don't do any comparisons */
		if (compareNode > 0) {
			do {
				compareNode = getParent(compareNode);

				/*
				 * If the element we are currently looking at is smaller than
				 * the small element of the parent, swap
				 */
				if ((DEPQH[cl][cr]).compareTo(DEPQH[0][compareNode]) < 0) {
					swap = DEPQH[cl][cr];
					DEPQH[cl][cr] = DEPQH[0][compareNode];
					DEPQH[0][compareNode] = swap;
					cl = 0;
				}

				/*
				 * If the element we are currently looking at is larger than the
				 * largest element of the parent, swap
				 */
				else if ((DEPQH[cl][cr]).compareTo(DEPQH[1][compareNode]) > 0) {
					swap = DEPQH[cl][cr];
					DEPQH[cl][cr] = DEPQH[1][compareNode];
					DEPQH[1][compareNode] = swap;
					cl = 1;
				}

				/*
				 * If it is within the range of it's parents, no more swaps need
				 * to occur
				 */
				else {
					break;
				}

				/* Sets the working node to the parent to we can bubble up */
				cr = compareNode;

				/* If we reach root (0) no where to swap to */
			} while (compareNode > 0);
		}
	}

	/**
	 * <h1>Return and Remove Smallest Element</h1> Removes the smallest element
	 * from the DEPQ and returns it. After it has been removed, the last element
	 * is put in it's place, and this element is bubbled down.
	 * 
	 * Removes the smallest element in the DEPQ (At position 0,0). This element
	 * is replaced with the last element in the DEPQ. I then find out the
	 * smallest element and the largest element the children of the node we are
	 * currently looking at. If both children do not contain 2 elements,
	 * considerations are made which ensure that no value is compared to NULL.
	 * If the element we are currently looking at breaks the rules of an
	 * interval heap (Meaning the children's values are not within the range of
	 * it's parent node) then an appropriate swap is made. This continues until
	 * the conditions of an interval heap are met, or the element enters a leaf
	 * and therefore cannot bubble any further.
	 * <p>
	 * This has O(logn) time complexity in the worst case O(1) time complexity
	 * in the best case
	 * 
	 * @return least Returns the smallest element in the DEPQ
	 */
	
	public Comparable getLeast() {
		Comparable least = DEPQH[0][0];
		// System.out.println("Get Least is " + least);
		int parentNode = 0;
		int currentPlace = 0;
		Comparable smallestChild = 0;
		Comparable largestChild = 0;
		Comparable smallest = null;
		Comparable largest = null;
		Comparable swap;

		/*
		 * Removes the last element that has been moved up to replace the
		 * "least" element
		 */
		removeElement(currentPlace);
		size--;

		/*
		 * If size < 3 then we must be in the parent node, and therefore no need
		 * to bubble
		 */
		if (size < 3) {
			return least;
		}

		/* The index of the children are fetched */
		int leftChild = getLeftChild(parentNode);
		int rightChild = getRightChild(parentNode);

		boolean LL;
		boolean LR;
		boolean RL;
		boolean RR;
		do {

			/*
			 * If [1][richChild] is not null, then both children have 2
			 * elements, and must compare both least and values of both children
			 * and most values of both children to see where the element we are
			 * bubbling may need to swap with
			 */
			if (DEPQH[1][rightChild] != null) {
				if ((DEPQH[0][leftChild]).compareTo(DEPQH[0][rightChild]) < 0) {
					smallest = DEPQH[0][leftChild];
					smallestChild = leftChild;
				} else {
					smallest = DEPQH[0][rightChild];
					smallestChild = rightChild;
				}
				if ((DEPQH[1][leftChild]).compareTo(DEPQH[1][rightChild]) > 0) {
					largest = DEPQH[1][leftChild];
					largestChild = leftChild;
				} else {
					largest = DEPQH[1][rightChild];
					largestChild = rightChild;
				}
			}

			/*
			 * If we get here, then it means that that largest child must be in
			 * the left child
			 */
			else if (DEPQH[0][rightChild] != null) {
				if ((DEPQH[0][leftChild]).compareTo(DEPQH[0][rightChild]) < 0) {
					smallest = DEPQH[0][leftChild];
					smallestChild = leftChild;
				} else {
					smallest = DEPQH[0][rightChild];
					smallestChild = rightChild;
				}
				largest = DEPQH[1][leftChild];
				largestChild = leftChild;
			}

			/*
			 * If we get here, then both largest and smallest must be in left
			 * child
			 */
			else if (DEPQH[1][leftChild] != null) {
				smallest = DEPQH[0][leftChild];
				smallestChild = leftChild;
				largest = DEPQH[1][leftChild];
				largestChild = leftChild;
			}

			/**
			 * If we get here then there is only 1 value in the children, and it
			 * is smallest. No largest needs to be set
			 */
			else if (DEPQH[0][leftChild] != null) {
				smallest = DEPQH[0][leftChild];
				smallestChild = leftChild;
			}

			/*
			 * If the element we are looking at is smaller than the smallest
			 * node, swap
			 */
			if (DEPQH[currentPlace][parentNode].compareTo(smallest) >= 0) {
				swap = DEPQH[currentPlace][parentNode];
				DEPQH[currentPlace][parentNode] = DEPQH[0][(int) smallestChild];
				DEPQH[0][(int) smallestChild] = swap;
				parentNode = (int) smallestChild;
			}

			/*
			 * As there may not have been a largest set earlier, we have to
			 * check if it's not null
			 */
			else if (largest != null
					&& DEPQH[currentPlace][parentNode].compareTo(largest) > 0) {
				swap = DEPQH[currentPlace][parentNode];
				DEPQH[currentPlace][parentNode] = DEPQH[0][(int) largestChild];
				DEPQH[1][(int) largestChild] = swap;
				parentNode = (int) largestChild;
			}

			/*
			 * This checks whether the node still has proper ordering, if not,
			 * swap node elements and current place changes
			 */
			if ((DEPQH[currentPlace][parentNode])
					.compareTo(DEPQH[0][parentNode]) < 0) {
				swap = DEPQH[currentPlace][parentNode];
				DEPQH[currentPlace][parentNode] = DEPQH[0][parentNode];
				DEPQH[0][parentNode] = swap;
			} else if (DEPQH[1][parentNode] != null
					&& (DEPQH[currentPlace][parentNode])
							.compareTo(DEPQH[1][parentNode]) > 0) {
				swap = DEPQH[currentPlace][parentNode];
				DEPQH[currentPlace][parentNode] = DEPQH[1][parentNode];
				DEPQH[1][parentNode] = swap;
			}

			leftChild = getLeftChild(parentNode);
			rightChild = getRightChild(parentNode);

			LL = false;
			LR = false;
			RL = false;
			RR = false;
			if (leftChild <= lastNode) {

				/**
				 * For this section, to avoid getting errors from comparing to a
				 * null value, I have a bunch of else ifs incrementally checking
				 * where the final element is in the heap, with each element
				 * less it has to make one less check.
				 * <p>
				 * These checks are to see if the children are both within the
				 * range of the current parent (of which the element we are
				 * currently looking at is located
				 */
				if (DEPQH[1][rightChild] != null) {
					if (DEPQH[currentPlace][parentNode]
							.compareTo(DEPQH[0][leftChild]) < 0) {
						LL = true;
					}
					if (DEPQH[currentPlace][parentNode]
							.compareTo(DEPQH[1][leftChild]) < 0) {
						RL = true;
					}
					if (DEPQH[currentPlace][parentNode]
							.compareTo(DEPQH[0][rightChild]) < 0) {
						LR = true;
					}
					if (DEPQH[currentPlace][parentNode]
							.compareTo(DEPQH[1][rightChild]) < 0) {
						RR = true;
					}
				} else if (DEPQH[0][rightChild] != null) {
					if (DEPQH[currentPlace][parentNode]
							.compareTo(DEPQH[0][leftChild]) < 0) {
						LL = true;
					}
					if (DEPQH[currentPlace][parentNode]
							.compareTo(DEPQH[1][leftChild]) < 0) {
						RL = true;
					}
					if (DEPQH[currentPlace][parentNode]
							.compareTo(DEPQH[0][rightChild]) < 0) {
						LR = true;
					}
					RR = true;
				} else if (DEPQH[1][leftChild] != null) {
					if (DEPQH[currentPlace][parentNode]
							.compareTo(DEPQH[0][leftChild]) < 0) {
						LL = true;
					}
					if (DEPQH[currentPlace][parentNode]
							.compareTo(DEPQH[1][leftChild]) < 0) {
						LR = true;
					}
					RL = true;
					RR = true;
				} else if (DEPQH[0][leftChild] != null) {
					if (DEPQH[currentPlace][parentNode]
							.compareTo(DEPQH[0][leftChild]) < 0) {
						LL = true;
					}
					LR = true;
					RL = true;
					RR = true;
				}
				if (DEPQH[0][leftChild] == null) {
					return least;
				}
			} else {
				return least;
			}

		} while (LL == false || LR == false || RL == false || RR == false);

		return least;
	}

	/**
	 * <h1>Return and Remove Largest Element</h1> Removes the largest element
	 * from the DEPQ and returns it. After it has been removed, the last element
	 * is put in it's place, and this element is bubbled down.
	 * 
	 * Removes the largest element in the DEPQ (At position 0,1) normally,
	 * however, if only 1 element exists in the DEPQ, then the largest value is
	 * at (0,0). This element is replaced with the last element in the DEPQ. I
	 * then find out the smallest element and the largest element the children
	 * of the node we are currently looking at. If both children do not contain
	 * 2 elements, considerations are made which ensure that no value is
	 * compared to NULL. If the element we are currently looking at breaks the
	 * rules of an interval heap (Meaning the children's values are not within
	 * the range of it's parent node) then an appropriate swap is made. This
	 * continues until the conditions of an interval heap are met, or the
	 * element enters a leaf and therefore cannot bubble any further.
	 * <p>
	 * This has O(logn) time complexity in the worst case O(1) time complexity
	 * in the best case
	 * 
	 * @return most Returns the smallest element in the DEPQ
	 */
	
	public Comparable getMost() {
		Comparable most = DEPQH[1][0];
		if (most == null) {
			most = DEPQH[0][0];
		}
		System.out.println("Get most is " + most);
		int parentNode = 0;
		int currentPlace = 1;
		Comparable smallestChild = 0;
		Comparable largestChild = 0;
		Comparable smallest = null;
		Comparable largest = null;
		Comparable swap;

		/*
		 * Removes the last element that has been moved up to replace the
		 * "least" element
		 */
		removeElement(currentPlace);
		size--;

		/*
		 * If size < 3 then we must be in the parent node, and therefore no need
		 * to bubble
		 */
		if (size < 3) {
			return most;
		}

		int leftChild = getLeftChild(parentNode);
		int rightChild = getRightChild(parentNode);

		boolean LL;
		boolean LR;
		boolean RL;
		boolean RR;
		do {

			/*
			 * If [1][richChild] is not null, then both children have 2
			 * elements, and must compare both least and values of both children
			 * and most values of both children to see where the element we are
			 * bubbling may need to swap with
			 */
			if (DEPQH[1][rightChild] != null) {
				if ((DEPQH[0][leftChild]).compareTo(DEPQH[0][rightChild]) < 0) {
					smallest = DEPQH[0][leftChild];
					smallestChild = leftChild;
				} else {
					smallest = DEPQH[0][rightChild];
					smallestChild = rightChild;

				}
				if ((DEPQH[1][leftChild]).compareTo(DEPQH[1][rightChild]) > 0) {
					largest = DEPQH[1][leftChild];
					largestChild = leftChild;
				} else {
					largest = DEPQH[1][rightChild];
					largestChild = rightChild;
				}
			}

			/*
			 * If we get here, then it means that that largest child must be in
			 * the left child
			 */
			else if (DEPQH[0][rightChild] != null) {
				if ((DEPQH[0][leftChild]).compareTo(DEPQH[0][rightChild]) < 0) {
					smallest = DEPQH[0][leftChild];
					smallestChild = leftChild;
				} else {
					smallest = DEPQH[0][rightChild];
					smallestChild = rightChild;
				}
				largest = DEPQH[1][leftChild];
				largestChild = leftChild;
			}

			/*
			 * If we get here, then both largest and smallest must be in left
			 * child
			 */
			else if (DEPQH[1][leftChild] != null) {
				smallest = DEPQH[0][leftChild];
				smallestChild = leftChild;
				largest = DEPQH[1][leftChild];
				largestChild = leftChild;
			}

			/**
			 * If we get here then there is only 1 value in the children, and it
			 * is smallest. No largest needs to be set
			 */
			else if (DEPQH[0][leftChild] != null) {
				smallest = DEPQH[0][leftChild];
				smallestChild = leftChild;
			}

			/*
			 * As there may not have been a largest set earlier, we have to
			 * check if it's not null
			 */
			if (largest != null
					&& DEPQH[currentPlace][parentNode].compareTo(largest) < 0) {
				swap = DEPQH[currentPlace][parentNode];
				DEPQH[currentPlace][parentNode] = DEPQH[1][(int) largestChild];
				DEPQH[1][(int) largestChild] = swap;
				parentNode = (int) largestChild;
			} else if (DEPQH[0][leftChild] == null) {
				return most;
			}

			/*
			 * If the element we are looking at is smaller than the smallest
			 * node, swap
			 */
			else if (DEPQH[currentPlace][parentNode].compareTo(smallest) < 0) {
				swap = DEPQH[currentPlace][parentNode];
				DEPQH[currentPlace][parentNode] = DEPQH[0][(int) smallestChild];
				DEPQH[0][(int) smallestChild] = swap;
				parentNode = (int) smallestChild;
			}

			/*
			 * This checks whether the node still has proper ordering, if not,
			 * swap node elements and current place changes
			 */
			if ((DEPQH[currentPlace][parentNode])
					.compareTo(DEPQH[0][parentNode]) < 0) {
				swap = DEPQH[currentPlace][parentNode];
				DEPQH[currentPlace][parentNode] = DEPQH[0][parentNode];
				DEPQH[0][parentNode] = swap;
			} else if (DEPQH[1][parentNode] != null
					&& (DEPQH[currentPlace][parentNode])
							.compareTo(DEPQH[1][parentNode]) > 0) {
				swap = DEPQH[currentPlace][parentNode];
				DEPQH[currentPlace][parentNode] = DEPQH[1][parentNode];
				DEPQH[1][parentNode] = swap;
			}
			leftChild = getLeftChild(parentNode);
			rightChild = getRightChild(parentNode);

			LL = false;
			LR = false;
			RL = false;
			RR = false;
			if (rightChild <= lastNode) {

				/**
				 * For this section, to avoid getting errors from comparing to a
				 * null value, I have a bunch of else ifs incrementally checking
				 * where the final element is in the heap, with each element
				 * less it has to make one less check.
				 * <p>
				 * These checks are to see if the children are both within the
				 * range of the current parent (of which the element we are
				 * currently looking at is located
				 */
				if (DEPQH[1][rightChild] != null) {
					if (DEPQH[currentPlace][parentNode]
							.compareTo(DEPQH[0][leftChild]) >= 0) {
						LL = true;
					}
					if (DEPQH[currentPlace][parentNode]
							.compareTo(DEPQH[1][leftChild]) >= 0) {
						RL = true;
					}
					if (DEPQH[currentPlace][parentNode]
							.compareTo(DEPQH[0][rightChild]) >= 0) {
						LR = true;
					}
					if (DEPQH[currentPlace][parentNode]
							.compareTo(DEPQH[1][rightChild]) >= 0) {
						RR = true;
					}
				} else if (DEPQH[0][rightChild] != null) {
					if (DEPQH[currentPlace][parentNode]
							.compareTo(DEPQH[0][leftChild]) >= 0) {
						LL = true;
					}
					if (DEPQH[currentPlace][parentNode]
							.compareTo(DEPQH[1][leftChild]) >= 0) {
						RL = true;
					}
					if (DEPQH[currentPlace][parentNode]
							.compareTo(DEPQH[0][rightChild]) >= 0) {
						LR = true;
					}
					RR = true;
				} else if (DEPQH[1][leftChild] != null) {
					if (DEPQH[currentPlace][parentNode]
							.compareTo(DEPQH[0][leftChild]) >= 0) {
						LL = true;
					}
					if (DEPQH[currentPlace][parentNode]
							.compareTo(DEPQH[1][leftChild]) >= 0) {
						LR = true;
					}
					RL = true;
					RR = true;
				} else if (DEPQH[0][leftChild] != null) {
					if (DEPQH[currentPlace][parentNode]
							.compareTo(DEPQH[0][leftChild]) >= 0) {
						LL = true;
					}
					LR = true;
					RL = true;
					RR = true;
				}

				if (DEPQH[0][leftChild] == null) {
					return most;
				}

			} else {
				return most;
			}

		} while (LL == false || LR == false || RL == false || RR == false);

		return most;
	}

	/**
	 * Checks if the DEPQ is empty, if the size is 0 it is empty, otherwise it
	 * is not
	 * <p>
	 * The time complexity for this method is O(1) in all cases
	 * 
	 * @return true/false Returns true if the queue is empty
	 */
	
	public boolean isEmpty() {
		if (size == 0) {
			return true;
		}
		return false;
	}

	/**
	 * Returns the size of the DEPQ
	 * <p>
	 * The time complexity for this method is O(1) in all cases
	 * 
	 * @return size Returns the number of elements currently in the DEPQ
	 */
	
	public int size() {
		return size;
	}

}