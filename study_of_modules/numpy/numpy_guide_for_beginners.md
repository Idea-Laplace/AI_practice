# NumPy: te absolute basics for beginners
Link:  https://numpy.org/doc/stable/user/absolute_beginners.html

## How to import numpy
---
	import numpy as np
---

This widespread convention allows access to NumPy features with a short, recognizable prefix (np.) while distinguishing NumPy features from others that have the same name.

## Reading the example code

```python
	>>> a = np.array([[1, 2, 3], [4, 5, 6]])   
	>>> type(a)    
	<class 'numpy.ndarray'>   
	>>> a.shape    
	(2, 3) # (row, column)    
```

## Why use NumPy?
Python lists are excellent, general-purpose containers. They can be “heterogeneous”, meaning that they can contain elements of a variety of types, and they are quite fast when used to perform individual operations on a handful of elements.

Depending on the characteristics of the data and the types of operations that need to be performed, other containers may be more appropriate; by exploiting these characteristics, we can improve speed, reduce memory consumption, and offer a high-level syntax for performing a variety of common processing tasks. NumPy shines when there are large quantities of “homogeneous” (same-type) data to be processed on the CPU.

## What is an “array”?
In computer programming, an array is a structure for storing and retrieving data. We often talk about an array as if it were a grid in space, with each cell storing one element of the data. For instance, if each element of the data were a number, we might visualize a “one-dimensional” array like a list:

 
A two-dimensional array would be like a table:

 
A three-dimensional array would be like a set of tables, perhaps stacked as though they were printed on separate pages. In NumPy, this idea is generalized to an arbitrary number of dimensions, and so the fundamental array class is called ndarray: it represents an “N-dimensional array”.

## Array attributes
This section covers the ndim, shape, size, and dtype attributes of an array.
<details>
<summary> ndim </summary>
<div markdown="1">
<br>The number of dimensions of an array is contained in the ndim attribute.</br>	
	>>> a.ndim </br>           
	2
</div>
</details>

<details>
<summary> shape </summary>
<div markdown="1">
contents would be filled later.
</div>
</details>


<details>
<summary> size </summary>
<div markdown="1">
contents would be filled later.
</div>
</details>

<details>
<summary> dtype </summary>
<div markdown="1">
contents would be filled later.
</div>
</details>