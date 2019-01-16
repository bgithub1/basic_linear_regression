## basic_linear_regression: 
A simple single layer pytorch model to show how ```torch.nn.Linear``` solves general multi-variate linear regression problems

___
### Idea:
The pytorch nn.Linear class can solve general Linear Equations of the kind:  $$\mathbf{y} = \mathbf{x}\mathbf{A^\top} + b$$ where:
* $\mathbf{A}$ is a transformation matrix which represent coefficients of the equation, 
* $\mathbf{x}$ is a vector of inputs and 
* $\mathbf{y}$ is a vector that is the result of the linear transformation

We are given: 
* rows of $\mathbf{x}$ values in a matrix $\mathbf{X}$,
* rows of $\mathbf{y}$ values in a matrix $\mathbf{Y}$

The model <span style="color:blue">SingleLayerNet</span> solves for the transformation matrix $\mathbf{A}$.

___
### Use:
##### In section 1.0
* set the scalar variable ```number_of_coefficients```
 * this variable determines the number of columns in Matrix $\mathbf{A}$
* set the scalar variable ```y_dimension```
 * the y_dimension determines:
   * the size of the $\mathbf{y}$ vector output of the linear transformation
   * this variable determines the number of rows in Matrix $\mathbf{A}$
* set the scalar bias term
* a scalar, (e.g. 0.1) in order to add some normally distributed random noise to the linear transformation

```
# exzmple from section 1.0
number_of_coefficients=3
y_dimension=number_of_coefficients
bias = 1
noise_level = 0.1
```

##### In section 2.0:
* run all cells from 2.0 on 

##### In section 3.0:
* run a single $\mathbf{x}$ vector test to see if the model's coefficients work
___