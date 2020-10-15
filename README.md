# Optimal-transport

Optimization class project based on Optimal transport made with Thomas Chatrefou. 

Solving Monge-Kantorovich problem by treating it as a linear programming problem using its dual dual form and applying the simplex algorithm.
Then we transform the dual problem into an unconstrained optimization problem through regularization using the log-sum-exp function and solve it using gradient descent. 
We also do entropic regularization leading to a constrained optimization problem which we solve by minimizing the Lagrangian.

The pdf file contains the report
The .py file contains the code which we implemented
