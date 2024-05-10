# Multiple Simulated Annealing with Range Limitation
This is the official repository for the `KAIST 2024 Spring IE437 Assignment 2`.

I suggest some technique, called Multiple Simulated Annealing with Range Limitation, **MSA+RL** for blackbox optimization problem.

### How to Run?
[SA.py](./classes/SA.py) is the main file for my algorithm. You can get solution for the assignment 2 by type below.

```bash
python result.py
```

There is a various coefficients for apply MSA+RL.

#### Object Function
The data consists of 6x9992 real numbers, and columns 'A', 'B', 'C', 'D', 'E' are input data and consist of real numbers between 0 and 1. and the 'score' column is the output data with real number.

There are four object functions for fitting data.

1. [Neural Network](./classes/NN.py)
2. [Neural Network with Batch Normalization](./classes/NN.py)
3. [Support Vector Regressor](./classes/SVR.py)
4. [Gradient Boosting Regressor](./classes/GR.py)

you can use above object functions by below.

###### [Neural Network](./classes/NN.py)
```python
from classes.NN import NN, NN_Model
model = NN_Model(hidden_layer = 64)
obj = NN(model)
```

###### [Neural Network with Batch Normalization](./classes/NN.py)
```python
from classes.NN import NN, NN_BN_Model
model = NN_BN_Model(hidden_layer = 64)
obj = NN(model)
```

###### [Support Vector Regressor](./classes/SVR.py)
```python
from classes.SVR import SVR_Model
obj = SVR(C = 2, gamma = 1, epsilon = 0.1)
```

###### [Gradient Boosting Regressor](./classes/GR.py)
```python
from classes.GR import GR_Model
obj = GR_Model(n_estimators=50, max_depth=100, learning_rate=0.1)
```

#### [Simulated Annealing](./classes/SA.py)
There are four Simulated Annealing algorithm.

1. Simulated Annleaing
2. Multiple Simulated Annealing
3. Simulated Annealing with Range Limitation
4. Multiple Simulated Annealing with Range Limitation

```python
from SA import Simulated_Annealing

obj = obj() # any object function
lims = 200 # the number of range limitation
n = 200 # the number of agent
range_lim = True # True when using range limitation

Simulated_Annealing(obj, lims, n, range_lim)
```
You can fix some coefficients with `lims`, `n`, `range_lim`. Additionally, You can edit basic coefficients, `COOLING`,`UPPER_LIMIT` or `LOWER_LIMIT` in [SA.py](./classes/SA.py).

for more information, see this file.