# Multiple Simulated Annealing with Range Limitation
This is the official repository for the `KAIST 2024 Spring IE437 Assignment 2`.

I suggest some technique, called Multiple Simulated Annealing with Range Limitation, **MSA+RL** for blackbox optimization problem.

### Algorithm Overview
Range limitation is the most important element in this algorithm. This is a method to apply multiple search more effectively and explores more deeply the midpoint of the solution of the agent with the highest score. The boundary is gradually reduced so that useless input ranges can be removed. Since it is difficult to find a global optimal value at the beginning of the algorithm, the range limitation only applies a little at the beginning, but becomes more significant towards the end.

**Step 0** : Initialize $X \in [d \times n]$ to random value, with range $R = [d \times 2]$. $d$ is the input dimension, and the $n$ is the number of agent. And similarly to multiple simulated annealing, global optimal value initialize by $f_{optimal} \Leftarrow 0$.

**Step 1** : In the existing solution $x$, the nearest neighbor solution $ x' \Leftarrow x + \delta $ is randomly selected. But $x_{i}'$ that satisfy $R_{i0} \leq x_{i}' \leq R_{i1}$ for all $i$.  

**Step 2** : Calculate $\alpha \Leftarrow f(x') - f_{optimal}$.

**Step 3** : If $\alpha > 0$, then substitute the existing solution as a neighboring solution $x \Leftarrow x'$ and return to Step 1. Otherwise, Randomly re-initialize the solution $x$ with the probability $e^\frac{-\alpha}{T}$. Finally, Change the temperature according to the cooling rule by $T \Leftarrow Cooling(T)$.

**Step 4** : When enough solutions are obtained or the temperature has dropped sufficiently, $R \Leftarrow R / r$, where $r = l(i)$, $l$ is the $R \Rightarrow R$ function which return the range limitation value, $i$ is the number of iteration. Function $l$ is more greater when $i$ is more greater. And come back to Step 1 with newly initialized $x$ that satisfy $R_{i0} \leq x_{i} \leq R_{i1}$ for all $i$.

**Step 5** : When algorithm get enough solution, then algorithm done.


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