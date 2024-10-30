# Example of how to locally install and import the code
Navigate to the src folder with `setup.py`
```
conda activate (YOUR_ENVIRONMENT_NAME) # you can choose to skip this step for global install
cd decision-focused-learning/src
pip install -e .
```
now you can import like the jupyter notebook example `decision-focused-learning/notebooks/shortest_path_example.ipynb`

```
from decision_learning.data.shortest_path_grid import genData
from decision_learning.modeling.loss import SPOPlus
from decision_learning.modeling.models import LinearRegression
from decision_learning.modeling.train import train, calc_test_regret
```