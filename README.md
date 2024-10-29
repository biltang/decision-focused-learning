# Design choices behind Loss Function, Optimization Model Setup
Each loss function has uniquely different set of inputs they require and solve for different objective functions. 
For below, we have following notation
- cp = cost predicted, c_hat
- wp = decision var solution to opt problem wih cp
- zp/z(cp) = solution to opt problem with cost vector cp
- c = true cost
- w = optimal solution variables to opt problem with true cost vector c
- z = optimal/true obj value

Examples:
- SPO (ref: https://arxiv.org/pdf/2206.14234 pg 11, sec 3.4.1)
    - inputs: cp, c, w, z
    - opt problem cost vector: 2 * cp - c
    - opt problem obj: zp = (2*cp - c) wp
    - loss: -zp + 2 * cp * w - z
 
- PG (Forward, Backwards, Central Variants) (ref: https://arxiv.org/pdf/2402.03256 pg 4, sec 2)
    - inputs: cp, c, h (window-width parameter)
    - opt problem cost vector (solve twice): 
        - backwards: cp, cp - h*c
        - central: cp + h*c, cp - h*c
    - opt problem obj: 
        - backwards: z_plus = cp*wp, z_minus = (cp - h*c)*wp
        - central: z_plus = (cp + h*c)*wp, z_minus = (cp - h*c)*wp
    - loss: 
        - backwards: (1/h)*(z_plus - z_minus)

- Cosine/MSE
    - inputs: cp, c
    - does not require solving opt problem or custom loss function behavior

We see that the behavior of the loss function (in terms of its inputs), optimization problem (in terms of the setting of its objective function) can differ between different use cases. Here are the important components:
1. data - each loss function requires different pieces of data as input, either into the optimization model, or to calculate special gradients. 
2. prediction model (this can be relatively siloed away from the other components since no matter what problem, loss function, opt model, it always just outputs the cost vector. We can therefore use different training orchestration functions for this)
3. optimization model - generally this requires just setting an appropriate cost function vector once the problem setting is specified (since the constraints will be fixed in our settings). However an appropriate cost vector differs between different loss functions and the number of times the optimization model will be called is also different. Also a major issue is allowing flexibility for batch solving vs iterative solving. For example pyepo expects to only solve 1 problem at a time and leverages a wrapper/orchestration function to iterate thru batches. 
4. loss function - requires differnet inputs for different calcs

Problems to consider:
1. Ideally you'd want the behavior of the optimization model to be separated from the loss function itself, and you can directly supply the components necesary for the loss function. However how the optimizaiton model is used at each data sample (a instantiation of a optimization problem) and at each epoch can differ between loss functions. Therefore there is a natural dependency between the opt model usage and the loss function. 
2. Furtheremore the orchestration training function needs to handle the case where different loss functions require different inputs, hence different function signatures. Having if-else statements may be unwieldy, and this is the case for pyepo because they have a universal dataset object that always returns mini-batches of [cp, c, w, z]

Proposed Solutions
1. For problem 1 there are 3 possible decisions points
    1. Keep opt behavior relatively uniform between different versions (in terms of inputs/outputs) since our loss functions need to expect a uniform behavior. In this case you'd need to define a wrapper function for handling different behaviors (what pyepo effectively does)
    2. Allow each opt to have very different behaviors behind the scenes (such as batch vs iterative solving). But each opt model should provide a common interface to be interacted with.
    3. Keep opt model behavior contained or not contained within the loss function.
    4. Solution: keep opt model solving outside loss function - baseline function and user supplies anything more complicated.
2. For problem 2 there are two possible solutions
    1. universal data loading/input procedure into the training function, but then you can have wrapper function to handle the if-else cases (and separate it out from training loop for modularity)
    2. universal data loading/input procedure into the training function, all loss functions can also accept arbitrary *args, **kwargs
    3. loss function specific dataset objects, therefore we require someone, for a new loss function, to define a specific dataset object for their loss function. (This doesn't necessarily have to be new, for example all loss functions that only require cp, c can just share a single dataset object)
    4. Solution: person who just wants to use -> supply solver, and data. (take care of ad hoc stuff like SPO behind the scene)
    custom loss function - supply data dictionary with necessary inputs for the loss function.

pass in solver, dataset, implement loss and new data -> dataset format.
    
## 10/29 - Change: Moving solver logic for sol, obj inside each unique loss function
Since each loss function requires very specific inputs to calculate the loss and the custom backwards autograd functions, but to some extent all require solving a downstream optimization problem to get the current obj,sol based on the predicted costs, we need to carefully consider where to place the optimization solver step. At a high level, any surrogate loss will require following steps in each epoch of the training loop:
1. Get current batch (requires a dataset object and dataloader)
2. Get Predictions
3. Get necessary components for loss function/backpropagation like current predicted cost solutions by solving optimization problem with given solver
4. Calculate loss function and back propogate

There are two possible approaches for 
1. Keeping the solver solving outside the loss function, getting the required loss function inputs and then passing them in.
    - Pro: Decouples the optimization solving from the loss function. Loss function only needs to calculate loss,backpropogate based on passed in inputs. 
    - Con: However, loss functions may require different things, have different function signatures. Therefore if we keep the solver outside the loss and within the training loop, we may require highly customized training loops, therefore losing the ability to plug and play easily by just passing in solver and dataset (which is what we really want)
2. Keeping the solver logic inside the loss function, therefore the loss function takes key inputs all loss functions may require like predicted cost and true cost and the passed in solver
    - Pro: Because each loss funciton may be highly customized and different from each other, we can now handle each custom logic inside the loss function without requiring specifialized customization in the training loop. 
    - Con: because the optimization solving logic is coupled now with the loss function, we need to specify a uniform signature of solver output in order to code loss function behavior universally.

In the framework of the two key decision points/problems listed in the previous section, this is all about addressing problem 1 with the two following solutions:
- "Allow each opt to have very different behaviors behind the scenes (such as batch vs iterative solving). But each opt model should provide a common interface to be interacted with" and 
- "keeping the solver inside the loss function"





