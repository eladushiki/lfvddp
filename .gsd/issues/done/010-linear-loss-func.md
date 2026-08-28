# Issue 010: Linear Loss Func

**Status:** Done

## Statement

After a discussion with Shikma we agreed to restrict even more the expression of the SR functions. That is, existing expression of $e^f$ in the loss function should be substituted by $(1+f)$, and $e^g$ with $(1-f)$. This means that $f$ goes to $\log(1+f)$ and $g$ to $\log(1-f)$ as well (along with any additional variation thereof).

This allows us to:
a. Remove unnecessary clutter from the loss function
b. Have both functions reciprocate around the average dist in a way that is multiplicative
c. On the way, save some arithmetics
d. Restrict the parameter space in which we search for optimum in a way that would improve training convergence.

## Acceptance criteria

- `DifferentiatingNetwork` should export the exact same interface.
- If implemented correctly, convergence states should be almost identical as well.
- Plots of sorts should display the updated expressions as well.

## Implementation

 Replaced the SR exponential parameterization with bounded linear weights and updated predictions, labels, and loss-reference tests. The reciprocal linear SR integral is reduced algebraically to scalar sums and one dot product, while contiguous A/B event groups are sliced directly instead of scanned through boolean masks. Neural nuisance CR terms likewise retain their category-contiguous representation, avoiding full-array zero-weight arithmetic each epoch.
