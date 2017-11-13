TO DO: 

Short term items:
[ ] Implement analytic solution for deflection of axially symmetric (q=1) Sersic profile. The axially symmetric solution is used to calculate the potential even for a non-symmetric Sersic lens, hence makes calculating time delays much faster.
[ ] Implement command to write best-fit lens/source model to script so that it can be immediately reloaded

Long term items:
[ ] Implement analytic source profile mode (as opposed to pixellated source mode)
[ ] Develop Python bindings!
[ ] Include full pixel covariance matrix as an option
[ ] Instead of re-allocating grid with every likelihood evaluation, just allocate all cells up to some max # of levels, then "activate" cells as needed...this will be more parallelizable if only analytic lens models are being used (because in that case, memory access becomes the bottleneck). Low priority, however

