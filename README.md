# EC_SAEA
## General
- This report run in problem Knapsack 2 objs, 250 items with framework pymoo
- The target of this respon is compare the quality and quantity of actual evaluate between the SAEA-supported model and the pure EC model.
## File description:
```NSGA_II.py```: Run pure EC model with pop_size(150) and genaration (100)  

```NSGA_II_SAEA```: Run SAEA-supported model with generation(50) and pop_infill(30)

```box_plot.py```: Store hypervolume_result by experimental runs

```make_instances.py``` and ```mokp_data.py```: Write and read data

```mokp_2obj_250items.csv```: Store data in this problem

```hv_boxplot.png```: Show the results
