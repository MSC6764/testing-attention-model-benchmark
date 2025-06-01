# testing-attention-model-benchmark

This code uses the work of Wouter Kool and Herke van Hoof and Max Welling in the paper Attention, Learn to Solve Routing Problems! available at https://github.com/wouterkool/attention-learn-to-route.

For general use the model use the provided link. The original code from this repository was used.

Steps for evaluting the pretrained models:

For generation of the data for tsp and vrp the "generate_all_test_data.sh" can be used.
Therefore, it is necessary to validate the pre-trained models as described in their repository.
This can be done via the "evaluate_pretrained_models.py"

Steps to reproduce the results of the paper:

For integration of the cmt instances the instances in the folder data/cmt can be used.

1. Adaption of the functions in the original code as described in "changes_for_cmt_within_attention_model.txt".

2. Solving CMT instances with "evaluate_pretrained_models_cmt_ready.py".

3. Plotting routes with plot_cmt_tours_attention.py

4. Visualization of gaps with the "result_generalization.ipynb".


Additional the "or_tools_solver.py" can be used to reproduce the results for OR-Tools.




