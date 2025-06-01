Testing Attention Model Benchmark
This project builds upon the work by Wouter Kool, Herke van Hoof, and Max Welling, as presented in their paper Attention, Learn to Solve Routing Problems!.

The original code from their repository was adapted and extended for benchmarking and evaluating pretrained attention-based models on various routing problems.

How to Evaluate Pretrained Models
Generate Test Data
Use the script generate_all_test_data.sh to generate data for TSP and VRP instances.

Evaluate Pretrained Models
Run evaluate_pretrained_models.py to validate the pretrained models as described in the original repository.

Reproducing Paper Results
To reproduce the results from the original paper and run evaluations on CMT benchmark instances:

CMT Instance Integration
Use the problem instances located in data/cmt.

Adapt Functions
Apply the necessary modifications as described in the file changes_for_cmt_within_attention_model.txt.

Solve CMT Instances
Run evaluate_pretrained_models_cmt_ready.py to solve CMT problems using the adapted model.

Plot Tours
Use plot_cmt_tours_attention.py to visualize the generated tours.

Visualize Performance Gaps
Open and run result_generalization.ipynb to visualize generalization gaps and model performance.

OR-Tools Baseline
To reproduce results using OR-Tools as a baseline solver, run or_tools_solver.py.

Feel free to contribute or raise issues if you encounter any problems!

