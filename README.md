# Deep_Learning_for_American_Options_Pricing
## Keywords:
Neural Network; American Options early exercise premium; Juzhong/Whaley approximation;
Quasi-Monte Carlo; Multi-Process; Google Cloud Platform (GCP)


## Summary
This project proposed an innovative way to predict/validate American option prices using Neural Network. \
To reduce computaional cost and expedite the pricing process, we tended to build an architecture with a relatively small number of layers and neurons, therefore making the neural network shallow. 
We trained our feedforward neural network under different combinations of:
- layers
- neurons
- activation functions \
in both local machine (12 cores) and Google cloud Virtual Machines (96 cores).

It turns out that the architecture with 2 layers of 20 neurons and relu & elu activation functions yields the best result after training 500,000 sample data, with an out-sample R-Square of 98.09%.\

In order to find the best architecture, we had to run many different experiments, and for that we built a system to help us keep track. This system also allows us to apply parallel computing into both data generation and network training. This helped us a great deal as we were running thousands of experiments each using up to 500,000 samples. The entire computation time is shortened down to around three hours while it could have easily taken weeks to run without the parallel computing.

## Usage
- Alter the system settings (in my_config.py)
   - If used on local machine: set file system FS = WindowsFS();
   - If used on google cloud: set file system FS = GcsFs().\
And include the right path to the code dir and files.
- Generate data
   - GenerateData_inputs.ipynb: \
   Implement Halton Quasi-Monte Carlo sequence or uniform random sequence to simulate neural network inputs.\
   Can choose whether to include European option prices as an input.
   - GenerateData_outputs.ipynb: \
   Implement Juzhong/Whaley approximation method to generate American Option prices as labels.\
   We built a system, i.e. DataSetManager to manage 24 combinations of data generating options:\
   ![Sample DataSetManager](https://github.com/claraye/Deep_Learning_for_American_Options_Pricing/sample_DataSetManager.png)

- Add experiments\
For different sets of data, we used the ExperimentManager system to create neural networks with different combinations of layers & neurons & activation functions, and choose the best architecture.
- Run experiments\
Run the experiments using multi-processing.
- Analyze\
Evaluate the performance using R-Sqaure: as the MSE is already close enough to 0, we want to have the goodness of fit as high as possible.
