
## Neuromodulation for dynamic few-shot learning



This repo contains the source code related to the modulatory trace learning rules introduced 
in the paper titled - ***[Exploring Neuromodulation for Dynamic Learning](https://www.frontiersin.org/articles/10.3389/fnins.2020.00928/full)***
The dynamic learning rules enable the network to modulate the output layer weights based on the context derived from the internal state of the system.
This repo also consists of the log files consisting of the trained model weights being used to ensure inference of the parameters.

The results can be obtained by following this command

`python3 OSL_network/main_run.py`

The key parameters to be updated are as followed in the `parameters.py` file:
```python
params = {
    'no_classes': 5,                                   # Number of classes in the N-way K-shot learning case
    'rule': 'mod'                                      # Learaning rule can be hebb or oja or mod
    'no_shots': 1,                                     # Number of 'shots' in the few-shots learning
    'rand_seed':0,                                     # Select the random seed file for taking the weights
    'no_filters' : 64,                                 # Number of filters in the convolutional layers
    'imagesize': 31,                                   # The size of the 2D images to be reshaped to
    'learningrate': 1e-5,                              # The initial learning rate for the network
    'present_test': 1,                                 # Number of times we present the testing class
    'no_test_iters': 100,
    'activation': 'tanh',                              # tanh or relu or selu
    'cuda': 1,                                         # GPU or CPU
    # For hardware emulation in fixed point 
    'precision': 16,                                   # Precision to store the weights           
    'fractional':10,                                   # The number of fractional bits in the weight representation
    'address' : '../omniglot_dataset/omniglot/python/', # enter the path of the dataset here
}
```

The different configurations to average over multiple runs can be performaed by running 
`./run_allscripts.sh`


For the hardware emulated version, the key parameters need to be set are 
+ `precision` - Enable the precision between 8 - 32 bits
+ `fractional` - Encode the fractional bit width from 6-30 bits 
The dependencies can be installed by 

`pip install -r requirements.txt`
For running the fixed point code, make sure to generate a new `FixedPoint.so` file in `hardware_emulated` directory by running
```
g++ -o FixedPoint.so -shared -fPIC FixedPoint.cpp
export LD_LIBRARY_PATH=.
```

A hardware emulated version of the code to evaluate the model performance on variable fixed point precision can be tested by running the following command

```python
python3 hardware_emulated/omniglot_test_hardware.py
```

### Acknowledgements

This code has been developed as part of a research project supported by the Lifelong Learning Machines (L2M) program of the Defence Advanced Research Projects Agency (DARPA)

---

