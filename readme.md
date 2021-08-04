---
## Neuromodulation for dynamic few-shot learning



This repo contains the source code related to the modulatory trace learning rules introduced 
in the paper titled - ***[Exploring Neuromodulation for Dynamic Learning](https://www.frontiersin.org/articles/10.3389/fnins.2020.00928/full)***
The dynamic learning rules enable the network to modulate the output layer weights based on the context derived from the internal state of the system.
This repo also consists of the log files consisting of the trained model weights being used to ensure inference of the parameters.

The results can be obtained by following this command

`python3 train.py --nbclasses 5  --nbiter 5000000 --rule mod --activ tanh --steplr 1000000 --nbshots 5 --gamma .666  --lr 3e-5 --rngseed 3`

The different configurations to average over multiple runs can be performaed by running 
`./run_allscripts.sh`

A hardware emulated version of the code to evaluate the model performance on variable fixed point precision can be tested by running the following commnad

```python
python3 hardware_emulated/omniglot_test_hardware.py
```

The dependencies can be installed by 

`pip install -r requirements.txt`

### Acknowledgements

This code has been developed as part of a research project supported by the Lifelong Learning Machines (L2M) program of the Defence Advanced Research Projects Agency (DARPA)

---

