---
This repo contains everything related to my modulatory trace learning rule introduced 
in the thesis titled - ***[Exploring Neuromodulation for Dynamic Learning](https://scholarworks.rit.edu/theses/10156/)***

This will also consist of the log files consisting of the trained model weights being used.
The results can be obtained by following this command

`python3 omniglot.py`

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

