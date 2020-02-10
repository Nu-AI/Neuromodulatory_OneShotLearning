#!/bin/bash

python3 omniglot.py --nbclasses 5  --nbiter 5000000 --rule oja --activ relu --steplr 1000000 --prestime 1 --prestimetest 1 --gamma .666 --alpha free --lr 3e-5
