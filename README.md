# Multi-Head Attention from a Scratch

This repo implements the multi-head attention block from scratch. 

This is compared to the PyTorch implementation of the multi-head attention block and the results are designed to be the same while the implementation here being more clear.

There are two classes, one for single head and one for multi-head. Both of these implementations assume dropout of 0 and no masking. The `test_attentions.py` has the pytest for comparing the output from PyCharm and from the implemented code.
