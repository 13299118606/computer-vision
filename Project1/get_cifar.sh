#!/bin/bash

echo "Downloading cifar"
curl https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O
tar xf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz
