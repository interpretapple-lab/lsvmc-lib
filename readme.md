## Introduction

*LSVMC-Lib* is an open-source library that incorporates a veracity component into support vector machine (SVM) predictions, aiming to enhance their transparency and reliability.

## Requirements

*LSVMC-Lib* has been implemented as an extension of [XSVMC-Lib](https://github.com/interpretapple-lab/xsvmc-lib). Thus it requires Python 3.12+ and the [SciKit-Learn](https://scikit-learn.org) package (python3 -m pip install sklearn).

Although not explicitly required by *LSVMC-Lib*, the following packages are essential for executing the provided examples:

- [Numpy](https://numpy.org) (```python3 -m pip install numpy```)
- [Matploplib](https://matplotlib.org) (```python3 -m pip install matploplib```)
- [NLTK](https://nltk.org) (```python3 -m pip install nltk```)
- [ETE Toolkit](https://etetoolkit.org/) (```python3 -m pip install ete3```)
- [Keras](https://keras.io/) (```python3 -m pip install keras```)
- [Tensorflow](https://www.tensorflow.org/) (```python3 -m pip install tensorflow```)
- [Jupyter](https://jupyter.org/) (```python3 -m pip install jupyter```)

## Examples

The usability of *LSVMC-Lib* is demonstrated through the following examples:

- [Leaves Example](examples/leaves_ALGs.ipynb)
- [Brain Tumors Example](examples/brain_ALGs.ipynb)
- [Neutrophils Example](examples/neutrophils_ALGs.ipynb)
- [Newswire Stories Example](examples/reuters_ALGs.ipynb)

For the purpose of reproducibility, a [Docker container](https://www.docker.com/) that includes all the necessary components to execute the examples is provided. 

The next command can be used for building an image based on the provided [Dockerfile](Dockerfile):

````
docker build -t lsvmc-demo:0.3 .
````

The next command can be used to execute the previous constructed image:
````
docker run -p 8888:8888 -v /YOUR/DIR/datasets/:/app/datasets -v /YOUR/DIR/nltk_data:/app/nltk_data --rm  lsvmc-demo:0.3

````

The datasets used in these examples must be downloaded from their respective sources.  


## License
*LSVMC-Lib* is released under the [Apache License, Version 2.0](LICENSE.txt).
