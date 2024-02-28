This project was used to implement the model building of a multi-head self-attention mechanism, which entails the complete process starting from the association matrix, to the generation of association tree data, which is then used for training.

dataFunction.py : A tree structure is generated based on the association matrix and then features for training are generated based on the tree structure.
my_model.py : Defines a model for a multi-head self-attention mechanism.
train.py : For data training, in addition, the experimental results presented in the paper need to be parameterized in this file.

The above code runs in the order dataFunction.py,my_model.py,train.py.

In addition, the author has commented important statements in the code file in English.
