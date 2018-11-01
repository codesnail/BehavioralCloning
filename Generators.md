# Using Generators with Keras and Tensorflow

Keras is great at hiding a lot of complexity in building a deep learning network. Of course, this abstraction comes with its own caveats. Usually it involves giving up some control, and learning a new framework specific way of performing some task. One of these tasks is batch training. The motivation for training in batches is to save a bottleneck resource, which is usually memory in case of big data. This plays at two levels - not only do we want to avoid loading up our model with all the training data at the same time, but if we truly have big data, it may not even be possible to fit it all in memory and then feed batches from it to train the model. The answer comes in the form of the Generator framework. A python generator looks like this:

