<!-- # transfer_learning

## TODO

## Prepare dataset in the format
`X = [[t1, i1], [t2, i2], ...]` where t1 is the text feature corresponding to image i1 and
`y = [y1, y2, ...]` is the label. -->

# Transfer Learning
This package demonstrates how to build a transfer learning network effortlessly with [Mozi](https://github.com/hycis/Mozi).

<!-- ![transfer learning](images/illustration.png "Title" {width=40px height=400px}) -->
<img src="images/illustration.png" width="300">


Standard `transfer learning` also known as `multi-task learning` or `multi-modal learning` consist of transforming multiple different feature spaces, for example, a text feature space and an image feature space, into a shared representation. For example in the figure above, we try to map both the image and text to a common shared feature space. One interesting feature of transfer learning is that once the model is trained, during testing, we can just use one of the input channels for classification, for example we can just use text or image as input for classification. Of course we can also switch on all the channels to improve the classification result. The code below illustrate how to build a transfer learning model.

First build a left layer
