<!-- # transfer_learning

## TODO

## Prepare dataset in the format
`X = [[t1, i1], [t2, i2], ...]` where t1 is the text feature corresponding to image i1 and
`y = [y1, y2, ...]` is the label. -->

# Transfer Learning
This package demonstrates how to build a transfer learning network effortlessly with [Mozi](https://github.com/hycis/Mozi).

![transfer learning](images/illustration.png)

Standard `transfer learning` also known as `multi-task learning` or `multi-modal learning` consist of multiple different feature spaces, for example, a text feature space and an image feature space. And there are usually two kinds of network, one has a `split` representation (Figure a), another has a `shared` representation (Figure b). They have the following main differences

| Shared | Split |
|--------|-------|å
| abc |  efg  |
