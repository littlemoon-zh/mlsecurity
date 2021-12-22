# Lab 3

## Models

| x     | model path             |
| ----- | ---------------------- |
| x=2%  | lab3/models/prune45.h5 |
| x=4%  | lab3/models/prune48.h5 |
| x=10% | lab3/models/prune52.h5 |
| x=30% | lab3/models/prune54.h5 |

## Usage

```shell
cd lab3
python myeval.py img.jpg
```

# project section

usage:
```shell script 
cd lab3
python p_eval.py model_filename data_filename img_filename
# output: a number
```

example: 
```shell script
cd lab3
python p_eval.py models/bd_net.h5 data/cl/valid.h5 img.png
# output: 1283
```