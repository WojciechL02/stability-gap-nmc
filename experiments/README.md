# Experiments

## To run main experiments (from root directory):

Main experiments on CIFAR100 and ImageNet100:
```
./experiments/cifar100ft.sh
./experiments/in100ft.sh
./experiments/cifar100joint.sh
```

Finegrained datasets (Aircrafts, Birds):
```
./experiments/finegrained.sh
```

Experiment with various constant and growing memory sizes:
```
./experiments/cifar100const_mem_t5.sh
./experiments/cifar100const_mem_t10.sh
./experiments/cifar100grow_mem_t5.sh
./experiments/cifar100grow_mem_t10.sh
```

Additional warm-up phase for finetuning:
```
./experiments/warmup.sh
```

## To run experiments from appendix (from root directory):

Other CIL approaches (LwF, EWC, SS-IL):
```
./experiments/cifar100other_appr.sh
./experiments/in100other_appr.sh
```

Different CNN architectures:
```
./experiments/cifar100cnns.sh
./experiments/in100cnns.sh
```
