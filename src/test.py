from datasets.data_loader import get_loaders

datasets = ["imagenet_subset_kaggle"]
num_tasks = 10
nc_first_task = 10
nc_per_task = None

trn_loader, val_loader, tst_loader, taskcla = get_loaders(datasets, num_tasks, nc_first_task,
                                                              nc_per_task,
                                                              1, 3,
                                                              pin_memory=False,
                                                              max_classes_per_dataset=None,
                                                              max_examples_per_class_trn=None,
                                                              max_examples_per_class_val=None,
                                                              max_examples_per_class_tst=None,
                                                              extra_aug='',
                                                              validation=0.0)

x1 = next(iter(trn_loader[0]))
print(x1[0].shape)
print("=====")
# x1 = next(iter(val_loader[0]))
# print(x1[0].shape)
print("=====")
x1 = next(iter(tst_loader[0]))
print(x1[0].shape)
print("=====")
print(taskcla)