# FewShotLearning
Codes for FewShotLearning

# dataloader.py
A dataloader for few-shot learning datasets
eg：load the data of Few-shot CIFAR-100(FC100)  
```
import dataloader
image_datasets = {}
setroot='/tmp/fs_dinoor/' #set the root directory of the dataset
traindataloader=dataloader.fc100(
        root=os.path.join(setroot, 'train'),
        state="train", 
        ways=5, 
        shots=5,
        query_num=5,
        epoch=2,shapesize=32)
        
for i, (supportInputs, queryInputs,queryLabels) in enumerate(traindataloader):
    #updata model
```

