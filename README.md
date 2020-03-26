# FewShotLearning
Codes for FewShotLearning

## dataloader.py
A dataloader for few-shot learning datasets
eg：load the data of Few-shot CIFAR-100(FC100)  
``` python
import dataloader
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

## ModelNet40Loader.py
A dataloader for 3d few-shot classification
eg：load the data of Few-shot ModelNet40
``` python
import ModelNet40Loader
import data_utils as d_utils
from torchvision import transforms

mytransforms = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
                d_utils.PointcloudScale(),
                d_utils.PointcloudTranslate(),
                d_utils.PointcloudJitter(),
            ]
        )

traindataloader=ModelNet40Loader.FewShotModelNet40Cls(num_points=1024, 
        transforms=mytransforms, 
        state="train",
        ways=5,
        shots=5,
        query_num=5,
        epoch=1000)
        
for i, (supportInputs, queryInputs,queryLabels) in enumerate(traindataloader):
    #updata model
```



