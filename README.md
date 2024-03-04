# RMR-Fusion_AAAI24
The code of A Robust Mutual-reinforcing Framework for 3D Multi-modal Medical Image Fusion based on Visual-semantic Consistency.
#### Recommended Environment:<br>
 - [ ] python = 3.8
 - [ ] torch = 1.12.1
 - [ ] monai = 1.1.0
 - [ ] numpy = 1.24.2
 - [ ] SimmpleITK = 2.2.1
 - [ ] pillow = 9.4.0
 - [ ] scikit-image = 0.19.3
 - [ ] imgaug = 0.4.0
 - [ ] trochio = 0.18

# Training:<br>
## Stage#1: Degradation-robust Autoencoder<br>
* Prepare training data:<br>
  * Put T1 modal model volume patches in '/dataset64/training/t1' and '/dataset64/validation/t1'<br>
  * Put T2-Flair model volume patches in '/dataset64/training/t2-flair' and '/dataset64/validation/t2-flair' for train and valid<br>
* Run ```CUDA_VISIBLE_DEVICES=0 python tarin_dualSwinAE.py```<br>
##  Stage#2: Mutual-reinforcing Fusionr<br>
* Prepare training data:<br>
  * Put T1 modal model volume patches in '/dataset/training/t1' and '/dataset/validation/t1'<br>
  * Put T2-Flair model volume patches in '/dataset/training/t2-flair' and '/dataset/validation/t2-flair'<br>
  * Put Label volume patches into '/dataset/training/label' and '/dataset/validation/label' for train and valid<br>
* Adjust ```DualSwinAE_model_ptah``` in ```train.py``` to the path where you store the model in Stage #1.<br>
* Run ```CUDA_VISIBLE_DEVICES=0 python tarin.py```<br>
# Test:<br>
* Prepare test data: put the processed data in './test_img'<br>
* Run ```CUDA_VISIBLE_DEVICES=0 python test.py```<br>
▢ This task is based on Stage #1, so the code and models in Stage #1 should be downloaded and prepared in advance.<br>
