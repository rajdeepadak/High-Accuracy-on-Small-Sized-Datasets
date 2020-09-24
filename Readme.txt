Libraries:
numpy = 1.19.1
opencv = 4.4.0
Tensorflow version = 2.1
keras = 2.3.1
cuDNN = 7.6.5
CUDA Toolkit = 10.1

System:
Windows 10 Pro
i7 7500U 2.7GHz
8GB RAM
Nvidia 940 MX 2GB Graphic Card
256 GB SSD, 1TB HDD 

Steps to run code:
1. Download 'Lighting_Classifier' folder.
2. Keep this folder in C: drive (SSD).
3. Open Lighting_Classifier. Extract the folder 'Image_Set' by clicking extract here.
4. 5 Folders namely dataset, Dummy_Image_Test, Test_AL, Test_NL, Random will be present in C:\Lighting_Classifier 
   after extraction.
5. Open python IDE. Select C:\Lighting_Classifier as working directory.
6. Run python code 'cnn_training_3.py'.


Following may improve performance:
1. Store dataset, codes in the same file location and in SSD (preferably).
2. Use keras 2.3.1 with tensorflow-gpu 2.1 backend. 
3. Use a separate environment for running code.
4. Open NVIDIA Control Panel >> Manage 3D settings >> preferred graphics processor: High-performance NVIDIA processor.
5. Go to Program Settings in Manage 3D settings. 
   Select a program to customize:
   Select the python .exe (executable) within the environment used by code compiler.
   Click Apply.
6. During training all other applications should be closed as RAM utility, CPU and GPU usage is increased. 


Outputs received: 90.3% Train Accuracy, 87.00% Test Accuracy (Best Record, See Training Results word document)
                  Time Consumed: 13 mins.
		  Time Consumed per epoch: 88(train) + 40(validation) seconds (avg)
		  GPU Load: 29% (avg), 99% (highest).
