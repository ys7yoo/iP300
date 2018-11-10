
## How to run
1. Prepare ERP data with channel masks.
  * EEG data format: (num trirals)x(num channels*samples)
  * mask data format: (num channels)x1
2. Put preprocessed data in `data` folder
  * make a symbolic link to your data folder
  ```
  ln -s ../P300_ensembleSVM/data ./data
  ```
3. Train & test using various classifiers (mainly from sci-kit learn, few from Tensorflow)

