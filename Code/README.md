# Classification of Food Images Using Convolution Neural Networks

## Requirements
- Python (3.5)
- PyTorch
- Nividia GPU 

## How to run the network using the preprocessed data
1. Upgrade pip command:
```
sudo pip install --upgrade pip
```
2. Install h5py:
```
pip3 install h5py
```
3. Run the following commands to download training and testing datasets:
```
wget https://s3.amazonaws.com/food-classification-datasets/food_train.h5
wget https://s3.amazonaws.com/food-classification-datasets/food_test.h5
```
4. Run the following command to train the CNN model and make predictions on test images:
```
python3 food_CNN.py
```

## Other code
1. create_dataset.py
- script used to create preprocessed training and testing database (food_train.h5 & food_test.h5) files from jpeg images found on Kaggle website (www.kaggle.com/kmader/food41)
