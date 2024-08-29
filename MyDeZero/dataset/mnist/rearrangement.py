import numpy as np

train_image_set = np.load('./train_image_set.npy')
train_index_set = np.load('./train_index_set.npy')
test_image_set = np.load('./test_image_set.npy')
test_index_set = np.load('./test_index_set.npy')


train_image_set = train_image_set.reshape(60000, -1, 28, 28)
test_image_set = test_image_set.reshape(10000, -1, 28, 28)

print(train_image_set.shape)
print(train_index_set.shape)
print(test_image_set.shape)
print(test_index_set.shape)

np.savez('train_set.npz', image_set=train_image_set, label_set=train_index_set)
np.savez('test_set.npz', image_set=test_image_set, label_set=test_index_set)
