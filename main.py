from model import *
from data import *
import matplotlib.pyplot as plt

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
result = model.fit(myGene,steps_per_epoch=1,epochs=3000,callbacks=[model_checkpoint])

#print(result.history.keys())
#print(result.history['loss'])

plt.plot(result.history['loss'])
plt.plot(result.history['accuracy'])
plt.title('Model loss/accuracy')
plt.ylabel('Loss/Accuracy')
plt.xlabel('Epoch')
plt.legend(['Loss', 'Accuracy'], loc='upper left')
plt.show()

plt.plot(result.history['precision_m'])
plt.plot(result.history['recall_m'])
plt.title('Model precision_m/recall_m')
plt.ylabel('Precision/Recall')
plt.xlabel('Epoch')
plt.legend(['Precision', 'Recall'], loc='upper left')
plt.show()

plt.plot(result.history['f1_m'])
plt.title('Model f1_m')
plt.ylabel('f1_m')
plt.xlabel('Epoch')
plt.legend(['F1_m'], loc='upper left')
plt.show()

testGene = testGenerator("data/membrane/test")
results = model.predict(testGene,5921,verbose=1)
saveResult("data/membrane/test",results)

