from model import *
from data import *
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

config = tf.ConfigProto(log_device_placement=True)
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# session = tf.Session(config=config)

# config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)




data_gen_args = dict(rotation_range=1.0,
                    width_shift_range=0.3,
                    height_shift_range=0.3,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    vertical_flip = True,
                    fill_mode='nearest')
myGene = trainGenerator(48,'C:/Users/dylee/Pictures/trainingSet','image','gt',data_gen_args,save_to_dir = None)

test_path = 'C:/Users/dylee/Pictures/testSet/image'



model = unet(pretrained_weights = True)
checkpoint_path = 'log2/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
epoch = 1
model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss',verbose=1, save_weights_only=True,save_best_only=True,period=epoch)
# model.load_weights("unet_membrane.hdf5")


# model = model.load_weights('log/my_model.pb')
# model.l('log1/entire_model.h5')



for epoch in range(100):
    model.fit_generator(myGene,steps_per_epoch=int(np.floor(600/48)),epochs=10,callbacks=[model_checkpoint],use_multiprocessing=False,workers=1)

    myTest = testGenerator(test_path, test_files=os.listdir(test_path), num_image=len(os.listdir(test_path)))
    results = model.predict_generator(myTest, len(os.listdir(test_path)), verbose=1)
    saveResult("C:/Users/dylee/Pictures/testSet/val_output/", os.listdir(test_path),epoch, results)
    model.save('log2/entire_model.h5')
    model.save_weights('log2/only_weights.h5')


# model = unet()

# model.load_weights("unet_membrane.hdf5")
