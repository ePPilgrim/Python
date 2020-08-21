
from headers import *

def CreateAndSaveModel(root_dir, model, test_set,name = "Model", extent = 'hdf5'):
    file_names = [file_name for file_name in os.listdir(root_dir)]
    for file_name in file_names:
        if file_name.endswith(extent):
            path = os.path.join(root_dir,file_name)
            print(path)
            model.load_weights(path)
            values = model.evaluate(test_set,verbose=0)
            print(values)
            model_file_name = "{}_L{:.3f}_A{:.3f}_model.h5".format(name, values[0], values[1])
            path = os.path.join(root_dir,model_file_name) 
            model.save(path)
            print(path)

def GetTrainTestSets(label_set, train_cnts, test_cnts):
    trgen = imgen.Generator(trainDfPath, trainRootDir, (224,224))
    trgen.SetDataFrameForGeneration(label_set,train_cnts)
    train_set = trgen.GetGenerator(bs = 64)

    tsgen = imgen.Generator(testDfPath,testRootDir, (224,224))
    tsgen.SetDataFrameForGeneration(label_set, test_cnts)
    test_set = tsgen.GetGenerator(bs = 128)
    return (train_set, test_set)

class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait = True)
