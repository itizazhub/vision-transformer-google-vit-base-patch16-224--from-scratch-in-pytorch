'''
dataset_folder
    |->class1_folder
    |->class2_folder
    ...
'''

class Config:
    def __init__(self):
        # Dataset options
        self.custom_dataset = True
        self.dataset_path = '../vit-datasets/custom_dataset/EuroSAT' # './custom_dataset'
        self.mnist_path =  './' # '../vit-datasets/MNIST_dataset'
        self.classes = 10
        self.mnist_classes = 10
        self.image_size = 224
        self.batch_size = 32

        # Model options
        self.encoder_blocks = 12
        self.channels = 3
        self.mnist_channels = 1
        self.patch_size = 16
        self.inner_dim = 3072
        self.H_mnist = 8 
        self.H = 12
        self.dropout = 0.1
        self.learning_rate = 0.00001
        self.momentum = 0.9
        #Note: only modify followings
        self.epochs = 10
        self.load_weights = True
        self.model_weights_path = '../vit-model-weights/model_weights' # './model_weights'
        self.pre_trained_model_path = './pre_trained_model'
        self.onnx_model_path = './onnx_model'
        self.inference_images = './inference_images'
        self.inference_image_name = 'image.jpg'
               

# Global variable to hold the configuration
config = Config()
