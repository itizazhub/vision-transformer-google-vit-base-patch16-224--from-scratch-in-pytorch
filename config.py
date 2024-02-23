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
        self.dataset_path = '/content/EuroSAT'
        self.mnist_path =  './'
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
        #Note: modify followings
        self.epochs = 5
        self.load_weights = True
        self.model_weights_path = './model_weights'
        self.pre_trained_model_path = './pre_trained_model'
        self.onnx_model_path = './onnx_model'
        self.inference_images = './inference_images'
        self.inference_image_name = 'test_image.jpg'
        self.result_folder_path = './results'
               

# Global variable to hold the configuration
config = Config()
