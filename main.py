from trainer import Trainer
from config import config
import random
import warnings

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    # random.seed(42)
    # torch.manual_seed(42)
    trainer_obj = Trainer(config=config)
    trainer_obj.trainer(config=config)
    trainer_obj.test(config=config)
    trainer_obj.convert_model_to_onnx(config=config)
    trainer_obj.plot_result(config=config)


