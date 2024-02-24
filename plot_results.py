# %matplotlib inline
from config import config
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

if not os.path.exists(Path(config.result_folder_path)):
    print("No such dir found: ", Path(config.result_folder_path))
else:
  df = pd.read_csv(Path(config.result_folder_path).joinpath("results.csv"))
  epochs = range(1, config.epochs + 1)

  plt.figure(figsize=(12, 6))

  # Plot training loss
  plt.subplot(1, 2, 1)
  plt.plot(epochs, df['training_loss'], 'b', label='Training loss')
  plt.plot(epochs, df['validation_loss'], 'r', label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  # Plot training accuracy
  plt.subplot(1, 2, 2)
  plt.plot(epochs, df['training_accuracy'], 'b', label='Training accuracy')
  plt.plot(epochs, df['validation_accuracy'], 'r', label='Validation accuracy')
  plt.title('Training and validation accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()

  plt.savefig(Path(config.result_folder_path).joinpath('plots.png'))
  
  plt.tight_layout()
  plt.show()