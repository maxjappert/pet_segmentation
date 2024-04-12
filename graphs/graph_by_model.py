from graph_load_pickle import *
import matplotlib.pyplot as plt
from config import *


# colors = green, lightgreen, blue, skyblue, darkred, lightcoral, saddlebrown, tan   (...)
# markers = s , ^ , o , v , > , <   (...)

epochs = range(1, len(pretrained_loss) + 1)

fig, axs = plt.subplots(1, 2, figsize=(15,5))

# Fine-tunning
axs[0].plot(epochs, finetuning_train_loss, label='Fine-Tuning Train Loss',marker='s', color='blue')
axs[0].plot(epochs, finetuning_val_loss, label='Fine-Tuning Validation Loss',marker='s', color='skyblue')
axs[0].plot(epochs, finetuning_train_acc, label='Fine-Tuning Train Accuracy',marker='^', color='saddlebrown')
axs[0].plot(epochs, finetuning_val_acc, label='Fine-Tuning Validation Accuracy',marker='^', color='tan')

axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].set_title('Fine-Tuning Loss and Accuracy Values Over Epochs',fontweight='bold')
axs[0].legend()
axs[0].grid(True)

# Benchmark
axs[1].plot(epochs, benchmark_train_acc, label='Benchmark Train Accuracy',marker='^', color='darkred')
axs[1].plot(epochs, benchmark_val_acc, label='Benchmark Validation Accuracy',marker='^', color='lightcoral')
axs[1].plot(epochs, benchmark_train_loss, label='Benchmark Train Loss',marker='s', color='green')
axs[1].plot(epochs, benchmark_val_loss, label='Benchmark Validation Loss',marker='s', color='lightgreen')

axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].set_title('Benchmark and Fine-Tuning Accuracy Values Over Epochs',fontweight='bold')
axs[1].legend()
axs[1].grid(True)

# Adjust layout
plt.tight_layout()

# save the plot
plt.savefig('graph_by_model.png')

# Show plots
plt.show()
