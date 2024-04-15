from graph_load_pickle import *
import matplotlib.pyplot as plt

# colors = green, lightgreen, blue, skyblue, darkred, lightcoral, saddlebrown, tan   (...)
# markers = s , ^ , o , v , > , <   (...)

epochs = range(1, len(pretrained_loss) + 1)

fig, axs = plt.subplots(2,2, figsize=(15,8))

# Plot benchmark loss
axs[0,0].plot(epochs, benchmark_train_loss, label='Train',marker='s', color='green')
axs[0,0].plot(epochs, benchmark_val_loss, label='Validation',marker='s', color='lightgreen')
axs[0,0].set_xlabel('Epochs')
axs[0,0].set_ylabel('Normalised Loss')
axs[0,0].set_title('Benchmark Loss over Training Epochs',fontweight='bold')
axs[0,0].legend()
axs[0,0].grid(True)

# Plot benchmark acc
axs[0,1].plot(epochs, benchmark_train_acc, label='Train',marker='^', color='darkred')
axs[0,1].plot(epochs, benchmark_val_acc, label='Validation',marker='^', color='lightcoral')
axs[0,1].set_xlabel('Epochs')
axs[0,1].set_ylabel('Accuracy')
axs[0,1].set_title('Benchmark Accuracy Over Epochs',fontweight='bold')
axs[0,1].legend()
axs[0,1].grid(True)

# Plot fine-tuning loss
axs[1,0].plot(epochs, finetuning_train_loss, label='Train',marker='s', color='blue')
axs[1,0].plot(epochs, finetuning_val_loss, label='Validation',marker='s', color='skyblue')
axs[1,0].set_xlabel('Epochs')
axs[1,0].set_ylabel('Normalised Loss')
axs[1,0].set_title('Fine-Tuning Loss Over Epochs',fontweight='bold')
axs[1,0].legend()
axs[1,0].grid(True)

# Plot fine-tuning acc
axs[1,1].plot(epochs, finetuning_train_acc, label='Train',marker='^', color='saddlebrown')
axs[1,1].plot(epochs, finetuning_val_acc, label='Validation',marker='^', color='tan')
axs[1,1].set_xlabel('Epochs')
axs[1,1].set_ylabel('Accuracy')
axs[1,1].set_title('Fine-Tuning Accuracy Over Epochs',fontweight='bold')
axs[1,1].legend()
axs[1,1].grid(True)

# Adjust layout
plt.tight_layout()

# save the plot 
plt.savefig('graph_all_separate.png')

# Show plots
plt.show()

plt.figure(figsize=(7.5, 4))
plt.plot(epochs, pretrained_loss, marker='^', color='saddlebrown', label='Train')
plt.plot(epochs, pretrained_val_loss, marker='^', color='tan', label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Normalised NT-Xent Loss')
plt.title('Pre-Training Loss Over Epochs', fontweight='bold')
plt.grid(True)
plt.legend()
plt.savefig('pretraining_loss.png')
plt.show()
