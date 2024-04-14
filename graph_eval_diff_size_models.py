import matplotlib.pyplot as plt

# Dictionary with evaluation metrics
evaluation_data = {
    'mrp_first_experiment_models/finetuned_0.1.pth': {'accuracy': 0.7153157285917484, 'iou': 0.7226901424864947, 'dice_coefficient': 0.7626636489609203, 'precision': 0.8252133361200604, 'recall': 0.8843903669782163},
    'mrp_first_experiment_models/finetuned_0.2.pth': {'accuracy': 0.7451285494546794, 'iou': 0.7715473440346053, 'dice_coefficient': 0.7956854949481891, 'precision': 0.8317619620262353, 'recall': 0.9318055489981778},
    'mrp_first_experiment_models/finetuned_0.3.pth': {'accuracy': 0.7540021653589939, 'iou': 0.7443045320864023, 'dice_coefficient': 0.7731004279997369, 'precision': 0.8792496002045258, 'recall': 0.876775267680831},
    'mrp_first_experiment_models/finetuned_0.4.pth': {'accuracy': 0.7687678198035909, 'iou': 0.8073610946817372, 'dice_coefficient': 0.8171788332055132, 'precision': 0.8451592457848457, 'recall': 0.9587119889678413},
    'mrp_first_experiment_models/finetuned_0.5.pth': {'accuracy': 0.7867896973388206, 'iou': 0.784717725197037, 'dice_coefficient': 0.7988570510503856, 'precision': 0.8990847491314821, 'recall': 0.9154501592371244},
    'mrp_first_experiment_models/finetuned_0.6.pth': {'accuracy': 0.7893193042970347, 'iou': 0.8011223379115129, 'dice_coefficient': 0.8091748506349542, 'precision': 0.8938098158706614, 'recall': 0.9306213786553552},
    'mrp_first_experiment_models/finetuned_0.7.pth': {'accuracy': 0.7964369533951583, 'iou': 0.8172013035543081, 'dice_coefficient': 0.8196218858005114, 'precision': 0.8954611218906791, 'recall': 0.9431347651471812},
    'mrp_first_experiment_models/finetuned_0.8.pth': {'accuracy': 0.7995640735724994, 'iou': 0.7821146130386479, 'dice_coefficient': 0.7935138950195777, 'precision': 0.9381300742973752, 'recall': 0.898511965234476},
    'mrp_first_experiment_models/finetuned_0.9.pth': {'accuracy': 0.7992384321327657, 'iou': 0.8083290351282962, 'dice_coefficient': 0.8136596244165389, 'precision': 0.9074957528437431, 'recall': 0.934358741530553},
    'mrp_first_experiment_models/finetuned_1.pth': {'accuracy': 0.8115153818242262, 'iou': 0.8379544814365142, 'dice_coefficient': 0.830817381315734, 'precision': 0.9117105881742414, 'recall': 0.9535173488538348}
}

# Extracting model names and metrics
model_names = list(evaluation_data.keys())
metrics = list(evaluation_data[model_names[0]].keys())

# Extracting fine-tuning factors
fine_tuning_factors = [float(model_name.split('_')[-1][:-4]) * 100 for model_name in model_names]

# Plotting
plt.figure(figsize=(10, 6))
for metric in metrics:
    plt.plot(fine_tuning_factors, [data[metric] for data in evaluation_data.values()], label=metric, marker='s')

plt.title('The comparison of the pretrained segmentation model with different fine-tunning dataset sizes', fontweight='bold')
plt.xlabel('Fine-tuning Dataset Size')
plt.ylabel('')
plt.xticks(fine_tuning_factors, [f'{factor:.0f}%' for factor in fine_tuning_factors])
plt.legend()
plt.grid(True)
plt.tight_layout()

# save the plot 
plt.savefig('graph_eval_first_exp.png')

plt.show()
