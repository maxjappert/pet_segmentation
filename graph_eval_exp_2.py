import matplotlib.pyplot as plt

# Dictionary with evaluation metrics
pretrained_model_data = {
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

benchmark_model_data = {
    'benchmark_0.1.pth': {'accuracy': 0.7012, 'iou': 0.7019, 'dice_coefficient': 0.7489, 'precision': 0.8306, 'recall': 0.8642},
    'benchmark_0.2.pth': {'accuracy': 0.7125, 'iou': 0.7294, 'dice_coefficient': 0.7697, 'precision': 0.8195, 'recall': 0.8990},
    'benchmark_0.3.pth': {'accuracy': 0.7312, 'iou': 0.7221, 'dice_coefficient': 0.7591, 'precision': 0.8625, 'recall': 0.8640},
    'benchmark_0.4.pth': {'accuracy': 0.7521, 'iou': 0.7438, 'dice_coefficient': 0.7726, 'precision': 0.8799, 'recall': 0.8817},
    'benchmark_0.5.pth': {'accuracy': 0.7547, 'iou': 0.7465, 'dice_coefficient': 0.7709, 'precision': 0.8825, 'recall': 0.8791},
    'benchmark_0.6.pth': {'accuracy': 0.7734, 'iou': 0.7787, 'dice_coefficient': 0.7894, 'precision': 0.8915, 'recall': 0.9091},
    'benchmark_0.7.pth': {'accuracy': 0.7685, 'iou': 0.7848, 'dice_coefficient': 0.7993, 'precision': 0.8731, 'recall': 0.9221},
    'benchmark_0.8.pth': {'accuracy': 0.7741, 'iou': 0.7620, 'dice_coefficient': 0.7799, 'precision': 0.9077, 'recall': 0.8841},
    'benchmark_0.9.pth': {'accuracy': 0.7790, 'iou': 0.7815, 'dice_coefficient': 0.7943, 'precision': 0.8969, 'recall': 0.9096},
    'benchmark_1.pth': {'accuracy': 0.7927, 'iou': 0.8217, 'dice_coefficient': 0.8098, 'precision': 0.8957, 'recall': 0.9506}
}


# Extracting model names and metrics
model_names_pretrained = list(pretrained_model_data.keys())
model_names_benchmark = list(benchmark_model_data.keys())


# Extracting accuracy and IOU
accuracies_pretrained = [data["accuracy"] for data in pretrained_model_data.values()]
ious_pretrained = [data["iou"] for data in pretrained_model_data.values()]

accuracies_benchmark = [data["accuracy"] for data in benchmark_model_data.values()]
ious_benchmark = [data["iou"] for data in benchmark_model_data.values()]

# Extracting fine-tuning factors
fine_tuning_factors = [float(model_name.split('_')[-1][:-4]) * 100 for model_name in model_names_pretrained]

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(fine_tuning_factors, accuracies_benchmark, label="Benchmark model Accuracy", marker='^',color='darkred')
plt.plot(fine_tuning_factors, ious_benchmark, label="Benchmark model IOU", marker='s',color='lightcoral')

plt.plot(fine_tuning_factors, accuracies_pretrained, label="Pretrained model Accuracy", marker='^',color='green')
plt.plot(fine_tuning_factors, ious_pretrained, label="Pretrained model IOU", marker='s',color='lightgreen')

plt.title('Pretrained segmentation model and benchmark model with different fine-tunning dataset sizes', fontweight='bold')
plt.xlabel('Fine-tuning Dataset Size')
plt.ylabel('')
plt.xticks(fine_tuning_factors, [f'{factor:.0f}%' for factor in fine_tuning_factors])
plt.legend()
plt.grid(True)
plt.tight_layout()

# save the plot 
plt.savefig('graph_eval_exp_2.png')

plt.show()
