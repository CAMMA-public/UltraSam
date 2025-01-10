import os
import torch
import matplotlib.pyplot as plt

def dump_masks(preds, gts, save_dir):
    """
    Dump predicted and ground truth masks side by side as images.

    Args:
        preds (torch.Tensor or np.ndarray): Predicted masks of shape (N, H, W).
        gts (torch.Tensor or np.ndarray): Ground truth masks of shape (N, H, W).
        save_dir (str): Directory where the masks should be saved.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    N = preds.shape[0]  # Number of masks

    # Iterate over each mask
    for i in range(N):
        pred_mask = preds[i].detach().cpu().numpy() if isinstance(preds, torch.Tensor) else preds[i]
        gt_mask = gts[i].detach().cpu().numpy() if isinstance(gts, torch.Tensor) else gts[i]

        # Create a plot with side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        # Plot predicted mask
        axes[0].imshow(pred_mask, cmap='gray')
        axes[0].set_title('Predicted Mask')
        axes[0].axis('off')  # Hide the axes

        # Plot ground truth mask
        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')  # Hide the axes

        # Save the image
        save_path = os.path.join(save_dir, f'mask_{i}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved: {save_path}")


def dump_fmap(preds, gts, save_dir, img_name):
    """
    Dump predicted and ground truth masks side by side as images.

    Args:
        preds (torch.Tensor or np.ndarray): Predicted masks of shape (N, H, W).
        gts (torch.Tensor or np.ndarray): Ground truth masks of shape (N, H, W).
        save_dir (str): Directory where the masks should be saved.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    N = preds.shape[0]  # Number of masks

    # Iterate over each mask
    for i in range(N):
        pred_mask = preds[i].detach().cpu().numpy() if isinstance(preds, torch.Tensor) else preds[i]
        gt_mask = gts[i].detach().cpu().numpy() if isinstance(gts, torch.Tensor) else gts[i]

        # Create a plot with side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        # Plot predicted mask
        axes[0].imshow(pred_mask, cmap='viridis')
        axes[0].set_title('Predicted Mask')
        axes[0].axis('off')  # Hide the axes

        # Plot ground truth mask
        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')  # Hide the axes

        # Save the image
        save_path = os.path.join(save_dir, img_name)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved: {save_path}")