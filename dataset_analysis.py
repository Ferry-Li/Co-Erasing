import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load CLIP model and processor for feature extraction
clip_model = CLIPModel.from_pretrained("/data/feiran/robustDiffusion/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("/data/feiran/robustDiffusion/clip-vit-base-patch32")

dir_num = 2


if dir_num == 2:

    # Define paths for the two directories
    dir1 = "/data/feiran/dataset/nsfw_dataset_v1/porn"
    dir2 = "/data/feiran/robustDiffusion/generation_dataset_v1_5/nudity"

    # Helper function to load and process images
    def load_images_from_dir(dir_path):
        images = []
        for file_name in os.listdir(dir_path)[:500]:
            file_path = os.path.join(dir_path, file_name)
            if os.path.isfile(file_path):
                image = Image.open(file_path).convert("RGB")
                images.append(image)
        return images

    # Get image embeddings
    def get_image_embeddings(images):
        inputs = clip_processor(images=images, return_tensors="pt")
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
        return image_features

    # Load images and get embeddings for each directory
    images_dir1 = load_images_from_dir(dir1)
    images_dir2 = load_images_from_dir(dir2)

    # Get embeddings and concatenate
    embeddings_dir1 = get_image_embeddings(images_dir1)
    embeddings_dir2 = get_image_embeddings(images_dir2)
    all_embeddings = torch.cat([embeddings_dir1, embeddings_dir2], dim=0)

    # Reduce embeddings to 2D with PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings.cpu().numpy())

    # Plotting
    plt.figure(figsize=(10, 10))
    # plt.axis('off')
    plt.scatter(embeddings_2d[:len(images_dir1), 0], embeddings_2d[:len(images_dir1), 1], color=(204 / 255, 135 / 255, 135 / 255, 0.5), s=300, linewidth=0.5, edgecolors='black', label='Real NSFW (porn)')
    plt.scatter(embeddings_2d[len(images_dir1):, 0], embeddings_2d[len(images_dir1):, 1], color=(157 / 255, 215 / 255, 227 / 255, 0.5), s=300, linewidth=0.5, edgecolors='black', label='Synthetic NSFW')
    # plt.xlabel("PCA Component 1")
    # plt.ylabel("PCA Component 2")
    plt.xticks([])  # Remove x-axis labels
    plt.yticks([])  # Remove y-axis labels
    plt.legend(loc="upper right", fontsize=24)
    # plt.title("Distribution of different datasets", fontsize=28)
    plt.savefig("2d_scatter_plot_class2.png", bbox_inches='tight')
    plt.show()

elif dir_num == 3:
    # Define paths for the three directories
    dir1 = "/data/feiran/dataset/nsfw_dataset_v1/porn"
    dir2 = "/data/feiran/robustDiffusion/generation_dataset_v1_5/nudity"
    dir3 = "/data/feiran/dataset/nsfw_dataset_v1/sexy"

    # Helper function to load and process images
    def load_images_from_dir(dir_path):
        images = []
        for file_name in os.listdir(dir_path)[:500]:
            file_path = os.path.join(dir_path, file_name)
            if os.path.isfile(file_path):
                image = Image.open(file_path).convert("RGB")
                images.append(image)
        return images

    # Get image embeddings
    def get_image_embeddings(images):
        inputs = clip_processor(images=images, return_tensors="pt")
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
        return image_features

    # Load images and get embeddings for each directory
    images_dir1 = load_images_from_dir(dir1)
    images_dir2 = load_images_from_dir(dir2)
    images_dir3 = load_images_from_dir(dir3)

    # Get embeddings and concatenate
    embeddings_dir1 = get_image_embeddings(images_dir1)
    embeddings_dir2 = get_image_embeddings(images_dir2)
    embeddings_dir3 = get_image_embeddings(images_dir3)
    all_embeddings = torch.cat([embeddings_dir1, embeddings_dir2, embeddings_dir3], dim=0)

    # Reduce embeddings to 2D with PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings.cpu().numpy())

    # Plotting
    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings_2d[:len(images_dir1), 0], embeddings_2d[:len(images_dir1), 1], color=(204 / 255, 135 / 255, 135 / 255, 0.5), s=300, edgecolors='black', linewidth=0.5, label='Real NSFW (porn)')
    plt.scatter(embeddings_2d[len(images_dir1):len(images_dir1) + len(images_dir2), 0],
                embeddings_2d[len(images_dir1):len(images_dir1) + len(images_dir2), 1], color=(157 / 255, 215 / 255, 227 / 255, 0.5), s=300, edgecolors='black', linewidth=0.5, label='Synthetic NSFW')
    plt.scatter(embeddings_2d[len(images_dir1) + len(images_dir2):, 0],
                embeddings_2d[len(images_dir1) + len(images_dir2):, 1], color=(249 / 255, 233 / 255, 164 / 255, 0.5), s=300, edgecolors='black', linewidth=0.5, label='Real NSFW (sexy)')
    # plt.xlabel("PCA Component 1")
    # plt.ylabel("PCA Component 2")
    plt.xticks([])  # Remove x-axis labels
    plt.yticks([])
    plt.legend(loc="upper left", fontsize=24)
    # plt.title("2D Scatter Plot of Image Embeddings", fontsize=28)
    plt.savefig("2d_scatter_plot_class3.png", bbox_inches='tight')
    plt.show()