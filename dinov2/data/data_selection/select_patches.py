import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageDraw, ImageFont


def load_h5(h5_path):
    import h5py
    with h5py.File(h5_path, 'r') as f:
        feats = f['features'][:]
        coords = f['coords'][:]
    return feats, coords


def process_h5(h5_path):
    feats, coords = load_h5(h5_path)
    return feats, coords

def test_fps(num_clusters=[5, 10, 50, 100]):
    # Test FPS selection
    #feats are 1000 samples in normal distribution
    feats = np.random.randn(1000, 2)
    #num_clusters = 100
    for k in num_clusters:
        fps_selected = select_representative_patches(feats, k, mode='fps')
        #print(f"FPS selected indices for K={k}:", fps_selected)
    #print the selected indices in the different subplots of the figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for i, k in enumerate(num_clusters):
        fps_selected = select_representative_patches(feats, k, mode='fps')
        ax = axes[i // 2, i % 2]
        ax.scatter(feats[:, 0], feats[:, 1], alpha=0.3, label="All Patches")
        ax.scatter(feats[fps_selected, 0], feats[fps_selected, 1], color='red', label="FPS", s=30)
        ax.set_title(f"K={k}")
        ax.legend()
    #plt.tight_layout()
    #save into a figure
    #set title
    plt.suptitle("FPS Selection")
    plt.savefig("fps_selection.png", dpi=300, bbox_inches='tight')


    return None

def select_representative_patches(features, num_clusters, mode='knn'):
    if mode == 'knn':
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        cluster_centers = kmeans.cluster_centers_
        selected_patches = []
        for i in range(len(cluster_centers)):
            cluster_indices = np.where(cluster_labels == i)[0]
            distances = np.linalg.norm(features[cluster_indices] - cluster_centers[i], axis=1)
            selected_patches.append(cluster_indices[np.argmin(distances)])  # Closest sample to cluster center
        return selected_patches

    elif mode == 'fps':
        import torch
        from torch_cluster import fps
        # Convert features to a torch tensor
        features_tensor = torch.from_numpy(features).float()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        features_tensor = features_tensor.to(device)
        # Create a batch vector (all zeros, as all points belong to one slide)
        batch = torch.zeros(features_tensor.size(0), dtype=torch.long, device=device)
        # Compute the sampling ratio. Note: ratio must be between 0 and 1.
        ratio = num_clusters / features_tensor.size(0)
        ratio = min(ratio, 1.0)
        # Run farthest point sampling. This returns indices of selected points.
        indices = fps(features_tensor, batch, ratio=ratio, random_start=True)
        selected_patches = indices.cpu().numpy().tolist()
        # Trim to exactly 'num_clusters' if necessary
        if len(selected_patches) > num_clusters:
            selected_patches = selected_patches[:num_clusters]
        return selected_patches
    elif mode == 'random':
        return random.sample(range(features.shape[0]), num_clusters)
    elif mode == 'knn_multi': #select random N samples to the cluster center
        total_patches = num_clusters
        num_clusters = 20
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        total_samples = features.shape[0]
        selected_patches = []
        for i in range(num_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            cluster_size = len(cluster_indices)
            # Compute the number of samples to select from this cluster.
            # We use the proportion of the cluster size relative to the entire dataset.
            #n_samples = int(round((cluster_size / total_samples) * total_patches))
            n_samples = int(total_patches/num_clusters)
            n_samples = max(1, n_samples)  # ensure at least one sample per cluster
            # Randomly select n_samples from this cluster without replacement if possible.
            replace = False if cluster_size >= n_samples else True
            selected = np.random.choice(cluster_indices, size=n_samples, replace=replace)
            selected_patches.extend(selected.tolist())
        # Adjust total number of selected patches to be exactly K.
        # if len(selected_patches) > total_patches:
        #     # Randomly remove extra samples
        #     selected_patches = np.random.choice(selected_patches, size=total_patches, replace=False).tolist()
        # elif len(selected_patches) < total_patches:
        #     # Add additional samples randomly from the non-selected indices.
        #     all_indices = set(range(total_samples))
        #     current_set = set(selected_patches)
        #     remaining = np.array(list(all_indices - current_set))
        #     additional_needed = total_patches - len(selected_patches)
        #     if len(remaining) >= additional_needed:
        #         additional = np.random.choice(remaining, size=additional_needed, replace=False).tolist()
        #     else:
        #         additional = np.random.choice(remaining, size=additional_needed, replace=True).tolist()
        #     selected_patches.extend(additional)
        print(f"KNN selected indices for K={total_patches}:", len(selected_patches))
        return selected_patches
    else:
        raise ValueError(f"Unknown mode: {mode}. Please use 'knn' or 'fps'.")


def read_selected_imgs(coors, slide_path,wsi,num=9):
    #from cucim.clara import CuImage
    from torchvision.transforms.functional import to_pil_image
    # wsi = CuImage(slide_path)
    imgs = []
    coors = coors[:num]
    for coor in coors:
        x, y = coor
        img = wsi.read_region(location=(x, y), size=(512, 512), level=0)
        img = np.asarray(img, dtype=np.uint8)
        img = to_pil_image(img)
        #resize to 256
        #img = img.resize((256, 256), resample=Image.LANCZOS)
        imgs.append(img)
    return imgs


def build_patch_grid(imgs, grid_size, patch_size, font, text_color,slide_id):
    """Build a grid image from a list of PIL images and annotate them."""
    grid_img = Image.new('RGB', (grid_size * patch_size, grid_size * patch_size), (255, 255, 255))
    draw = ImageDraw.Draw(grid_img)
    for idx, img in enumerate(imgs):
        if idx < grid_size * grid_size:
            img_resized = img.resize((patch_size, patch_size), resample=Image.LANCZOS)
            row = idx // grid_size
            col = idx % grid_size
            grid_img.paste(img_resized, (col * patch_size, row * patch_size))
            # Annotate: label with number (e.g., 1 to num_clusters)
            # text_position = (col * patch_size + 5, row * patch_size + 5)
            # draw.text(text_position, str(idx + 1), fill=text_color, font=font)
    return grid_img

def viz_slide_with_grid(wsi, coords, knn_selected, knn_multi_selected, fps_selected, random_selected, level=3, slide_id=None,num_clusters=9):
    """
    Visualizes a low-resolution version of the slide with overlaid grid boxes.

    Parameters:
      wsi: CuImage instance of the whole slide.
      coords: Array of patch coordinates at level 0 (e.g., [[x1,y1], [x2,y2], ...]).
      knn_selected: List of indices corresponding to selected patches.
      patch_size: Patch size at level 0.
      level: The lower resolution level to display (e.g., 3).
    """
    from cucim.clara import CuImage
    from PIL import Image, ImageDraw
    import numpy as np
    patch_size = 512
    # Get the dimensions of the slide at the desired low resolution level.
    # (Assumes wsi.level_dimensions is available.)
    try:
        low_res_dim  = wsi.resolutions["level_dimensions"][level]  # e.g., (width, height)
    except:
        level = level-1
        low_res_dim = wsi.resolutions["level_dimensions"][level]
    # Read the low-resolution slide image
    slide_img_np = wsi.read_region((0, 0), size=low_res_dim, level=level)
    slide_img_np = np.asarray(slide_img_np, dtype=np.uint8)
    slide_img = Image.fromarray(slide_img_np)
    # Calculate scale factors to map level-0 coordinates to the low resolution level.
    level0_dim = wsi.resolutions["level_dimensions"][0]
    scale_x = low_res_dim[0] / level0_dim[0]
    scale_y = low_res_dim[1] / level0_dim[1]
    scale = (scale_x + scale_y) / 2.0  # assuming near isotropic scaling
    # Compute the patch size in the low resolution image.
    patch_size_low = patch_size * scale
    # Create a drawing context on the low-res slide image.
    draw = ImageDraw.Draw(slide_img)
    # Draw grid boxes for each patch coordinate.
    # for i, (x, y) in enumerate(coords):
    #     # Scale the coordinate from level 0 to the low-res level.
    #     x_low = x * scale
    #     y_low = y * scale
    #     bbox = [x_low, y_low, x_low + patch_size_low, y_low + patch_size_low]
    #     draw.rectangle(bbox, outline=(0, 0, 255), width=1)

    for i, (x, y) in enumerate(coords):
        # If this patch is selected, use a different color.
        # if i in knn_selected:
        #     color = 'red'  # red for selected patches
        # elif i in fps_selected:
        #     color = 'blue'  # blue for non-selected patches
        if i in random_selected:
            color = 'blue'
        elif i in knn_multi_selected:
            color = 'black'
        else:
            continue
        # Scale the coordinate from level 0 to the low-res level.
        x_low = x * scale
        y_low = y * scale
        bbox = [x_low, y_low, x_low + patch_size_low, y_low + patch_size_low]
        draw.rectangle(bbox, outline='black', width=1,fill=color)


    # Save to file:
    os.makedirs("slides", exist_ok=True)
    out_path = f"slides/selected_patches_{slide_id}_k={num_clusters}.png" if slide_id else "slides/selected_patches.png"
    slide_img.save(out_path)
    return out_path


# Example usage:
# Assuming you have a CuImage instance 'wsi', an array 'coords' of patch coordinates,
# and a list 'knn_selected' with indices of selected patches.
# viz_slide_with_grid(wsi, coords, knn_selected, patch_size=256, level=3)
def create_pca_patches_composite(feats, coords, knn_selected, fps_selected, random_selected,
                                 knn_imgs, fps_imgs, random_imgs, slide_id=None,num_clusters=9):
    """
    Create a composite figure that includes:
      - A PCA scatter plot of all patches with points highlighted for each selection.
      - Patch grids for KNN, FPS, and random selections.
    The composite is saved as "pca_patches_{slide_id}.png".
    """
    # Set up figure layout with GridSpec:
    # Row0: PCA plot (span full width)
    # Row1: Two patch grids: left = KNN grid, right = FPS grid.
    # Row2: A single patch grid for Random selections (spanning full width).
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, height_ratios=[1, 1, 1])

    # PCA plot in row 0 spanning both columns
    ax_pca = fig.add_subplot(gs[:, 0])
    pca = PCA(n_components=2)
    pca_feats = pca.fit_transform(feats)
    ax_pca.scatter(pca_feats[:, 0], pca_feats[:, 1], alpha=0.3,color='gray', label="All Patches")
    ax_pca.scatter(pca_feats[random_selected, 0], pca_feats[random_selected, 1], color='green', alpha=0.7,
                   label="Random", s=30)
    ax_pca.scatter(pca_feats[fps_selected, 0], pca_feats[fps_selected, 1], color='blue', alpha=0.7, label="FPS", s=30)
    ax_pca.scatter(pca_feats[knn_selected, 0], pca_feats[knn_selected, 1], color='red',alpha=0.7, label="KNN",s=30)


    ax_pca.set_title("PCA Projection")
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")
    ax_pca.legend()

    # Parameters for patch grids
    grid_size = int(np.sqrt(len(knn_imgs)))  # assuming same number for each grid
    patch_size_disp = 128
    font = ImageFont.load_default()  # adjust or use truetype for larger font if needed

    # KNN patch grid: place in Row1, Col0
    ax_knn = fig.add_subplot(gs[0, 1])
    knn_grid = build_patch_grid(knn_imgs, grid_size, patch_size_disp, font, text_color=(255, 0, 0), slide_id=slide_id)
    ax_knn.imshow(knn_grid)
    ax_knn.axis("off")
    ax_knn.set_title("KNN Patches")

    # FPS patch grid: place in Row1, Col1
    ax_fps = fig.add_subplot(gs[1, 1])
    fps_grid = build_patch_grid(fps_imgs, grid_size, patch_size_disp, font, text_color=(0, 0, 255), slide_id=slide_id)
    ax_fps.imshow(fps_grid)
    ax_fps.axis("off")
    ax_fps.set_title("FPS Patches")

    # Random patch grid: place in Row2 spanning both columns
    ax_rand = fig.add_subplot(gs[2, 1])
    rand_grid = build_patch_grid(random_imgs, grid_size, patch_size_disp, font, text_color=(0, 0, 0), slide_id=slide_id)
    ax_rand.imshow(rand_grid)
    ax_rand.axis("off")
    ax_rand.set_title("Randomly Sampled Patches")

    out_path = f"pca_patches_{slide_id}.png" if slide_id else "pca_patches.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path


def final_visualization(wsi, feats, coords, knn_selected, knn_multi_selected, fps_selected, random_selected, slide_path, slide_id,num_clusters,draw_patches):
    """
    Two-step final visualization:
      Step 1: Visualize the slide with grid boxes and save as an image.
      Step 2: Create a composite image (PCA and patch grids) and save it.
      Finally, load these two images into one final figure with two subplots.
    """
    # Step 1: Create and save WSI with grid boxes.
    wsi_img_path = viz_slide_with_grid(wsi, coords, knn_selected, knn_multi_selected, fps_selected, random_selected, level=3,
                                       slide_id=slide_id,num_clusters=num_clusters)

    if draw_patches:
        # Step 2: Read patch images for each method.
        knn_imgs = read_selected_imgs(coors=coords[knn_selected], slide_path=slide_path, wsi=wsi) if knn_selected else []
        #knn_multi_imgs = read_selected_imgs(coors=coords[knn_multi_selected], slide_path=slide_path, wsi=wsi) if knn_multi_selected else []
        fps_imgs = read_selected_imgs(coors=coords[fps_selected], slide_path=slide_path, wsi=wsi) if fps_selected else []
        random_imgs = read_selected_imgs(coors=coords[random_selected], slide_path=slide_path, wsi=wsi) if random_selected else []

        composite_img_path = create_pca_patches_composite(feats, coords, knn_selected, fps_selected, random_selected,
                                                          knn_imgs, fps_imgs, random_imgs, slide_id=slide_id,num_clusters=num_clusters)

        # Step 3: Load the two saved images and combine them in a final figure.
        final_fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        wsi_img = Image.open(wsi_img_path)
        comp_img = Image.open(composite_img_path)
        axes[0].imshow(wsi_img)
        axes[0].axis("off")
        axes[0].set_title(f"WSI with Grid Boxes (K={num_clusters})")
        axes[1].imshow(comp_img)
        axes[1].axis("off")
        axes[1].set_title(f"PCA & Patch Grids (K={num_clusters})")

        final_out = f"combined_selection/final_viz_{slide_id}_k={num_clusters}.png" if slide_id else "final_viz.png"
        final_fig.savefig(final_out, dpi=300, bbox_inches='tight')
        plt.close(final_fig)
    return None

def select_patches(num_clusters=9, h5_folder='/mnt/nfs01-R0/TUM_breast_cancer/feats_uni2/h5_files/',slides_folder='',slides_ls=[],draw_patches=False):
    h5_files = os.listdir(h5_folder)
    i = 0
    #load the csv file with format like
    #slide_path, patch_path
    # /mnt/nfs01-R0/TUM_breast_cancer/neg/h2023016486t3-a-1_114844.svs
    # /mnt/nfs01-R0/TUM_breast_cancer/neg/h2023004265t1-c-1_132110.svs
    import pandas as pd
    #df = pd.read_csv('viz_slides.csv')
    #load slide_path
    #slide_path_ls = df['slide_path'].tolist()

    for h5_file in h5_files:
        h5_path = os.path.join(h5_folder, h5_file)
        slide_path = slides_folder + h5_file.replace('.h5', '.svs')
        slide_path = slide_path if os.path.exists(slide_path) else slides_folder + h5_file.replace('.h5', '.tif')
        print(h5_file.replace('.h5', ''))
        if h5_file.replace('.h5', '') not in slides_ls:
            continue
        # Step 1: Load features and coordinates
        feats, coords = process_h5(h5_path)
        # Step 2: Obtain both KNN and FPS selected patch indices
        knn_selected = select_representative_patches(feats, num_clusters, mode='knn')
        knn_multi_selected = select_representative_patches(feats, num_clusters, mode='knn_multi')
        fps_selected = select_representative_patches(feats, num_clusters, mode='fps')
        random_selected = select_representative_patches(feats, num_clusters, mode='random')
        # print("KNN selected indices:", knn_selected)
        # print("FPS selected indices:", fps_selected)

        # Step 2.5: Visualize the selected samples in WS image
        from cucim.clara import CuImage
        wsi = CuImage(slide_path)
        slide_id = h5_file.replace('.h5', '')
        #viz_slide_with_grid(wsi, coords, knn_selected, fps_selected,random_selected, 3,slide_id)

        # Step 3: PCA for visualization (same PCA for both methods)
        # knn_imgs = read_selected_imgs(coors=coords[knn_selected], slide_path=slide_path,wsi=wsi)
        # fps_imgs = read_selected_imgs(coors=coords[fps_selected], slide_path=slide_path,wsi=wsi)
        # random_imgs = read_selected_imgs(coors=coords[random_selected], slide_path=slide_path,wsi=wsi)
        #create_pca_patches_composite(feats, coords, knn_selected, fps_selected, random_selected,
                                     #knn_imgs, fps_imgs, random_imgs, slide_id=slide_id)

        # Step 4: Final visualization combining PCA and patch grids
        final_visualization(wsi, feats, coords, knn_selected,knn_multi_selected, fps_selected, random_selected, slide_path, slide_id,num_clusters, draw_patches)


        # pca = PCA(n_components=2)
        # pca_feats = pca.fit_transform(feats)
        #
        # # Step 4: Read the selected patches images for both methods
        #

        #
        # # Step 5: Create a combined figure using GridSpec:
        # fig = plt.figure(figsize=(16, 10))
        # gs = GridSpec(3, 2, width_ratios=[2, 1])
        #
        # # Left: PCA plot spanning two rows
        # ax_pca = fig.add_subplot(gs[:, 0])
        # ax_pca.scatter(pca_feats[:, 0], pca_feats[:, 1], alpha=0.5, label="All Patches")
        # # Plot KNN selected points in red
        # ax_pca.scatter(pca_feats[knn_selected, 0], pca_feats[knn_selected, 1], color='red', label="KNN")
        # # Plot FPS selected points in green
        # ax_pca.scatter(pca_feats[fps_selected, 0], pca_feats[fps_selected, 1], color='Magenta', label="FPS")
        # ax_pca.scatter(pca_feats[random_selected, 0], pca_feats[random_selected, 1], color='black', label="Random")
        # ax_pca.set_title("PCA Projection of Feature Space")
        # ax_pca.set_xlabel("PCA Component 1")
        # ax_pca.set_ylabel("PCA Component 2")
        # ax_pca.legend()
        # Annotate: add labels “K1”, “K2”, ... for KNN and “F1”, “F2”, ... for FPS.
        # for j, idx in enumerate(knn_selected):
        #     x, y = pca_feats[idx, 0], pca_feats[idx, 1]
        #     ax_pca.text(x + 0.2, y + 0.2, f"K{j + 1}", color='red', fontsize=10, weight='bold')
        # for j, idx in enumerate(fps_selected):
        #     x, y = pca_feats[idx, 0], pca_feats[idx, 1]
        #     ax_pca.text(x + 0.2, y - 0.2, f"F{j + 1}", color='Magenta', fontsize=10, weight='bold')

        # Right top: KNN patches grid
        # ax_knn = fig.add_subplot(gs[0, 1])
        # grid_size = int(np.sqrt(num_clusters))
        # patch_size = 128  # Adjust as needed
        # font = ImageFont.load_default(size=20)  # or use truetype if available: ImageFont.truetype("arial.ttf", 20)
        # knn_grid_img = build_patch_grid(knn_imgs, grid_size, patch_size, font, text_color=(255, 0, 0))
        # ax_knn.imshow(knn_grid_img)
        # ax_knn.axis("off")
        # ax_knn.set_title("KNN Selected Patches")
        #
        # # Right bottom: FPS patches grid
        # ax_fps = fig.add_subplot(gs[1, 1])
        # fps_grid_img = build_patch_grid(fps_imgs, grid_size, patch_size, font, text_color=(255, 0, 255))
        # ax_fps.imshow(fps_grid_img)
        # ax_fps.axis("off")
        # ax_fps.set_title("FPS Selected Patches")
        #
        # # Right bottom: FPS patches grid
        # ax_fps = fig.add_subplot(gs[2, 1])
        # fps_grid_img = build_patch_grid(fps_imgs, grid_size, patch_size, font, text_color=(255, 0, 255))
        # ax_fps.imshow(fps_grid_img)
        # ax_fps.axis("off")
        # ax_fps.set_title("FPS Selected Patches")
        #
        # # Save the combined figure
        # output_dir = "combined_selection"
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # fig.savefig(os.path.join(output_dir, "selected_patches_" + h5_file.replace('.svs', '') + ".png"), dpi=300,
        #             bbox_inches='tight')

        #plt.close(fig)
        i += 1
        #break
def draw_multi_cluster_for_single_slide(slides_ls=[]):
    #for each slide, draw the viz_slide_with_grid of different k in [9, 100, 500, 1000] figures as subplots in one matplotlib figure

    for i, slide_id in enumerate(slides_ls):
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        for j, k in enumerate([100,200, 300,400, 500]):
            #read the generated imges with title like "selected_patches_{slide_id}_k={num_clusters}.png" into each subplot
            img_path = f"slides/selected_patches_{slide_id}_k={k}.png"
            img = Image.open(img_path)
            ax = axes[j // 2, j % 2]
            ax.imshow(img)
            #set the title
            ax.set_title(f"K={k}")
            #set the total title of the figure
            #plt.suptitle(f"Slide {slide_id} patch selection")
            #set legend manually: blue box for FPS, red box for KNN, green box for Random
            #only add legend for the first subplot
            if j == 0:
                import matplotlib.patches as mpatches
                # ax.legend(handles=[mpatches.Patch(facecolor='blue', edgecolor='black', label='FPS'),
                #                    mpatches.Patch(facecolor='red', edgecolor='black', label='KNN'),
                #                    mpatches.Patch(facecolor='black', edgecolor='black', label='KNN_multi'),
                #                    mpatches.Patch(facecolor='green', edgecolor='black', label='Random')])

                ax.legend(handles=[
                                   mpatches.Patch(facecolor='black', edgecolor='black', label='KNN_multi'),
                                   mpatches.Patch(facecolor='blue', edgecolor='black', label='Random')])
        #save the figure
        plt.suptitle(f"Slide {i} patch selection", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"knn_multi/{slide_id}_patch_selection.png", dpi=300, bbox_inches='tight')
        plt.close(fig)




slides_ls = ['h2021002184t3-p-1_125914','73f792fd-b354-383a-11ec-7ceb7002fe29_120957']
#test_fps()
#for k in [9, 100, 500, 1000]:
   #select_patches(num_clusters=k, h5_folder='/mnt/nfs01-R0/patch_selection/feats/h5_files/',slides_folder='/mnt/nfs01-R0/patch_selection/slides/',slides_ls=slides_ls)
#draw_multi_cluster_for_single_slide(slides_ls=slides_ls)

for k in [100,200, 300,400, 500]:
    select_patches(num_clusters=k, h5_folder='/mnt/nfs01-R0/patch_selection/feats/h5_files/',slides_folder='/mnt/nfs01-R0/patch_selection/slides/',slides_ls=slides_ls,draw_patches=False)
draw_multi_cluster_for_single_slide(slides_ls=slides_ls)