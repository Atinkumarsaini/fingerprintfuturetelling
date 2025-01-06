#  github code -----------------------------

# import cv2 as cv
# from glob import glob
# import os
# import numpy as np
# from utils.poincare import calculate_singularities
# from utils.segmentation import create_segmented_and_variance_images
# from utils.normalization import normalize
# from utils.gabor_filter import gabor_filter
# from utils.frequency import ridge_freq
# from utils import orientation
# from utils.crossing_number import calculate_minutiaes
# from tqdm import tqdm
# from utils.skeletonize import skeletonize


# def fingerprint_pipline(input_img):
#     block_size = 16

#     # pipe line picture re https://www.cse.iitk.ac.in/users/biometrics/pages/111.JPG
#     # normalization -> orientation -> frequency -> mask -> filtering

#     # normalization - removes the effects of sensor noise and finger pressure differences.
#     normalized_img = normalize(input_img.copy(), float(100), float(100))

#     # color threshold
#     # threshold_img = normalized_img
#     # _, threshold_im = cv.threshold(normalized_img,127,255,cv.THRESH_OTSU)
#     # cv.imshow('color_threshold', normalized_img); cv.waitKeyEx()

#     # ROI and normalisation
#     (segmented_img, normim, mask) = create_segmented_and_variance_images(normalized_img, block_size, 0.2)

#     # orientations
#     angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=False)
#     orientation_img = orientation.visualize_angles(segmented_img, mask, angles, W=block_size)

#     # find the overall frequency of ridges in Wavelet Domain
#     freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)

#     # create gabor filter and do the actual filtering
#     gabor_img = gabor_filter(normim, angles, freq)

#     # thinning oor skeletonize
#     thin_image = skeletonize(gabor_img)

#     # minutias
#     minutias = calculate_minutiaes(thin_image)

#     # singularities
#     singularities_img = calculate_singularities(thin_image, angles, 1, block_size, mask)

#     # visualize pipeline stage by stage
#     output_imgs = [input_img, normalized_img, segmented_img, orientation_img, gabor_img, thin_image, minutias, singularities_img]
#     for i in range(len(output_imgs)):
#         if len(output_imgs[i].shape) == 2:
#             output_imgs[i] = cv.cvtColor(output_imgs[i], cv.COLOR_GRAY2RGB)
#     results = np.concatenate([np.concatenate(output_imgs[:4], 1), np.concatenate(output_imgs[4:], 1)]).astype(np.uint8)

#     return results


# if __name__ == '__main__':
#     # open images
#     img_dir = 'sample_inputs/*'
#     output_dir = 'output1/'
#     def open_images(directory):
#         images_paths = glob(directory)
#         return np.array([cv.imread(img_path,0) for img_path in images_paths])

#     images = open_images(img_dir)

#     # image pipeline
#     os.makedirs(output_dir, exist_ok=True)
#     for i, img in enumerate(tqdm(images)):
#         results = fingerprint_pipline(img)
#         cv.imwrite(output_dir+str(i)+'.png', results)
#         # cv.imshow('image pipeline', results); cv.waitKeyEx()




# main code -------------------------------------------

import cv2 as cv
from glob import glob
import os
import numpy as np
from utils.poincare import calculate_singularities
from utils.segmentation import create_segmented_and_variance_images
from utils.normalization import normalize
from utils.gabor_filter import gabor_filter
from utils.frequency import ridge_freq
from utils import orientation
from utils.crossing_number import calculate_minutiaes
from tqdm import tqdm
from utils.skeletonize import skeletonize

def fingerprint_pipline(input_img):
    block_size = 16

    # normalization
    normalized_img = normalize(input_img.copy(), float(100), float(100))

    # ROI and normalisation
    (segmented_img, normim, mask) = create_segmented_and_variance_images(normalized_img, block_size, 0.2)

    # orientations
    angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=False)
    orientation_img = orientation.visualize_angles(segmented_img, mask, angles, W=block_size)

    # find the overall frequency of ridges
    freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)

    # create gabor filter and do the actual filtering
    gabor_img = gabor_filter(normim, angles, freq)

    # thinning
    thin_image = skeletonize(gabor_img)

    # minutias
    minutias = calculate_minutiaes(thin_image)

    # singularities with connecting lines and ridge counting
    singularities_img = calculate_singularities(thin_image, angles, 1, block_size, mask)

    # visualize pipeline
    output_imgs = [input_img, normalized_img, segmented_img, orientation_img, gabor_img, thin_image, minutias, singularities_img]
    for i in range(len(output_imgs)):
        if len(output_imgs[i].shape) == 2:
            output_imgs[i] = cv.cvtColor(output_imgs[i], cv.COLOR_GRAY2RGB)
    results = np.concatenate([np.concatenate(output_imgs[:4], 1), np.concatenate(output_imgs[4:], 1)]).astype(np.uint8)

    return results

# if __name__ == '__main__':
#     # open images
#     img_dir = '617_MARIA GORETI KULLU/*'
#     output_dir = 'output1/'
    
#     def open_images(directory):
#         images_paths = glob(directory)
#         return np.array([cv.imread(img_path,0) for img_path in images_paths])

#     images = open_images(img_dir)

#     # image pipeline
#     os.makedirs(output_dir, exist_ok=True)
#     for i, img in enumerate(tqdm(images)):
#         results = fingerprint_pipline(img)
#         cv.imwrite(output_dir+str(i)+'.png', results)








# if __name__ == '__main__':
#     # open images
#     img_dir = '617_MARIA GORETI KULLU/*'
#     output_dir = 'output1/'
    
#     def open_images(directory):
#         images_paths = glob(directory)
#         return np.array([cv.imread(img_path,0) for img_path in images_paths]), images_paths

#     images, image_paths = open_images(img_dir)
#     ridge_counts = []

#     # image pipeline
#     os.makedirs(output_dir, exist_ok=True)
#     for i, (img, img_path) in enumerate(tqdm(zip(images, image_paths))):
#         # Modify fingerprint_pipeline function to return ridge count
#         def fingerprint_pipline(input_img):
#             block_size = 16
#             normalized_img = normalize(input_img.copy(), float(100), float(100))
#             (segmented_img, normim, mask) = create_segmented_and_variance_images(normalized_img, block_size, 0.2)
#             angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=False)
#             orientation_img = orientation.visualize_angles(segmented_img, mask, angles, W=block_size)
#             freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)
#             gabor_img = gabor_filter(normim, angles, freq)
#             thin_image = skeletonize(gabor_img)
#             minutias = calculate_minutiaes(thin_image)
#             singularities_img, ridge_count = calculate_singularities(thin_image, angles, 1, block_size, mask)
            
#             output_imgs = [input_img, normalized_img, segmented_img, orientation_img, gabor_img, thin_image, minutias, singularities_img]
#             for i in range(len(output_imgs)):
#                 if len(output_imgs[i].shape) == 2:
#                     output_imgs[i] = cv.cvtColor(output_imgs[i], cv.COLOR_GRAY2RGB)
#             results = np.concatenate([np.concatenate(output_imgs[:4], 1), np.concatenate(output_imgs[4:], 1)]).astype(np.uint8)
            
#             return results, ridge_count

#         results, count = fingerprint_pipline(img)
#         filename = os.path.basename(img_path)
#         ridge_counts.append((filename, count))
#         cv.imwrite(output_dir+str(i)+'.png', results)

#     # Print ridge counts
#     print("\nRidge Counts:")
#     print("[")
#     for filename, count in ridge_counts:
#         print(f"    ({filename}, no of count - {count}),")
#     print("]")









if __name__ == '__main__':
    # open images
    img_dir = '617_MARIA GORETI KULLU/*'
    output_dir = 'output1/'
    
    def open_images(directory):
        images_paths = glob(directory)
        return np.array([cv.imread(img_path,0) for img_path in images_paths]), images_paths

    images, image_paths = open_images(img_dir)
    fingerprint_data = []

    # image pipeline
    os.makedirs(output_dir, exist_ok=True)
    for i, (img, img_path) in enumerate(tqdm(zip(images, image_paths))):
        # Modify fingerprint_pipeline function to return all data
        def fingerprint_pipline(input_img):
            block_size = 16
            normalized_img = normalize(input_img.copy(), float(100), float(100))
            (segmented_img, normim, mask) = create_segmented_and_variance_images(normalized_img, block_size, 0.2)
            angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=False)
            orientation_img = orientation.visualize_angles(segmented_img, mask, angles, W=block_size)
            freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)
            gabor_img = gabor_filter(normim, angles, freq)
            thin_image = skeletonize(gabor_img)
            minutias = calculate_minutiaes(thin_image)
            singularities_img, ridge_count, num_loops, num_deltas = calculate_singularities(thin_image, angles, 1, block_size, mask)
            
            output_imgs = [input_img, normalized_img, segmented_img, orientation_img, gabor_img, thin_image, minutias, singularities_img]
            for i in range(len(output_imgs)):
                if len(output_imgs[i].shape) == 2:
                    output_imgs[i] = cv.cvtColor(output_imgs[i], cv.COLOR_GRAY2RGB)
            results = np.concatenate([np.concatenate(output_imgs[:4], 1), np.concatenate(output_imgs[4:], 1)]).astype(np.uint8)
            
            return results, ridge_count, num_loops, num_deltas

        results, count, loops, deltas = fingerprint_pipline(img)
        filename = os.path.basename(img_path)
        fingerprint_data.append({
            'filename': filename,
            'ridge_count': count,
            'loops': loops,
            'deltas': deltas
        })
        cv.imwrite(output_dir+str(i)+'.png', results)

    # Print results in the exact format requested
    print("\nFingerprint Analysis Results:")
    print("[")
    for data in fingerprint_data:
        loop_text = "no loop found" if data['loops'] == 0 else f"{data['loops']} loop found"
        delta_text = "no delta found" if data['deltas'] == 0 else f"{data['deltas']} delta found"
        print(f"    ({data['filename']}, no of count - {data['ridge_count']}, {loop_text}, {delta_text}),")
    print("]")







# --------------------------------------------------