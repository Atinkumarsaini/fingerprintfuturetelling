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

#     # normalization
#     normalized_img = normalize(input_img.copy(), float(100), float(100))

#     # ROI and normalisation
#     (segmented_img, normim, mask) = create_segmented_and_variance_images(normalized_img, block_size, 0.2)

#     # orientations
#     angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=False)
#     orientation_img = orientation.visualize_angles(segmented_img, mask, angles, W=block_size)

#     # find the overall frequency of ridges
#     freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)

#     # create gabor filter and do the actual filtering
#     gabor_img = gabor_filter(normim, angles, freq)

#     # thinning
#     thin_image = skeletonize(gabor_img)

#     # minutias
#     minutias = calculate_minutiaes(thin_image)

#     # singularities with connecting lines and ridge counting
#     singularities_img = calculate_singularities(thin_image, angles, 1, block_size, mask)

#     # visualize pipeline
#     output_imgs = [input_img, normalized_img, segmented_img, orientation_img, gabor_img, thin_image, minutias, singularities_img]
#     for i in range(len(output_imgs)):
#         if len(output_imgs[i].shape) == 2:
#             output_imgs[i] = cv.cvtColor(output_imgs[i], cv.COLOR_GRAY2RGB)
#     results = np.concatenate([np.concatenate(output_imgs[:4], 1), np.concatenate(output_imgs[4:], 1)]).astype(np.uint8)

#     return results



#  main code -------------------

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





# print statement of image and count --------------



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
#         print(f"    ({filename}, no of ridge count - {count}),")
#     print("]")







# print statement of loop and delta ----------------------


# if __name__ == '__main__':
#     # open images
#     img_dir = '617_MARIA GORETI KULLU/*'
#     output_dir = 'output1/'
    
#     def open_images(directory):
#         images_paths = glob(directory)
#         return np.array([cv.imread(img_path,0) for img_path in images_paths]), images_paths

#     images, image_paths = open_images(img_dir)
#     fingerprint_data = []

#     # image pipeline
#     os.makedirs(output_dir, exist_ok=True)
#     for i, (img, img_path) in enumerate(tqdm(zip(images, image_paths))):
#         # Modify fingerprint_pipeline function to return all data
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
#             singularities_img, ridge_count, num_loops, num_deltas = calculate_singularities(thin_image, angles, 1, block_size, mask)
            
#             output_imgs = [input_img, normalized_img, segmented_img, orientation_img, gabor_img, thin_image, minutias, singularities_img]
#             for i in range(len(output_imgs)):
#                 if len(output_imgs[i].shape) == 2:
#                     output_imgs[i] = cv.cvtColor(output_imgs[i], cv.COLOR_GRAY2RGB)
#             results = np.concatenate([np.concatenate(output_imgs[:4], 1), np.concatenate(output_imgs[4:], 1)]).astype(np.uint8)
            
#             return results, ridge_count, num_loops, num_deltas

#         results, count, loops, deltas = fingerprint_pipline(img)
#         filename = os.path.basename(img_path)
#         fingerprint_data.append({
#             'filename': filename,
#             'ridge_count': count,
#             'loops': loops,
#             'deltas': deltas
#         })
#         cv.imwrite(output_dir+str(i)+'.png', results)

#     # Print results in the exact format requested
#     print("\nFingerprint Analysis Results:")
#     print("[")
#     for data in fingerprint_data:
#         loop_text = "no loop found" if data['loops'] == 0 else f"{data['loops']} loop found"
#         delta_text = "no delta found" if data['deltas'] == 0 else f"{data['deltas']} delta found"
#         print(f"    ({data['filename']}, no of count - {data['ridge_count']}, {loop_text}, {delta_text}),")
#     print("]")







# # --------------------------------------------------

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

# def fingerprint_pipeline(input_img):
#     block_size = 16

#     # normalization
#     normalized_img = normalize(input_img.copy(), float(100), float(100))

#     # ROI and normalisation
#     (segmented_img, normim, mask) = create_segmented_and_variance_images(normalized_img, block_size, 0.2)

#     # orientations
#     angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=False)
#     orientation_img = orientation.visualize_angles(segmented_img, mask, angles, W=block_size)

#     # find the overall frequency of ridges
#     freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)

#     # create gabor filter and do the actual filtering
#     gabor_img = gabor_filter(normim, angles, freq)

#     # thinning
#     thin_image = skeletonize(gabor_img)

#     # minutias
#     minutias = calculate_minutiaes(thin_image)

#     # singularities with connecting lines and ridge counting
#     singularities_img, ridge_count, num_loops, num_deltas = calculate_singularities(thin_image, angles, 1, block_size, mask)

#     # visualize pipeline
#     output_imgs = [input_img, normalized_img, segmented_img, orientation_img, gabor_img, thin_image, minutias, singularities_img]
#     for i in range(len(output_imgs)):
#         if len(output_imgs[i].shape) == 2:
#             output_imgs[i] = cv.cvtColor(output_imgs[i], cv.COLOR_GRAY2RGB)
#     results = np.concatenate([np.concatenate(output_imgs[:4], 1), np.concatenate(output_imgs[4:], 1)]).astype(np.uint8)

#     return results, ridge_count, num_loops, num_deltas

# def process_fingerprints(img_dir, output_dir):
#     # open images
#     def open_images(directory):
#         images_paths = glob(directory)
#         return np.array([cv.imread(img_path, 0) for img_path in images_paths]), images_paths

#     images, image_paths = open_images(img_dir)
#     fingerprint_data = []

#     # image pipeline
#     os.makedirs(output_dir, exist_ok=True)
#     for i, (img, img_path) in enumerate(tqdm(zip(images, image_paths))):
#         results, count, loops, deltas = fingerprint_pipeline(img)
#         filename = os.path.basename(img_path)
#         fingerprint_data.append({
#             'filename': filename,
#             'ridge_count': count,
#             'loops': loops,
#             'deltas': deltas
#         })
#         cv.imwrite(output_dir + str(i) + '.png', results)

#     # Print results in the exact format requested
#     print("\nFingerprint Analysis Results:")
#     print("[")
#     for data in fingerprint_data:
#         loop_text = "no loop found" if data['loops'] == 0 else f"{data['loops']} loop found"
#         delta_text = "no delta found" if data['deltas'] == 0 else f"{data['deltas']} delta found"
#         print(f"    ({data['filename']}, no of count - {data['ridge_count']}, {loop_text}, {delta_text}),")
#     print("]")

# if __name__ == '__main__':
#     img_dir = '617_MARIA GORETI KULLU/L1ThumbC.bmp'
#     output_dir = 'output1/'
#     process_fingerprints(img_dir, output_dir)











# fastapi code ---------------------------------------


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2 as cv
import numpy as np
import os
from typing import List
import shutil
from datetime import datetime
import uvicorn
from glob import glob

# Import your existing utilities
from utils.poincare import calculate_singularities
from utils.segmentation import create_segmented_and_variance_images
from utils.normalization import normalize
from utils.gabor_filter import gabor_filter
from utils.frequency import ridge_freq
from utils import orientation
from utils.crossing_number import calculate_minutiaes
from utils.skeletonize import skeletonize

app = FastAPI(
    title="Fingerprint Analysis API",
    description="API for fingerprint image processing and analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directory if it doesn't exist
OUTPUT_DIR = "output_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def ensure_uint8(img):
    """Convert image to uint8 format"""
    if img is None:
        return None
    
    # Normalize float images to 0-255 range
    if img.dtype == np.float64 or img.dtype == np.float32:
        img = np.clip(img * 255.0, 0, 255)  # Scale floats to 0-255 range
        img = img.astype(np.uint8)
    return img

def fingerprint_pipeline(input_img):
    """Enhanced fingerprint pipeline with proper image type handling"""
    try:
        block_size = 16
        
        # Input validation
        if input_img is None or input_img.size == 0:
            raise ValueError("Invalid input image")
        
        if min(input_img.shape) < block_size:
            raise ValueError(f"Image dimensions too small, minimum size is {block_size}x{block_size}")

        # normalization
        normalized_img = normalize(input_img.copy(), float(100), float(100))
        if normalized_img is None:
            raise ValueError("Normalization failed")

        # ROI and normalisation
        segmented_img, normim, mask = create_segmented_and_variance_images(normalized_img, block_size, 0.2)
        if segmented_img is None or normim is None or mask is None:
            raise ValueError("Segmentation failed")

        # orientations
        angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=False)
        if angles is None:
            raise ValueError("Orientation calculation failed")
            
        orientation_img = orientation.visualize_angles(segmented_img, mask, angles, W=block_size)

        # find the overall frequency of ridges
        freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)
        
        # create gabor filter and do the actual filtering
        gabor_img = gabor_filter(normim, angles, freq)
        if gabor_img is None:
            raise ValueError("Gabor filtering failed")

        # thinning
        thin_image = skeletonize(gabor_img)
        if thin_image is None:
            raise ValueError("Skeletonization failed")

        # minutias
        minutias = calculate_minutiaes(thin_image)
        if minutias is None:
            raise ValueError("Minutiae extraction failed")

        # singularities with connecting lines and ridge counting
        singularities_img, ridge_count, num_loops, num_deltas = calculate_singularities(thin_image, angles, 1, block_size, mask)
        
        # Prepare output visualization
        output_imgs = [
            input_img,
            ensure_uint8(normalized_img),
            ensure_uint8(segmented_img),
            ensure_uint8(orientation_img),
            ensure_uint8(gabor_img),
            ensure_uint8(thin_image),
            ensure_uint8(minutias),
            ensure_uint8(singularities_img)
        ]
        
        # Convert grayscale images to RGB
        for i in range(len(output_imgs)):
            if output_imgs[i] is None:
                output_imgs[i] = np.zeros_like(input_img, dtype=np.uint8)
            elif len(output_imgs[i].shape) == 2:
                output_imgs[i] = cv.cvtColor(output_imgs[i], cv.COLOR_GRAY2RGB)

        # Concatenate images
        try:
            results = np.concatenate([
                np.concatenate(output_imgs[:4], 1),
                np.concatenate(output_imgs[4:], 1)
            ])
        except Exception as e:
            raise ValueError(f"Failed to concatenate result images: {str(e)}")

        return results, ridge_count, num_loops, num_deltas

    except Exception as e:
        print(f"Pipeline error: {str(e)}")
        raise

def is_valid_bmp(filename: str) -> bool:
    """Check if the file is a valid BMP file"""
    return filename.lower().endswith('.bmp')

async def save_upload_file(upload_file: UploadFile) -> str:
    """Save uploaded file and return the file path"""
    try:
        # Validate file type
        if not is_valid_bmp(upload_file.filename):
            raise HTTPException(status_code=400, detail="Only .bmp files are accepted")

        # Create a unique filename using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{upload_file.filename}"
        file_path = os.path.join(OUTPUT_DIR, filename)
        
        # Save the file
        with open(file_path, "wb") as buffer:
            content = await upload_file.read()
            buffer.write(content)
        
        return file_path
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")

@app.post("/process-single")
async def process_single_fingerprint(file: UploadFile = File(...)):
    """Process a single fingerprint image with error handling"""
    # Validate file type
    if not is_valid_bmp(file.filename):
        raise HTTPException(status_code=400, detail="Only .bmp files are accepted")
    
    try:
        # Save uploaded file
        file_path = await save_upload_file(file)
        
        # Read and process image
        img = cv.imread(file_path, 0)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not read image file")
            
        # Validate image dimensions
        if img.size == 0 or min(img.shape) < 16:  # 16 is minimum block size
            raise HTTPException(status_code=400, detail="Image is too small or invalid")
            
        # Ensure input image is uint8
        img = ensure_uint8(img)
            
        # Process image with error handling
        try:
            results, count, loops, deltas = fingerprint_pipeline(img)
        except Exception as e:
            print(f"Pipeline error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing fingerprint: {str(e)}")
        
        # Save results
        output_filename = f"processed_{os.path.basename(file_path)}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Ensure results is valid before saving
        if results is None or results.size == 0:
            raise HTTPException(status_code=500, detail="Processing produced invalid results")
            
        cv.imwrite(output_path, results)
        
        # Prepare response
        analysis_result = {
            'filename': file.filename,
            'ridge_count': int(count) if count is not None else 0,
            'loop_text': "no loop found" if loops == 0 else f"{loops} loop found",
            'delta_text': "no delta found" if deltas == 0 else f"{deltas} delta found",
            'processed_image_url': f"/get-image/{output_filename}"
        }
        
        return JSONResponse(content=analysis_result)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        # Cleanup temporary files if needed
        try:
            if 'file_path' in locals():
                os.remove(file_path)
        except:
            pass





@app.post("/process-multiple")
async def process_multiple_fingerprints(files: List[UploadFile] = File(...)):
    """Process multiple fingerprint images with error handling"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    results = []
    processed_files = []
    
    try:
        for file in files:
            # Validate file type
            if not is_valid_bmp(file.filename):
                continue  # Skip non-BMP files
                
            try:
                # Save and process each file
                file_path = await save_upload_file(file)
                processed_files.append(file_path)
                
                # Read and process image
                img = cv.imread(file_path, 0)
                if img is None:
                    continue
                    
                # Validate image dimensions
                if img.size == 0 or min(img.shape) < 16:
                    continue
                    
                # Ensure input image is uint8
                img = ensure_uint8(img)
                
                # Process image
                processed_img, count, loops, deltas = fingerprint_pipeline(img)
                
                # Save processed image
                output_filename = f"processed_{os.path.basename(file_path)}"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                cv.imwrite(output_path, processed_img)
                
                # Add to results
                results.append({
                    'filename': file.filename,
                    'ridge_count': int(count) if count is not None else 0,
                    'loop_text': "no loop found" if loops == 0 else f"{loops} loop found",
                    'delta_text': "no delta found" if deltas == 0 else f"{deltas} delta found",
                    'processed_image_url': f"/get-image/{output_filename}"
                })
                
            except Exception as e:
                print(f"Error processing {file.filename}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
        
    finally:
        # Cleanup temporary files
        for file_path in processed_files:
            try:
                os.remove(file_path)
            except:
                pass
    
    if not results:
        raise HTTPException(status_code=400, detail="No valid BMP files were processed")
        
    # Print results in terminal
    print("\nFingerprint Analysis Results:")
    print("[")
    for result in results:
        print(f"    ({result['filename']}, no of count - {result['ridge_count']}, {result['loop_text']}, {result['delta_text']}),")
    print("]")
        
    return JSONResponse(content=results)



@app.get("/get-image/{image_name}")
async def get_image(image_name: str):
    """Retrieve a processed image"""
    image_path = os.path.join(OUTPUT_DIR, image_name)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)




@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Fingerprint Analysis API",
        "version": "1.0.0",
        "note": "This API only accepts .bmp files",
        "endpoints": {
            "/process-single": "Process a single fingerprint image (BMP only)",
            "/get-image/{image_name}": "Retrieve a processed image"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)