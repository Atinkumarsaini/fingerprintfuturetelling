#  github code ---------------------------------

# from utils import orientation
# import math
# import cv2 as cv
# import numpy as np

# def poincare_index_at(i, j, angles, tolerance):
#     """
#     compute the summation difference between the adjacent orientations such that the orientations is less then 90 degrees
#     https://books.google.pl/books?id=1Wpx25D8qOwC&lpg=PA120&ots=9wRY0Rosb7&dq=poincare%20index%20fingerprint&hl=pl&pg=PA120#v=onepage&q=poincare%20index%20fingerprint&f=false
#     :param i:
#     :param j:
#     :param angles:
#     :param tolerance:
#     :return:
#     """
#     cells = [(-1, -1), (-1, 0), (-1, 1),         # p1 p2 p3
#             (0, 1),  (1, 1),  (1, 0),            # p8    p4
#             (1, -1), (0, -1), (-1, -1)]          # p7 p6 p5

#     angles_around_index = [math.degrees(angles[i - k][j - l]) for k, l in cells]
#     index = 0
#     for k in range(0, 8):

#         # calculate the difference
#         difference = angles_around_index[k] - angles_around_index[k + 1]
#         if difference > 90:
#             difference -= 180
#         elif difference < -90:
#             difference += 180

#         index += difference

#     if 180 - tolerance <= index <= 180 + tolerance:
#         return "loop"
#     if -180 - tolerance <= index <= -180 + tolerance:
#         return "delta"
#     if 360 - tolerance <= index <= 360 + tolerance:
#         return "whorl"
#     return "none"


# def calculate_singularities(im, angles, tolerance, W, mask):
#     result = cv.cvtColor(im, cv.COLOR_GRAY2RGB)

#     # DELTA: RED, LOOP:ORAGNE, whorl:INK
#     colors = {"loop" : (0, 0, 255), "delta" : (0, 128, 255), "whorl": (255, 153, 255)}

#     for i in range(3, len(angles) - 2):             # Y
#         for j in range(3, len(angles[i]) - 2):      # x
#             # mask any singularity outside of the mask
#             mask_slice = mask[(i-2)*W:(i+3)*W, (j-2)*W:(j+3)*W]
#             mask_flag = np.sum(mask_slice)
#             if mask_flag == (W*5)**2:
#                 singularity = poincare_index_at(i, j, angles, tolerance)
#                 if singularity != "none":
#                     cv.rectangle(result, ((j+0)*W, (i+0)*W), ((j+1)*W, (i+1)*W), colors[singularity], 3)

#     return result



# if __name__ == '__main__':
#     img = cv.imread('/test_img.png', 0)
#     cv.imshow('original', img)
#     angles = orientation.calculate_angles(img, 16, smoth=True)
#     result = calculate_singularities(img, angles, 1, 16)


# ------------------------------------------------------------------






















# main train code ridge count -----------------------------------------




from utils import orientation
import math
import cv2 as cv
import numpy as np

def poincare_index_at(i, j, angles, tolerance):
    cells = [(-1, -1), (-1, 0), (-1, 1),  # p1 p2 p3
             (0, 1), (1, 1), (1, 0),      # p8    p4
             (1, -1), (0, -1), (-1, -1)]  # p7 p6 p5
    
    angles_around_index = [math.degrees(angles[i - k][j - l]) for k, l in cells]
    index = 0
    
    for k in range(0, 8):
        difference = angles_around_index[k] - angles_around_index[k + 1]
        if difference > 90:
            difference -= 180
        elif difference < -90:
            difference += 180
        index += difference
    
    if 180 - tolerance <= index <= 180 + tolerance:
        return "loop"
    if -180 - tolerance <= index <= -180 + tolerance:
        return "delta"
    if 360 - tolerance <= index <= 360 + tolerance:
        return "whorl"
    return "none"

def get_box_center(point, W):
    x, y = point
    center_x = x + W//2
    center_y = y + W//2
    return (center_x, center_y)

def find_farthest_points(loop_points, delta_points, W):
    if not loop_points or not delta_points:
        return None, None
    
    max_distance = -float('inf')
    farthest_loop = None
    farthest_delta = None
    
    for loop_point in loop_points:
        loop_center = get_box_center(loop_point, W)
        for delta_point in delta_points:
            delta_center = get_box_center(delta_point, W)
            distance = np.sqrt((loop_center[0] - delta_center[0])**2 + 
                             (loop_center[1] - delta_center[1])**2)
            if distance > max_distance:
                max_distance = distance
                farthest_loop = loop_center
                farthest_delta = delta_center
                
    return farthest_loop, farthest_delta

def find_ridge_crossings(image, start_point, end_point):
    """
    Enhanced ridge crossing detection
    """
    x1, y1 = start_point
    x2, y2 = end_point
    
    # Get line points using numpy's linspace
    t = np.linspace(0, 1, int(np.hypot(x2-x1, y2-y1) * 2))
    x_points = np.int32((1-t) * x1 + t * x2)
    y_points = np.int32((1-t) * y1 + t * y2)
    
    # Create arrays for intensity values
    intensities = []
    points = []
    
    # Get intensity profile along the line
    for x, y in zip(x_points, y_points):
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            intensities.append(image[y, x])
            points.append((x, y))
    
    intensities = np.array(intensities)
    
    # Smooth the intensity profile
    kernel_size = 3
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(intensities, kernel, mode='same')
    
    # Find zero crossings (ridge edges)
    crossings = []
    threshold = 127
    
    for i in range(1, len(smoothed)-1):
        if (smoothed[i-1] < threshold and smoothed[i] >= threshold) or \
           (smoothed[i-1] >= threshold and smoothed[i] < threshold):
            crossings.append(points[i])
    
    return crossings


# print statement of loop and delta 



# def calculate_singularities(im, angles, tolerance, W, mask):
#     result = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
#     ridge_count = 0  # Initialize ridge count
    
#     # Store singularity points
#     singularity_points = {
#         "loop": [],
#         "delta": [],
#         "whorl": []
#     }
    
#     colors = {
#         "loop": (0, 128, 255),   # Orange
#         "delta": (0, 0, 255),    # Red
#         "whorl": (255, 153, 255) # Pink
#     }
    
#     # Find singularities
#     for i in range(3, len(angles) - 2):
#         for j in range(3, len(angles[i]) - 2):
#             mask_slice = mask[(i-2)*W:(i+3)*W, (j-2)*W:(j+3)*W]
#             mask_flag = np.sum(mask_slice)
#             if mask_flag == (W*5)**2:
#                 singularity = poincare_index_at(i, j, angles, tolerance)
#                 if singularity != "none":
#                     corner_point = ((j+0)*W, (i+0)*W)
#                     singularity_points[singularity].append(corner_point)
#                     cv.rectangle(result, corner_point, 
#                                ((j+1)*W, (i+1)*W), colors[singularity], 3)
    
#     # Find farthest loop and delta points
#     loop_center, delta_center = find_farthest_points(
#         singularity_points["loop"], 
#         singularity_points["delta"],
#         W
#     )
    
#     if loop_center and delta_center:
#         # Enhance image for better ridge detection
#         enhanced = cv.equalizeHist(im)
#         _, binary = cv.threshold(enhanced, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
#         # Find ridge crossings
#         crossings = find_ridge_crossings(binary, loop_center, delta_center)
#         ridge_count = len(crossings)//2
        
#         # Draw dashed line
#         cv.line(result, loop_center, delta_center, (0, 255, 255), 1, cv.LINE_AA)
        
#         # Draw crossing points
#         for point in crossings:
#             cv.circle(result, point, 3, (0, 255, 255), -1)
    
#     # Count number of each type of singularity
#     num_loops = len(singularity_points["loop"])
#     num_deltas = len(singularity_points["delta"])
    
#     return result, ridge_count, num_loops, num_deltas






# print statement of image and count --------------


# def calculate_singularities(im, angles, tolerance, W, mask):
#     result = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
#     ridge_count = 0  # Initialize ridge count
    
#     # Store singularity points
#     singularity_points = {
#         "loop": [],
#         "delta": [],
#         "whorl": []
#     }
    
#     colors = {
#         "loop": (0, 128, 255),   # Orange
#         "delta": (0, 0, 255),    # Red
#         "whorl": (255, 153, 255) # Pink
#     }
    
#     # Find singularities
#     for i in range(3, len(angles) - 2):
#         for j in range(3, len(angles[i]) - 2):
#             mask_slice = mask[(i-2)*W:(i+3)*W, (j-2)*W:(j+3)*W]
#             mask_flag = np.sum(mask_slice)
#             if mask_flag == (W*5)**2:
#                 singularity = poincare_index_at(i, j, angles, tolerance)
#                 if singularity != "none":
#                     corner_point = ((j+0)*W, (i+0)*W)
#                     singularity_points[singularity].append(corner_point)
#                     cv.rectangle(result, corner_point, 
#                                ((j+1)*W, (i+1)*W), colors[singularity], 3)
    
#     # Find farthest loop and delta points
#     loop_center, delta_center = find_farthest_points(
#         singularity_points["loop"], 
#         singularity_points["delta"],
#         W
#     )
    
#     if loop_center and delta_center:
#         # Enhance image for better ridge detection
#         enhanced = cv.equalizeHist(im)
#         _, binary = cv.threshold(enhanced, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
#         # Find ridge crossings
#         crossings = find_ridge_crossings(binary, loop_center, delta_center)
#         ridge_count = len(crossings)//2
        
#         # Draw dashed line
#         cv.line(result, loop_center, delta_center, (0, 255, 255), 1, cv.LINE_AA)
        
#         # Draw crossing points
#         for point in crossings:
#             cv.circle(result, point, 3, (0, 255, 255), -1)
    
#     return result, ridge_count





#  main function --------------



# def calculate_singularities(im, angles, tolerance, W, mask):
#     result = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
    
#     # Store singularity points
#     singularity_points = {
#         "loop": [],
#         "delta": [],
#         "whorl": []
#     }
    
#     colors = {
#         "loop": (0, 128, 255),   # Orange
#         "delta": (0, 0, 255),    # Red
#         "whorl": (255, 153, 255) # Pink
#     }
    
#     # Find singularities
#     for i in range(3, len(angles) - 2):
#         for j in range(3, len(angles[i]) - 2):
#             mask_slice = mask[(i-2)*W:(i+3)*W, (j-2)*W:(j+3)*W]
#             mask_flag = np.sum(mask_slice)
#             if mask_flag == (W*5)**2:
#                 singularity = poincare_index_at(i, j, angles, tolerance)
#                 if singularity != "none":
#                     corner_point = ((j+0)*W, (i+0)*W)
#                     singularity_points[singularity].append(corner_point)
#                     cv.rectangle(result, corner_point, 
#                                ((j+1)*W, (i+1)*W), colors[singularity], 3)
    
#     # Find farthest loop and delta points
#     loop_center, delta_center = find_farthest_points(
#         singularity_points["loop"], 
#         singularity_points["delta"],
#         W
#     )
    
#     if loop_center and delta_center:
#         # Enhance image for better ridge detection
#         enhanced = cv.equalizeHist(im)
#         _, binary = cv.threshold(enhanced, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
#         # Find ridge crossings
#         crossings = find_ridge_crossings(binary, loop_center, delta_center)
        
#         # Draw dashed line
#         cv.line(result, loop_center, delta_center, (0, 255, 255), 1, cv.LINE_AA)
        
#         # Draw crossing points
#         for point in crossings:
#             cv.circle(result, point, 3, (0, 255, 255), -1)
        
#         # Print ridge count
#         print(f"Number of ridge crossings: {len(crossings)//2}")
    
#     return result







# Example usage:
# img = cv.imread('fingerprint.png', cv.IMREAD_GRAYSCALE)
# angles = orientation.calculate_angles(img)
# mask = np.ones_like(img)
# result = calculate_singularities(img, angles, tolerance=5, W=16, mask=mask)
# cv.imshow('Result', result)
# cv.waitKey(0)
# cv.destroyAllWindows()




# --------------------------------------------------







