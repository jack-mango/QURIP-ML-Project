# Algorithms

# `ImageProcessor.find_centroids()`

## Purpose

Finding the positions of tweezers in a possibly noisy image.

## Overview

1. Get the first 10% of images from each loop to search for tweezer positions. These are more likely to contain bright tweezers than images from other parts of the loop as. This will make it easier to find tweezer positions. Then convert these images to the correct format to work with openCV.
    
    ```python
    frac_stack = self.fractional_stack(stack, 0.1)
    frac_stack = self.to_uint8_img(frac_stack)
    ```
    
2. Fit a bimodal Gaussian distribution to the image data. One mode will correspond to dark pixels, the other to bright pixels. Set the threshold used in adaptive thresholding to be four times the dark distribution’s standard deviation.
    
    ```python
    model = GaussianMixture(frac_stack.flatten())
    params, r_sq = model.fit()
    dark_params, bright_params = params
    thresh = -4 * dark_params[1]
    ```
    
3. Iterate over images from the first 10% of each loop. Apply bilateral filtering to remove noise followed by adaptive thresholding. Pixels above the average over a small region, plus some bias will be set to 255, those below to zero. Then perform connected component analysis. This will identify groups of pixels that survived the thresholding. Filter out any components that contain only one pixel as these are likely noise and the first connected component, as this is the background of the image. Finally add the filtered connected components to an array containing any previously found ones.
    
    ```python
    for img in frac_stack:
                img = cv2.adaptiveThreshold(
                    img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, thresh)
                n_sites, labeled_img, stats, centroids = cv2.connectedComponentsWithStats(img)
                for stat in stats:
                    if stat[-1] <= 1:
                        img[stat[1]:stat[1] + stat[3], stat[0]:stat[0] +
                            stat[2]] = np.zeros((stat[3], stat[2]))
                final = np.maximum(img, final)
            n_sites, labeled_img, stats, centroids = cv2.connectedComponentsWithStats(
                final)
    ```
    
4. Perform connected component analysis once more. This time we’ll get the centroids of each connected component. We skip the component corresponding to the background once more and return the `n_tweezers` largest connected components. 
    
    ```python
    n_sites, labeled_img, stats, centroids = cv2.connectedComponentsWithStats(
                final)
            centroids = centroids[np.argsort(stats[:, -1])]
            return centroids[-2:-self.n_tweezers - 2:-1, ::-1]
    ```
    

## Further Resources

[Bilateral Filtering](https://people.csail.mit.edu/sparis/publi/2009/fntcgv/Paris_09_Bilateral_filtering.pdf)

[Adaptive Thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)

[Connected Component Analysis](https://www.mathworks.com/help/images/label-and-measure-objects-in-a-binary-image.html)

# `ImageProcessor.closest_pair_distance()`

## Purpose

Finding the smallest horizontal or vertical distance that two tweezer’s could be. This isn’t the true nearest neighbor distance, instead it’s the projection of that distance onto the x or y axis.

This approach is $O(n \log (n))$ whereas the brute force approach scales as $O(n^2)$. One can imagine with a large number of tweezers that this could become computationally expensive.

## Overview

1. Divide the image in half vertically and recursively search each for the closest pair of positions in each half until a region contains three or fewer points. At the bottom of the recursive call stack check each combination of remaining positions to find the closest two.
    
    ```python
    if len(points) <= 3:
                return self.closest_pair_bf(points)
            points = points[points[:, 0].argsort(
            )]
            mid_point = len(points) // 2
            left_points = points[:mid_point]
            right_points = points[mid_point:]
            min_dist_left, closest_left = self.closest_pair_distance(left_points)
            min_dist_right, closest_right = self.closest_pair_distance(
                right_points)
            if min_dist_left < min_dist_right:
                min_distance = min_dist_left
                closest_pair = closest_left
            else:
                min_distance = min_dist_right
                closest_pair = closest_right
    ```
    
2. Search the regions between each half for points that could be closer than the closest already found. 
    
    ```python
    strip_points = points[np.abs(
                points[:, 0] - points[mid_point, 0]) < min_distance]
            strip_points = strip_points[strip_points[:, 1].argsort()]
            min_dist_strip = min_distance
            closest_strip = (None, None)
            for i in range(len(strip_points)):
                j = i + 1
                while j < len(strip_points) and strip_points[j, 1] - strip_points[i, 1] < min_dist_strip:
                    dist = np.linalg.norm(strip_points[i] - strip_points[j])
                    if dist < min_dist_strip:
                        min_dist_strip = dist
                        closest_strip = (strip_points[i], strip_points[j])
                    j += 1
    ```
    
3. Finally compare the closest distance in the halved regions with those in the boundaries of each region to find the two closest points.
    
    ```python
    if min_dist_strip < min_distance:
                return min_dist_strip, closest_strip
            else:
                return min_distance, closest_pair
    ```
    

# `ImageProcessor.position_tile_sort()`

## Purpose

Sorting the positions of tweezers in a sensical way. Tweezer positions won’t always be a perfect rectangular shape due to the tweezer finding algorithm, however it makes sense to us that the top left most tweezer should probably correspond to the first in a list of positions.

Instead of just using the position coordinates to sort, use which square tile the tweezer falls in to determine the position.

# `ImageProcessor.transform_positions()`

## Purpose

In order to label the crops for tweezers for blue imaging analysis we need the crops taken from the pvcam corresponding to them. Crops made for images are ordered the same as tweezer positions they correspond to. It becomes much easier to match labels generated from pvcam images to those in from the nuvu camera if we can maintain the order of positions. This is easy if we can find the mapping of positions from pvcam images to nuvu camera images. Thus the purpose of this algorithm is to find that mapping and apply it to find the matching nuvu camera tweezer positions.

## Overview

****************Note:**************** usually the source stack is the pvcam stack and the target stack is the nuvu camera stack.

1. Find tweezer positions from the source stack.
    
    ```python
    src_positions = self.find_tweezer_positions(src_stack)
    ```
    
2. Create instances of objects that will be used to find keypoints and descriptors, then match them.
    
    ```python
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    orb = cv2.ORB_create(edgeThreshold=10)
    ```
    
3. Randomly choose ten images from the source and target stacks. Iterating over them, apply a bilateral filter to remove noise, then find keypoints and descriptors in both images. Keypoints are easier to find in individual images since not every lattice site is filled. The lattice sites that are filled then become distinguishable features that are detected as keypoints. After finding keypoints a matching algorithm is applied to find which keypoints in each image correspond to each other based on their descriptors.
    
    
    Finally make two lists of the points in each image that correspond to each other. Note that this will sometimes give faulty matches, however only a significant fraction have to be correct to calculate the affine transform. 
    
    ```python
    for i in np.random.choice(src_stack.shape[0], 10, replace=False):
                src_img = self.to_uint8_img(src_stack[i])
                target_img = self.to_uint8_img(target_stack[i])
                src_img = cv2.bilateralFilter(src_img, 5, 25, 25)
                target_img = cv2.bilateralFilter(target_img, 5, 25, 25)
                src_kp, src_desc = orb.detectAndCompute(src_img.T, None)
                target_kp, target_desc = orb.detectAndCompute(target_img.T, None)
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(src_desc, target_desc)
                src_pts.extend([src_kp[m.queryIdx].pt for m in matches])
                target_pts.extend([target_kp[m.trainIdx].pt for m in matches])
    ```
    
4. Calculate the affine transform matrix based on point mappings. The affine transform will be that which is most consistent with the point mappings, so it’s okay if some of them are wrong or a little bit off. Finally transform the source positions to get the target potisions.
    
    ```python
    affine_mat, _ = cv2.estimateAffine2D(np.array(src_pts), np.array(target_pts))
    target_positions = np.reshape(cv2.transform(np.reshape(src_positions, (-1, 1, 2)), affine_mat), (-1, 2))
    self.positions = target_positions
    self.find_nn_dist()
    ```
    

## Further Resources

[Bilateral Filtering](https://people.csail.mit.edu/sparis/publi/2009/fntcgv/Paris_09_Bilateral_filtering.pdf)

[ORB](https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html) for feature identification

[Brute force feature matching](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)

[Estimating affine transform](https://docs.opencv.org/3.4/d3/d63/classcv_1_1Mat.html#details) 

# `Labeler.slicer()`

## Purpose

When labeling continuously collected green images, those that are between two other images that have been labeled bright should also be labeled bright, and the same for dark. Those that are in between a dark and a bright image we can’t label for certain. This algorithm finds images that are between two that have been labeled with the same thing and gives them the same label. It also marks images as unlabeled if they’re between two images that cross opposite thresholds.

## Overview

1. Start with the “head” and “tail” of a segment of images at the start of a loop. Assume that an image starts off as bright.
    
    ```python
    labels = np.empty(arr.size)
    head = tail = 0
    bright = True
    ```
    
2. Iterate over each image in the loop. If an image is brighter than a given threshold, then advance the “head” of the segment of bright images to the current index. If we find an image that crosses the dark threshold and the last image that crossed a threshold was bright, then we label the segment starting from the tail index to the head index as bright, and everything between the head index and the current index as unknown, since it falls between two thresholds. We also note that the last threshold crossing was for dark images. We repeat the same process but using the opposite labels if we transition from a dark to a bright image.
    
    ```python
    for i, val in enumerate(arr):
    	if val >= upper_thresh and bright:
    		head = i + 1
    	elif val >= upper_thresh and not bright:
    		labels[head:i] = np.full(i - head, np.NaN)
    		labels[tail:head] = np.zeros(head - tail)
    		tail = i
    		head = i
    		bright = True
    elif val <= lower_thresh and not bright:
        head = i + 1
    elif val <= lower_thresh and bright:
        labels[head:i] = np.full(i - head, np.NaN)
        labels[tail:head] = np.ones(head - tail)
        head = i
        tail = i
        bright = False
    ```
    
3. At the end of a loop, if the image at the head index was dark, then we say everything up to the end was also dark. If instead the head index was bright, we label everything between the head and tail as bright and everything after as unknown.
    
    ```python
    if bright:
        labels[tail:head] = np.ones(head - tail)
        labels[head:] = np.full(labels.size - head, np.NaN)
    else:
        labels[tail:] = np.zeros(labels.size - tail)
    ```
    

# `Labeler.find_thresholds()` and `Labeler.bright_dark_fit()`

## Purpose

We need bright and dark thresholds in order to label images using the `slicer` method. We want the thresholds to be quite high for bright images and quite low for dark images so we don’t accidentally label either incorrectly. One way we can do this is by finding the probability distributions for bright and dark images, and finding the values for each distribution that ensure nearly all belonging to that distribution values are higher or lower. That way we can be sure that any images with values higher (or lower) are extremely unlikely to belong to the lower distribution.

## Overview

1. `bright_dark_fits()` — We start off iterating over each tweezer and fitting a bimodal gaussian distribution to each using a Gaussian mixture model. This gives us the mean and standard deviation for each mode for each tweezer. 
    
    ```python
    fits = np.empty((self.n_tweezers, 2, 3))
    r_sq = np.empty(self.n_tweezers)
    for i in range(self.n_tweezers):
        model = GaussianMixture(self.img_vals[i])
        fits[i], r_sq[i] = model.fit()
    self.fits = fits
    return self.fits, r_sq
    ```
    
2. Iterate over each tweezer’s fits. We say that an image that is `z` standard deviations below the mean of the distribution with the greater mean is dark, setting the lower threshold to that value. We set the upper threshold to be the same number of standard deviations above the mean of the distribution with the lower mean.
    
    ```python
    for i, fit in enumerate(fits):
        lower_thresh = fit[1, 0] - fit[1, 1] * z
        upper_thresh = fit[0, 0] + fit[0, 1] * z
        thresholds[i] = np.array([lower_thresh, upper_thresh])
    ```