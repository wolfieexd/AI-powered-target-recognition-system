"""
Image Preprocessing Utilities for Enhanced Weapon Detection
Provides adaptive enhancement for low-light, low-contrast images
"""

import cv2
import numpy as np


class ImagePreprocessor:
    """Advanced preprocessing for improved detection in challenging conditions"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.auto_enhance = self.config.get('auto_enhance', True)
        self.clahe_enabled = self.config.get('clahe_enabled', True)
        self.gamma_correction = self.config.get('gamma_correction', True)
        
    def preprocess(self, image, enhance_mode='auto'):
        """
        Apply preprocessing pipeline to enhance detection quality
        
        Args:
            image: Input BGR image
            enhance_mode: 'auto', 'low_light', 'high_contrast', 'none'
        
        Returns:
            Enhanced BGR image
        """
        if enhance_mode == 'none' or not self.auto_enhance:
            return image
        
        # Auto-detect conditions if needed
        if enhance_mode == 'auto':
            enhance_mode = self._detect_image_conditions(image)
        
        enhanced = image.copy()
        
        if enhance_mode == 'low_light':
            enhanced = self._enhance_low_light(enhanced)
        elif enhance_mode == 'high_contrast':
            enhanced = self._enhance_contrast(enhanced)
        
        return enhanced
    
    def _detect_image_conditions(self, image):
        """Automatically detect if image needs enhancement"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness (mean intensity)
        mean_brightness = np.mean(gray)
        
        # Calculate contrast (std deviation)
        contrast = np.std(gray)
        
        # Decision thresholds
        LOW_BRIGHTNESS_THRESHOLD = 80  # 0-255 scale
        LOW_CONTRAST_THRESHOLD = 40
        
        if mean_brightness < LOW_BRIGHTNESS_THRESHOLD:
            return 'low_light'
        elif contrast < LOW_CONTRAST_THRESHOLD:
            return 'high_contrast'
        else:
            return 'none'
    
    def _enhance_low_light(self, image):
        """Enhance images taken in low-light conditions"""
        # Method 1: Gamma correction for brightness
        if self.gamma_correction:
            image = self._apply_gamma_correction(image, gamma=1.5)
        
        # Method 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if self.clahe_enabled:
            image = self._apply_clahe(image)
        
        return image
    
    def _enhance_contrast(self, image):
        """Enhance low-contrast images"""
        if self.clahe_enabled:
            return self._apply_clahe(image, clip_limit=3.0)
        else:
            # Simple contrast stretching
            return self._contrast_stretching(image)
    
    def _apply_gamma_correction(self, image, gamma=1.5):
        """
        Apply gamma correction to adjust brightness
        gamma < 1: darker, gamma > 1: brighter
        """
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in range(256)]).astype("uint8")
        
        # Apply gamma correction
        return cv2.LUT(image, table)
    
    def _apply_clahe(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Works on LAB color space to preserve colors
        """
        # Convert BGR to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel only
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_clahe = clahe.apply(l)
        
        # Merge channels
        lab_clahe = cv2.merge([l_clahe, a, b])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _contrast_stretching(self, image):
        """Simple min-max contrast stretching"""
        # Convert to float
        img_float = image.astype(np.float32)
        
        # Stretch each channel
        for i in range(3):
            channel = img_float[:, :, i]
            min_val = np.percentile(channel, 2)  # 2nd percentile
            max_val = np.percentile(channel, 98)  # 98th percentile
            
            # Stretch
            channel = (channel - min_val) / (max_val - min_val) * 255
            channel = np.clip(channel, 0, 255)
            img_float[:, :, i] = channel
        
        return img_float.astype(np.uint8)
    
    def apply_multi_scale_pyramid(self, image, scales=[0.5, 1.0, 1.5]):
        """
        Generate image pyramid at multiple scales for multi-scale detection
        
        Args:
            image: Input image
            scales: List of scale factors
        
        Returns:
            List of (scaled_image, scale_factor) tuples
        """
        pyramid = []
        h, w = image.shape[:2]
        
        for scale in scales:
            if scale == 1.0:
                pyramid.append((image, scale))
            else:
                new_h, new_w = int(h * scale), int(w * scale)
                scaled = cv2.resize(image, (new_w, new_h), 
                                   interpolation=cv2.INTER_LINEAR if scale > 1 
                                   else cv2.INTER_AREA)
                pyramid.append((scaled, scale))
        
        return pyramid


class TiledInference:
    """Sliding window tiled inference for better small object detection"""
    
    def __init__(self, tile_size=640, overlap=0.2):
        """
        Args:
            tile_size: Size of each tile (square)
            overlap: Overlap ratio between tiles (0.0 to 0.5)
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = int(tile_size * (1 - overlap))
    
    def split_image_to_tiles(self, image):
        """
        Split large image into overlapping tiles
        
        Returns:
            List of (tile, x_offset, y_offset) tuples
        """
        h, w = image.shape[:2]
        tiles = []
        
        # If image is smaller than tile size, return the whole image
        if h < self.tile_size or w < self.tile_size:
            # Pad image to tile_size
            pad_h = max(0, self.tile_size - h)
            pad_w = max(0, self.tile_size - w)
            padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, 
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))
            return [(padded, 0, 0)]
        
        # Generate tile positions
        y_positions = list(range(0, h - self.tile_size + 1, self.stride))
        x_positions = list(range(0, w - self.tile_size + 1, self.stride))
        
        # Ensure at least one tile
        if not y_positions:
            y_positions = [0]
        if not x_positions:
            x_positions = [0]
        
        # Add final tiles to cover edges if needed
        if len(y_positions) > 0 and y_positions[-1] + self.tile_size < h:
            y_positions.append(max(0, h - self.tile_size))
        if len(x_positions) > 0 and x_positions[-1] + self.tile_size < w:
            x_positions.append(max(0, w - self.tile_size))
        
        # Extract tiles
        for y in y_positions:
            for x in x_positions:
                tile = image[y:y+self.tile_size, x:x+self.tile_size]
                # Ensure tile is exactly tile_size x tile_size
                if tile.shape[0] != self.tile_size or tile.shape[1] != self.tile_size:
                    # Pad if needed
                    pad_h = max(0, self.tile_size - tile.shape[0])
                    pad_w = max(0, self.tile_size - tile.shape[1])
                    tile = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w,
                                            cv2.BORDER_CONSTANT, value=(0, 0, 0))
                tiles.append((tile, x, y))
        
        return tiles
    
    def merge_tile_detections(self, all_detections, image_shape, iou_threshold=0.5):
        """
        Merge detections from all tiles using NMS
        
        Args:
            all_detections: List of detection dicts with 'xyxy', 'conf', 'cls', 'class_name'
            image_shape: Original image (h, w)
            iou_threshold: IoU threshold for NMS
        
        Returns:
            Merged detection results (same format as input)
        """
        if not all_detections:
            return []
        
        # Extract boxes, scores, and classes
        boxes = []
        scores = []
        classes = []
        class_names = []
        
        for det in all_detections:
            boxes.append(det['xyxy'])
            scores.append(det['conf'])
            classes.append(det['cls'])
            class_names.append(det['class_name'])
        
        # Apply NMS
        keep_indices = self._apply_nms_indices(boxes, scores, classes, iou_threshold)
        
        # Return only kept detections
        merged = []
        for idx in keep_indices:
            merged.append({
                'xyxy': boxes[idx],
                'conf': scores[idx],
                'cls': classes[idx],
                'class_name': class_names[idx]
            })
        
        return merged
    
    def _apply_nms_indices(self, boxes, scores, classes, iou_threshold):
        """
        Apply NMS and return indices to keep
        
        Returns:
            List of indices to keep
        """
        if len(boxes) == 0:
            return []
        
        import numpy as np
        
        boxes_array = np.array(boxes, dtype=np.float32)
        scores_array = np.array(scores, dtype=np.float32)
        classes_array = np.array(classes, dtype=np.int32)
        
        # Sort by score (descending)
        order = scores_array.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(int(i))
            
            if len(order) == 1:
                break
            
            # Compute IoU with remaining boxes
            ious = []
            for j in order[1:]:
                iou = self._compute_iou(boxes_array[i], boxes_array[j])
                ious.append(iou)
            
            ious = np.array(ious)
            
            # Keep boxes with IoU below threshold OR different class
            mask = np.logical_or(
                ious < iou_threshold,
                classes_array[order[1:]] != classes_array[i]
            )
            
            order = order[1:][mask]
        
        return keep
    
    def _compute_iou(self, box1, box2):
        """Compute IoU between two boxes [x1, y1, x2, y2]"""
        import numpy as np
        
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        
        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0
        
        return inter_area / union_area
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        classes = np.array(classes)
        
        # Sort by score
        indices = np.argsort(scores)[::-1]
        
        keep = []
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            ious = self._calculate_iou_batch(boxes[current], boxes[indices[1:]])
            
            # Keep boxes with IoU below threshold or different class
            mask = (ious < iou_threshold) | (classes[indices[1:]] != classes[current])
            indices = indices[1:][mask]
        
        # Return kept detections
        final = []
        for idx in keep:
            final.append({
                'box': boxes[idx].tolist(),
                'confidence': float(scores[idx]),
                'class': classes[idx]
            })
        
        return final
    
    def _calculate_iou_batch(self, box, boxes):
        """Calculate IoU between one box and multiple boxes"""
        # box: [x1, y1, x2, y2]
        # boxes: Nx4 array
        
        # Intersection
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Union
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - intersection
        
        # IoU
        iou = intersection / (union + 1e-6)
        
        return iou


class TestTimeAugmentation:
    """Test-time augmentation for robust detection"""
    
    def __init__(self, use_flips=True, use_scales=True):
        self.use_flips = use_flips
        self.use_scales = use_scales
        self.augmentations = self._build_augmentations()
    
    def _build_augmentations(self):
        """Build list of augmentation transforms"""
        augs = [
            {'name': 'original', 'transform': None, 'inverse': None}
        ]
        
        if self.use_flips:
            # Horizontal flip
            augs.append({
                'name': 'hflip',
                'transform': lambda img: cv2.flip(img, 1),
                'inverse': lambda img, boxes: self._inverse_hflip(img, boxes)
            })
        
        if self.use_scales:
            # Scale variations
            for scale in [0.8, 1.2]:
                augs.append({
                    'name': f'scale_{scale}',
                    'transform': lambda img, s=scale: self._scale_image(img, s),
                    'inverse': lambda img, boxes, s=scale: self._inverse_scale(img, boxes, s)
                })
        
        return augs
    
    def _scale_image(self, image, scale):
        """Scale image"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    def _inverse_hflip(self, original_image, boxes):
        """Invert horizontal flip on bounding boxes"""
        w = original_image.shape[1]
        inverted_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box['box']
            box_copy = box.copy()
            box_copy['box'] = [w - x2, y1, w - x1, y2]
            inverted_boxes.append(box_copy)
        return inverted_boxes
    
    def _inverse_scale(self, original_image, boxes, scale):
        """Invert scaling on bounding boxes"""
        inv_scale = 1.0 / scale
        inverted_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box['box']
            box_copy = box.copy()
            box_copy['box'] = [
                x1 * inv_scale, y1 * inv_scale,
                x2 * inv_scale, y2 * inv_scale
            ]
            inverted_boxes.append(box_copy)
        return inverted_boxes
    
    def apply(self, image, detection_fn):
        """
        Apply TTA and merge results
        
        Args:
            image: Input image
            detection_fn: Function that takes image and returns detections
        
        Returns:
            Merged detections from all augmentations
        """
        all_detections = []
        
        for aug in self.augmentations:
            # Apply augmentation
            if aug['transform'] is None:
                aug_image = image
            else:
                aug_image = aug['transform'](image)
            
            # Run detection
            detections = detection_fn(aug_image)
            
            # Inverse transform boxes back to original coordinates
            if aug['inverse'] is not None and detections:
                detections = aug['inverse'](image, detections)
            
            all_detections.extend(detections)
        
        # Merge using NMS
        if all_detections:
            tiled_inf = TiledInference()
            boxes = [d['box'] for d in all_detections]
            scores = [d['confidence'] for d in all_detections]
            classes = [d['class'] for d in all_detections]
            
            merged = tiled_inf._apply_nms(boxes, scores, classes, iou_threshold=0.5)
            return merged
        
        return []
