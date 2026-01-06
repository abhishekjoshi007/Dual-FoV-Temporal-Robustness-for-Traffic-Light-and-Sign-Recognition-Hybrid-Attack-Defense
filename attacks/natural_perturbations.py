import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import torch
from scipy.ndimage import gaussian_filter
import logging

logger = logging.getLogger(__name__)


class NaturalPerturbationSuite:
    
    def __init__(
        self,
        perturbation_types: List[str] = None,
        intensity_range: Tuple[float, float] = (0.3, 0.7),
        temporal_persistence: Tuple[int, int] = (60, 150),
        object_aware: bool = True,
    ):
        if perturbation_types is None:
            perturbation_types = [
                'rain', 'fog', 'sun_glare', 'dirt', 'motion_blur',
                'headlight', 'lens_flare', 'droplets', 'snow'
            ]
        
        self.perturbation_types = perturbation_types
        self.intensity_range = intensity_range
        self.temporal_persistence = temporal_persistence
        self.object_aware = object_aware
        
        self.perturbation_functions = {
            'rain': self.add_rain,
            'fog': self.add_fog,
            'sun_glare': self.add_sun_glare,
            'dirt': self.add_dirt,
            'motion_blur': self.add_motion_blur,
            'headlight': self.add_headlight_glare,
            'lens_flare': self.add_lens_flare,
            'droplets': self.add_water_droplets,
            'snow': self.add_snow,
        }
        
        logger.info(f"NaturalPerturbationSuite initialized with {len(perturbation_types)} perturbation types")
    
    def apply_perturbation(
        self,
        image: np.ndarray,
        perturbation_type: str,
        intensity: Optional[float] = None,
        bbox: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if perturbation_type not in self.perturbation_functions:
            logger.warning(f"Unknown perturbation type: {perturbation_type}")
            return image
        
        if intensity is None:
            intensity = np.random.uniform(*self.intensity_range)
        
        perturb_func = self.perturbation_functions[perturbation_type]
        
        if self.object_aware and bbox is not None:
            return self._apply_object_aware(image, perturb_func, intensity, bbox)
        else:
            return perturb_func(image, intensity)
    
    def _apply_object_aware(
        self,
        image: np.ndarray,
        perturb_func,
        intensity: float,
        bbox: np.ndarray,
    ) -> np.ndarray:
        perturbed = image.copy()
        
        x_min, y_min, x_max, y_max = map(int, bbox)
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image.shape[1], x_max)
        y_max = min(image.shape[0], y_max)
        
        if x_max <= x_min or y_max <= y_min:
            return image
        
        roi = image[y_min:y_max, x_min:x_max].copy()
        perturbed_roi = perturb_func(roi, intensity)
        perturbed[y_min:y_max, x_min:x_max] = perturbed_roi
        
        return perturbed
    
    def add_rain(self, image: np.ndarray, intensity: float = 0.5) -> np.ndarray:

        H, W = image.shape[:2]

        # Droplet coverage 15-35% of object area
        # Map intensity [0,1] to coverage [15%, 35%]
        coverage_percent = 0.15 + (0.35 - 0.15) * intensity
        num_drops = int(H * W * coverage_percent * 0.001)  # Adjust multiplier for visible drops

        rain_layer = image.copy()

        for _ in range(num_drops):
            x = np.random.randint(0, W)
            y = np.random.randint(0, H)
            # Streak length 50-120 pixels 
            length = np.random.randint(50, 120)
            thickness = np.random.choice([1, 2])

            if y + length < H:
                brightness = np.random.randint(180, 255)
                color = (brightness, brightness, brightness)
                cv2.line(rain_layer, (x, y), (x + np.random.randint(-2, 3), y + length), color, thickness)

        rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)

        result = cv2.addWeighted(image, 1 - intensity * 0.3, rain_layer, intensity * 0.3, 0)
        return result.astype(np.uint8)
    
    def add_fog(self, image: np.ndarray, intensity: float = 0.5, beta: float = 0.02) -> np.ndarray:

        H, W = image.shape[:2]

        # Implement proper atmospheric scattering 
        # Simulate distance field: intensity maps to distance in meters (0-100m range)
        distance = intensity * 100  # Scale intensity to distance
        attenuation = np.exp(-beta * distance)

        # Atmospheric light (gray fog with slight color variation)
        atmospheric_light = 200.0
        atmospheric_light += np.random.uniform(-5, 5)  

        # Apply Koschmieder's scattering formula
        fog_layer = image.astype(np.float32) * attenuation + atmospheric_light * (1 - attenuation)

        return np.clip(fog_layer, 0, 255).astype(np.uint8)
    
    def add_sun_glare(self, image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        H, W = image.shape[:2]
        
        center_x = np.random.randint(int(W * 0.3), int(W * 0.7))
        center_y = np.random.randint(int(H * 0.2), int(H * 0.5))
        
        radius = int(min(H, W) * 0.3 * intensity)
        
        glare_mask = np.zeros((H, W), dtype=np.float32)
        cv2.circle(glare_mask, (center_x, center_y), radius, 1.0, -1)
        
        glare_mask = gaussian_filter(glare_mask, sigma=radius * 0.5)
        glare_mask = glare_mask / (glare_mask.max() + 1e-10)
        
        glare_intensity = 100 * intensity
        glare_mask = glare_mask[:, :, np.newaxis] * glare_intensity
        
        result = image.astype(np.float32) + glare_mask
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def add_dirt(self, image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        H, W = image.shape[:2]
        
        num_spots = int(30 * intensity)
        dirt_layer = np.zeros((H, W), dtype=np.uint8)
        
        for _ in range(num_spots):
            x = np.random.randint(0, W)
            y = np.random.randint(0, H)
            size = np.random.randint(5, 20)
            
            cv2.circle(dirt_layer, (x, y), size, 255, -1)
        
        dirt_layer = cv2.GaussianBlur(dirt_layer, (15, 15), 0)
        
        dirt_color = np.random.randint(50, 100, 3)
        dirt_overlay = np.zeros_like(image)
        for c in range(3):
            dirt_overlay[:, :, c] = dirt_color[c]
        
        dirt_mask = dirt_layer[:, :, np.newaxis] / 255.0 * intensity * 0.6
        
        result = image * (1 - dirt_mask) + dirt_overlay * dirt_mask
        return result.astype(np.uint8)
    
    def add_motion_blur(self, image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        kernel_size = int(3 + intensity * 10)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = 1.0
        kernel = kernel / kernel_size
        
        angle = np.random.uniform(0, 360)
        M = cv2.getRotationMatrix2D((kernel_size // 2, kernel_size // 2), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        
        result = cv2.filter2D(image, -1, kernel)
        return result.astype(np.uint8)
    
    def add_headlight_glare(self, image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        H, W = image.shape[:2]
        
        num_lights = np.random.randint(1, 3)
        result = image.copy().astype(np.float32)
        
        for _ in range(num_lights):
            x = np.random.randint(int(W * 0.2), int(W * 0.8))
            y = np.random.randint(int(H * 0.5), H)
            
            radius = int(min(H, W) * 0.1 * intensity)
            
            glare_mask = np.zeros((H, W), dtype=np.float32)
            cv2.circle(glare_mask, (x, y), radius, 1.0, -1)
            glare_mask = gaussian_filter(glare_mask, sigma=radius * 0.4)
            glare_mask = glare_mask / (glare_mask.max() + 1e-10)
            
            glare_intensity = 150 * intensity
            glare_mask = glare_mask[:, :, np.newaxis] * glare_intensity
            
            result += glare_mask
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def add_lens_flare(self, image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        H, W = image.shape[:2]
        
        center_x = np.random.randint(int(W * 0.3), int(W * 0.7))
        center_y = np.random.randint(0, int(H * 0.4))
        
        flare_mask = np.zeros((H, W), dtype=np.float32)
        
        num_circles = 5
        for i in range(num_circles):
            offset_x = int((center_x - W // 2) * (i / num_circles))
            offset_y = int((center_y - H // 2) * (i / num_circles))
            
            x = W // 2 + offset_x
            y = H // 2 + offset_y
            
            radius = int(20 + i * 10)
            cv2.circle(flare_mask, (x, y), radius, 1.0 / (i + 1), -1)
        
        flare_mask = gaussian_filter(flare_mask, sigma=20)
        flare_mask = flare_mask / (flare_mask.max() + 1e-10)
        
        flare_intensity = 80 * intensity
        flare_mask = flare_mask[:, :, np.newaxis] * flare_intensity
        
        result = image.astype(np.float32) + flare_mask
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def add_water_droplets(self, image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        H, W = image.shape[:2]
        
        num_droplets = int(50 * intensity)
        result = image.copy()
        
        for _ in range(num_droplets):
            x = np.random.randint(0, W)
            y = np.random.randint(0, H)
            size = np.random.randint(3, 8)
            
            mask = np.zeros((H, W), dtype=np.float32)
            cv2.circle(mask, (x, y), size, 1.0, -1)
            mask = gaussian_filter(mask, sigma=size * 0.3)
            
            blur_strength = size * 2 + 1
            if blur_strength % 2 == 0:
                blur_strength += 1
            blurred = cv2.GaussianBlur(result, (blur_strength, blur_strength), 0)
            
            mask_3d = mask[:, :, np.newaxis]
            result = (result * (1 - mask_3d) + blurred * mask_3d).astype(np.uint8)
        
        return result
    
    def add_snow(self, image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        H, W = image.shape[:2]
        
        num_flakes = int(H * W * 0.001 * intensity)
        snow_layer = image.copy()
        
        for _ in range(num_flakes):
            x = np.random.randint(0, W)
            y = np.random.randint(0, H)
            size = np.random.randint(1, 4)
            
            brightness = np.random.randint(200, 255)
            cv2.circle(snow_layer, (x, y), size, (brightness, brightness, brightness), -1)
        
        snow_layer = cv2.GaussianBlur(snow_layer, (3, 3), 0)
        
        result = cv2.addWeighted(image, 1 - intensity * 0.3, snow_layer, intensity * 0.3, 0)
        return result.astype(np.uint8)
    
    def apply_compound_perturbation(
        self,
        image: np.ndarray,
        perturbations: List[str],
        intensities: Optional[List[float]] = None,
        bbox: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        result = image.copy()
        
        if intensities is None:
            intensities = [np.random.uniform(*self.intensity_range) for _ in perturbations]
        
        for perturbation, intensity in zip(perturbations, intensities):
            result = self.apply_perturbation(result, perturbation, intensity, bbox)
        
        return result
    
    def apply_sequence_perturbation(
        self,
        sequence: np.ndarray,
        perturbation_type: str,
        intensity: Optional[float] = None,
        bboxes: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        T, H, W, C = sequence.shape
        
        if intensity is None:
            intensity = np.random.uniform(*self.intensity_range)
        
        persistence_frames = np.random.randint(*self.temporal_persistence)
        persistence_frames = min(persistence_frames, T)
        
        start_frame = np.random.randi