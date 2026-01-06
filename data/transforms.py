import torch
import numpy as np
import cv2
from typing import Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SequenceTransform:
    
    def __init__(self, img_size: Tuple[int, int] = (640, 640), augment: bool = True):
        self.img_size = img_size
        self.augment = augment
        
        if augment:
            self.transform = A.Compose([
                A.Resize(height=img_size[1], width=img_size[0]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.1),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=img_size[1], width=img_size[0]),
            ])
    
    def __call__(self, mid_range: np.ndarray, long_range: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        T, H, W, C = mid_range.shape
        
        mid_transformed = []
        long_transformed = []
        
        for t in range(T):
            mid_img = mid_range[t]
            long_img = long_range[t]
            
            mid_aug = self.transform(image=mid_img)['image']
            long_aug = self.transform(image=long_img)['image']
            
            mid_transformed.append(mid_aug)
            long_transformed.append(long_aug)
        
        mid_transformed = np.stack(mid_transformed, axis=0)
        long_transformed = np.stack(long_transformed, axis=0)
        
        mid_tensor = torch.from_numpy(mid_transformed).permute(0, 3, 1, 2).float() / 255.0
        long_tensor = torch.from_numpy(long_transformed).permute(0, 3, 1, 2).float() / 255.0
        
        return mid_tensor, long_tensor


class Resize:
    
    def __init__(self, size: Tuple[int, int] = (640, 640)):
        self.size = size
    
    def __call__(self, images: np.ndarray) -> torch.Tensor:
        T, H, W, C = images.shape
        resized = []
        
        for i in range(T):
            img = cv2.resize(images[i], self.size, interpolation=cv2.INTER_LINEAR)
            resized.append(img)
        
        resized = np.stack(resized, axis=0)
        tensor = torch.from_numpy(resized).permute(0, 3, 1, 2).float() / 255.0
        return tensor


class Normalize:
    
    def __init__(self, mean: list = [0.485, 0.456, 0.406], std: list = [0.229, 0.224, 0.225]):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std


class FeatureSqueezeTransform:
    
    def __init__(self, bit_depth: int = 5, median_kernel: int = 3):
        self.bit_depth = bit_depth
        self.median_kernel = median_kernel
    
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        images_np = (images * 255).cpu().numpy().astype(np.uint8)
        
        if len(images_np.shape) == 4:
            B, C, H, W = images.shape
            squeezed = []
            
            for i in range(B):
                img = images_np[i].transpose(1, 2, 0)
                squeezed_img = self._squeeze_single_image(img)
                squeezed.append(squeezed_img)
            
            squeezed = np.stack(squeezed, axis=0)
            squeezed = torch.from_numpy(squeezed).permute(0, 3, 1, 2).float() / 255.0
        else:
            img = images_np.transpose(1, 2, 0)
            squeezed = self._squeeze_single_image(img)
            squeezed = torch.from_numpy(squeezed).permute(2, 0, 1).float() / 255.0
        
        return squeezed.to(images.device)
    
    def _squeeze_single_image(self, img: np.ndarray) -> np.ndarray:
        quantized = np.floor(img * (2 ** self.bit_depth) / 256) * (256 / (2 ** self.bit_depth))
        quantized = quantized.astype(np.uint8)
        
        if self.median_kernel > 1:
            filtered = cv2.medianBlur(quantized, self.median_kernel)
        else:
            filtered = quantized
        
        return filtered


class CompositeTransform:
    
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, *args):
        for transform in self.transforms:
            args = transform(*args) if isinstance(args, tuple) else (transform(args),)
        return args[0] if len(args) == 1 else args


class TemporalAugmentation:
    
    def __init__(
        self,
        motion_blur_prob: float = 0.1,
        temporal_dropout_prob: float = 0.05,
        brightness_jitter: float = 0.1
    ):
        self.motion_blur_prob = motion_blur_prob
        self.temporal_dropout_prob = temporal_dropout_prob
        self.brightness_jitter = brightness_jitter
    
    def __call__(self, sequence: np.ndarray) -> np.ndarray:
        T, H, W, C = sequence.shape
        augmented = sequence.copy()
        
        for t in range(T):
            if np.random.random() < self.motion_blur_prob:
                kernel_size = np.random.choice([3, 5, 7])
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[kernel_size // 2, :] = 1.0
                kernel = kernel / kernel_size
                
                angle = np.random.uniform(0, 360)
                M = cv2.getRotationMatrix2D((kernel_size // 2, kernel_size // 2), angle, 1.0)
                kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
                
                augmented[t] = cv2.filter2D(augmented[t], -1, kernel)
            
            if np.random.random() < self.temporal_dropout_prob:
                if t > 0:
                    augmented[t] = augmented[t - 1]
            
            if np.random.random() < 0.3:
                brightness_factor = 1.0 + np.random.uniform(-self.brightness_jitter, self.brightness_jitter)
                augmented[t] = np.clip(augmented[t] * brightness_factor, 0, 255).astype(np.uint8)
        
        return augmented


class OcclusionAugmentation:
    
    def __init__(self, occlusion_prob: float = 0.1, max_occlusion_ratio: float = 0.3):
        self.occlusion_prob = occlusion_prob
        self.max_occlusion_ratio = max_occlusion_ratio
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() > self.occlusion_prob:
            return image
        
        H, W = image.shape[:2]
        
        occlusion_h = int(H * np.random.uniform(0.1, self.max_occlusion_ratio))
        occlusion_w = int(W * np.random.uniform(0.1, self.max_occlusion_ratio))
        
        x = np.random.randint(0, W - occlusion_w)
        y = np.random.randint(0, H - occlusion_h)
        
        occlusion_type = np.random.choice(['black', 'noise', 'blur'])
        
        occluded = image.copy()
        
        if occlusion_type == 'black':
            occluded[y:y+occlusion_h, x:x+occlusion_w] = 0
        elif occlusion_type == 'noise':
            noise = np.random.randint(0, 255, (occlusion_h, occlusion_w, image.shape[2]), dtype=np.uint8)
            occluded[y:y+occlusion_h, x:x+occlusion_w] = noise
        else:
            region = occluded[y:y+occlusion_h, x:x+occlusion_w]
            blurred = cv2.GaussianBlur(region, (15, 15), 0)
            occluded[y:y+occlusion_h, x:x+occlusion_w] = blurred
        
        return occluded


class WeatherAugmentation:
    
    def __init__(self, rain_prob: float = 0.1, fog_prob: float = 0.1):
        self.rain_prob = rain_prob
        self.fog_prob = fog_prob
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        augmented = image.copy()
        
        if np.random.random() < self.rain_prob:
            augmented = self._add_rain(augmented)
        
        if np.random.random() < self.fog_prob:
            augmented = self._add_fog(augmented)
        
        return augmented
    
    def _add_rain(self, image: np.ndarray) -> np.ndarray:
        H, W = image.shape[:2]
        
        rain_intensity = np.random.uniform(0.3, 0.7)
        num_drops = int(H * W * 0.0005 * rain_intensity)
        
        rain_layer = image.copy()
        
        for _ in range(num_drops):
            x = np.random.randint(0, W)
            y = np.random.randint(0, H)
            length = np.random.randint(10, 30)
            thickness = 1
            
            if y + length < H:
                cv2.line(rain_layer, (x, y), (x, y + length), (200, 200, 200), thickness)
        
        result = cv2.addWeighted(image, 1 - rain_intensity * 0.3, rain_layer, rain_intensity * 0.3, 0)
        return result.astype(np.uint8)
    
    def _add_fog(self, image: np.ndarray) -> np.ndarray:
        H, W = image.shape[:2]
        
        fog_intensity = np.random.uniform(0.3, 0.6)
        
        fog = np.ones_like(image) * 255
        
        result = cv2.addWeighted(image, 1 - fog_intensity, fog, fog_intensity, 0)
        return result.astype(np.uint8)


class ToYOLOFormat:
    
    def __init__(self, img_size: Tuple[int, int] = (640, 640)):
        self.img_size = img_size
    
    def __call__(self, boxes: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        if len(boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        
        orig_h, orig_w = original_size
        target_h, target_w = self.img_size
        
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        yolo_boxes = boxes.copy()
        
        yolo_boxes[:, 1] *= scale_x
        yolo_boxes[:, 2] *= scale_y
        yolo_boxes[:, 3] *= scale_x
        yolo_boxes[:, 4] *= scale_y
        
        return yolo_boxes


def get_train_transforms(img_size: Tuple[int, int] = (640, 640)) -> SequenceTransform:
    return SequenceTransform(img_size=img_size, augment=True)


def get_val_transforms(img_size: Tuple[int, int] = (640, 640)) -> SequenceTransform:
    return SequenceTransform(img_size=img_size, augment=False)


def get_defense_transforms(bit_depth: int = 5, median_kernel: int = 3) -> FeatureSqueezeTransform:
    return FeatureSqueezeTransform(bit_depth=bit_depth, median_kernel=median_kernel)