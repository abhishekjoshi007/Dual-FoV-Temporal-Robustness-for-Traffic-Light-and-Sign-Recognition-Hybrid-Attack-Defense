import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging
from .natural_perturbations import NaturalPerturbationSuite

logger = logging.getLogger(__name__)


class HybridAttackSuite:
    
    def __init__(
        self,
        natural_suite: Optional[NaturalPerturbationSuite] = None,
        epsilon: float = 8.0 / 255.0,
        alpha: float = None,
        num_iter: int = 10,
        targeted: bool = False,
    ):
        self.natural_suite = natural_suite if natural_suite else NaturalPerturbationSuite()
        self.epsilon = epsilon
        # Calculate alpha α = 2.5 × ε / 10 
        # For ε=8/255, this gives α ≈ 0.000078 (not 2.0/255 = 0.00784)
        self.alpha = alpha if alpha is not None else 2.5 * epsilon / 10
        self.num_iter = num_iter
        self.targeted = targeted

        logger.info(f"HybridAttackSuite initialized with epsilon={epsilon}, alpha={self.alpha}, num_iter={num_iter}")
    
    def fgsm_attack(
        self,
        model,
        images: torch.Tensor,
        targets: torch.Tensor,
        epsilon: Optional[float] = None,
    ) -> torch.Tensor:
        if epsilon is None:
            epsilon = self.epsilon
        
        images = images.clone().detach().requires_grad_(True)
        
        outputs = model(images)
        loss = F.cross_entropy(outputs, targets)
        
        model.zero_grad()
        loss.backward()
        
        grad = images.grad.data
        
        if self.targeted:
            perturbed = images - epsilon * grad.sign()
        else:
            perturbed = images + epsilon * grad.sign()
        
        perturbed = torch.clamp(perturbed, 0, 1)
        
        return perturbed
    
    def pgd_attack(
        self,
        model,
        images: torch.Tensor,
        targets: torch.Tensor,
        epsilon: Optional[float] = None,
        alpha: Optional[float] = None,
        num_iter: Optional[int] = None,
    ) -> torch.Tensor:
        if epsilon is None:
            epsilon = self.epsilon
        if alpha is None:
            alpha = self.alpha
        if num_iter is None:
            num_iter = self.num_iter
        
        perturbed = images.clone().detach()
        perturbed = perturbed + torch.empty_like(perturbed).uniform_(-epsilon, epsilon)
        perturbed = torch.clamp(perturbed, 0, 1)
        
        for i in range(num_iter):
            perturbed.requires_grad = True
            
            outputs = model(perturbed)
            loss = F.cross_entropy(outputs, targets)
            
            model.zero_grad()
            loss.backward()
            
            grad = perturbed.grad.data
            
            if self.targeted:
                perturbed = perturbed - alpha * grad.sign()
            else:
                perturbed = perturbed + alpha * grad.sign()
            
            eta = torch.clamp(perturbed - images, -epsilon, epsilon)
            perturbed = torch.clamp(images + eta, 0, 1).detach()
        
        return perturbed
    
    def uap_attack(
        self,
        model,
        images: torch.Tensor,
        perturbation: torch.Tensor,
        epsilon: Optional[float] = None,
    ) -> torch.Tensor:
        if epsilon is None:
            epsilon = self.epsilon
        
        perturbed = images + perturbation
        perturbed = torch.clamp(perturbed, 0, 1)
        
        return perturbed
    
    def generate_uap(
        self,
        model,
        dataloader,
        num_images: int = 200,
        epsilon: Optional[float] = None,
        num_iter: int = 10,
    ) -> torch.Tensor:
        if epsilon is None:
            epsilon = self.epsilon
        
        device = next(model.parameters()).device
        
        sample_images = []
        for batch_idx, batch in enumerate(dataloader):
            if len(sample_images) >= num_images:
                break
            
            images = batch['mid_range'][:, 0, :, :, :] if images.dim() == 5 else batch['mid_range']
            sample_images.append(images)
        
        sample_images = torch.cat(sample_images, dim=0)[:num_images].to(device)
        
        perturbation = torch.zeros_like(sample_images[0]).to(device)
        
        for iteration in range(num_iter):
            fooled_count = 0
            
            for img in sample_images:
                img = img.unsqueeze(0)
                img.requires_grad = True
                
                perturbed = torch.clamp(img + perturbation, 0, 1)
                
                outputs = model(perturbed)
                original_outputs = model(img)
                
                original_pred = torch.argmax(original_outputs, dim=1)
                perturbed_pred = torch.argmax(outputs, dim=1)
                
                if original_pred != perturbed_pred:
                    fooled_count += 1
                    continue
                
                loss = F.cross_entropy(outputs, original_pred)
                
                model.zero_grad()
                loss.backward()
                
                grad = img.grad.data
                perturbation = perturbation + self.alpha * grad.sign().squeeze(0)
                perturbation = torch.clamp(perturbation, -epsilon, epsilon)
            
            logger.info(f"UAP iteration {iteration + 1}/{num_iter}, fooled {fooled_count}/{len(sample_images)}")
        
        return perturbation
    
    def hybrid_attack(
        self,
        model,
        images: torch.Tensor,
        targets: torch.Tensor,
        natural_perturbation: str = 'rain',
        natural_intensity: float = 0.5,
        digital_attack: str = 'pgd',
        bbox: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        images_np = (images.cpu().numpy() * 255).astype(np.uint8)
        
        if images_np.ndim == 4:
            perturbed_np = []
            for i in range(len(images_np)):
                img = images_np[i].transpose(1, 2, 0)
                perturbed = self.natural_suite.apply_perturbation(
                    img, natural_perturbation, natural_intensity, bbox
                )
                perturbed_np.append(perturbed)
            perturbed_np = np.stack(perturbed_np, axis=0)
            perturbed_np = perturbed_np.transpose(0, 3, 1, 2)
        else:
            img = images_np.transpose(1, 2, 0)
            perturbed_np = self.natural_suite.apply_perturbation(
                img, natural_perturbation, natural_intensity, bbox
            )
            perturbed_np = perturbed_np.transpose(2, 0, 1)
        
        perturbed_tensor = torch.from_numpy(perturbed_np).float() / 255.0
        perturbed_tensor = perturbed_tensor.to(images.device)
        
        if digital_attack == 'fgsm':
            perturbed_tensor = self.fgsm_attack(model, perturbed_tensor, targets)
        elif digital_attack == 'pgd':
            perturbed_tensor = self.pgd_attack(model, perturbed_tensor, targets)
        else:
            logger.warning(f"Unknown digital attack: {digital_attack}")
        
        return perturbed_tensor
    
    def hybrid_sequence_attack(
        self,
        model,
        sequence: torch.Tensor,
        targets: List[torch.Tensor],
        natural_perturbation: str = 'rain',
        natural_intensity: float = 0.5,
        digital_attack: str = 'pgd',
        bboxes: Optional[List[np.ndarray]] = None,
    ) -> torch.Tensor:
        B, T, C, H, W = sequence.shape
        
        sequence_np = (sequence.cpu().numpy() * 255).astype(np.uint8)
        
        perturbed_sequences = []
        for b in range(B):
            seq = sequence_np[b].transpose(0, 2, 3, 1)
            
            perturbed_seq = self.natural_suite.apply_sequence_perturbation(
                seq, natural_perturbation, natural_intensity, bboxes
            )
            
            perturbed_seq = perturbed_seq.transpose(0, 3, 1, 2)
            perturbed_sequences.append(perturbed_seq)
        
        perturbed_sequences = np.stack(perturbed_sequences, axis=0)
        perturbed_tensor = torch.from_numpy(perturbed_sequences).float() / 255.0
        perturbed_tensor = perturbed_tensor.to(sequence.device)
        
        if digital_attack in ['fgsm', 'pgd']:
            for t in range(T):
                frame = perturbed_tensor[:, t, :, :, :]
                target = targets[t] if isinstance(targets, list) else targets
                
                if digital_attack == 'fgsm':
                    perturbed_frame = self.fgsm_attack(model, frame, target)
                else:
                    perturbed_frame = self.pgd_attack(model, frame, target)
                
                perturbed_tensor[:, t, :, :, :] = perturbed_frame
        
        return perturbed_tensor
    
    def evaluate_attack_success(
        self,
        model,
        clean_images: torch.Tensor,
        perturbed_images: torch.Tensor,
    ) -> Dict:
        with torch.no_grad():
            clean_outputs = model(clean_images)
            perturbed_outputs = model(perturbed_images)
        
        clean_preds = torch.argmax(clean_outputs, dim=1)
        perturbed_preds = torch.argmax(perturbed_outputs, dim=1)
        
        misclassified = (clean_preds != perturbed_preds).sum().item()
        total = len(clean_images)
        
        asr = (misclassified / total) * 100
        
        l2_distance = torch.norm(perturbed_images - clean_images, p=2).item()
        linf_distance = torch.max(torch.abs(perturbed_images - clean_images)).item()
        
        return {
            'asr': asr,
            'misclassified': misclassified,
            'total': total,
            'l2_distance': l2_distance,
            'linf_distance': linf_distance,
        }