"""
Flow Matching algorithm implementation.
Adapted from MeanAudio project for 3DTalkingHeadCodeBase.
"""
import logging
from typing import Callable, Optional

import torch
from torchdiffeq import odeint

log = logging.getLogger(__name__)


class FlowMatching:
    """Flow Matching for generative modeling."""

    def __init__(self, min_sigma: float = 0.0, 
                 inference_mode='euler', 
                 num_steps: int = 25, 
                 reverse_flow: bool = True):
        """
        Args:
            min_sigma: Minimum sigma value for numerical stability
            inference_mode: 'euler' or 'adaptive' for ODE solving
            num_steps: Number of steps in the euler inference mode
            reverse_flow: Whether to use reverse flow (x1->x0) or forward flow (x0->x1)
        """
        super().__init__()
        self.min_sigma = min_sigma
        self.inference_mode = inference_mode
        self.num_steps = num_steps
        self.reverse_flow = reverse_flow

        assert self.inference_mode in ['euler', 'adaptive']
        if self.inference_mode == 'adaptive' and num_steps > 0:
            log.info('The number of steps is ignored in adaptive inference mode')

    def get_conditional_flow(self, x0: torch.Tensor, x1: torch.Tensor,
                             t: torch.Tensor) -> torch.Tensor:
        """
        Get the conditional flow at time t.
        This is psi_t(x), eq 22 in flow matching for generative models.
        
        Args:
            x0: Prior samples (noise)
            x1: Data samples
            t: Time steps [0, 1]
            
        Returns:
            Interpolated samples at time t
        """
        t = t[:, None, None].expand_as(x0)
        if self.reverse_flow:
            return (1 - t) * x1 + t * x0  # xt = (1-t)*x1 + t*x0 -> vt = x0 - x1
        else:
            return (1 - t) * x0 + t * x1  # xt = (1-t)*x0 + t*x1 -> vt = x1 - x0

    def loss(self, predicted_v: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        Compute the flow matching loss.
        
        Args:
            predicted_v: Predicted velocity/flow
            x0: Prior samples (noise)
            x1: Data samples
            
        Returns:
            Loss tensor (batch dimension not reduced)
        """
        # return the mean error without reducing the batch dimension
        reduce_dim = list(range(1, len(predicted_v.shape)))
        if self.reverse_flow:
            target_v = x0 - x1  
        else:
            target_v = x1 - x0 
        return (predicted_v - target_v).pow(2).mean(dim=reduce_dim)

    def get_x0_xt_c(
        self,
        x1: torch.Tensor,
        t: torch.Tensor,
        Cs: list[torch.Tensor],
        generator: Optional[torch.Generator] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        Get x0, x1, xt and conditions for training.
        
        Args:
            x1: Data samples
            t: Time steps
            Cs: List of condition tensors
            generator: Random generator for reproducibility
            
        Returns:
            Tuple of (x0, x1, xt, Cs)
        """
        x0 = torch.empty_like(x1).normal_(generator=generator)
        xt = self.get_conditional_flow(x0, x1, t)
        return x0, x1, xt, Cs

    def to_prior(self, fn: Callable, x1: torch.Tensor) -> torch.Tensor:
        """
        Convert data samples to prior samples (encoding).
        
        Args:
            fn: ODE function that takes (t, x) and returns velocity
            x1: Data samples
            
        Returns:
            Prior samples x0
        """
        if self.reverse_flow:
            return self.run_t0_to_t1(fn, x1, 0, 1)
        else: 
            return self.run_t0_to_t1(fn, x1, 1, 0)

    def to_data(self, fn: Callable, x0: torch.Tensor) -> torch.Tensor:
        """
        Convert prior samples to data samples (decoding/sampling).
        
        Args:
            fn: ODE function that takes (t, x) and returns velocity
            x0: Prior samples
            
        Returns:
            Data samples x1
        """
        if self.reverse_flow:
            return self.run_t0_to_t1(fn, x0, 1, 0)
        else:
            return self.run_t0_to_t1(fn, x0, 0, 1)

    def run_t0_to_t1(self, fn: Callable, x0: torch.Tensor, t0: float, t1: float) -> torch.Tensor:
        """
        Solve ODE from t0 to t1.
        
        Args:
            fn: A function that takes (t, x) and returns the velocity
            x0: Initial state
            t0: Start time
            t1: End time
            
        Returns:
            Final state at time t1
        """
        if self.inference_mode == 'adaptive':
            return odeint(fn, x0, torch.tensor([t0, t1], device=x0.device, dtype=x0.dtype))[-1]
        elif self.inference_mode == 'euler':
            x = x0
            steps = torch.linspace(t0, t1 - self.min_sigma, self.num_steps + 1, device=x0.device)
            for ti, t in enumerate(steps[:-1]):
                flow = fn(t, x)
                next_t = steps[ti + 1]
                dt = next_t - t
                x = x + dt * flow
            return x
        else:
            raise ValueError(f'Unknown inference mode: {self.inference_mode}')
