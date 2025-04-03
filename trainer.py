import torch.optim as optim
import math
from net import *
import util
import torch.nn.functional as F
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, lrate, wdecay, clip, step_size, seq_out_len, scaler, device, cl=True):
        """
        Initialize the trainer with model and training parameters.
        
        Args:
            model: The neural network model
            lrate: Learning rate
            wdecay: Weight decay
            clip: Gradient clipping value
            step_size: Step size for task level updates
            seq_out_len: Output sequence length
            scaler: Data scaler for normalization
            device: Device to run the model on
            cl: Whether to use curriculum learning
        """
        try:
            self.scaler = scaler
            self.model = model
            self.model.to(device)
            self.device = device
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lrate,
                weight_decay=wdecay,
                eps=1e-8  # Added for numerical stability
            )
            self.loss = util.masked_mae
            self.clip = clip
            self.step = step_size
            self.iter = 1
            self.task_level = 1
            self.seq_out_len = seq_out_len
            self.cl = cl
            
            # Initialize learning rate scheduler
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
            
            logger.info("Trainer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing trainer: {str(e)}")
            raise

    def train(self, input, real_val, mask=None):
        """
        Train the model for one epoch.
        
        Args:
            input: Input data
            real_val: Ground truth values
            mask: Optional mask for loss calculation
            
        Returns:
            float: Training loss
        """
        try:
            self.model.train()
            self.optimizer.zero_grad()
            
            # Move input to device
            input = input.to(self.device)
            real_val = real_val.to(self.device)
            if mask is not None:
                mask = mask.to(self.device)
            
            # Forward pass
            output = self.model(input)
            
            # Calculate loss
            if mask is not None:
                loss = self.loss(output, real_val, mask)
            else:
                loss = self.loss(output, real_val)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            
            # Update parameters
            self.optimizer.step()
            
            # Update learning rate if using curriculum learning
            if self.cl:
                if self.iter % self.step == 0:
                    self.task_level += 1
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.5
                self.iter += 1
            
            return loss.item()
            
        except Exception as e:
            logger.error(f"Error in training step: {str(e)}")
            raise

    def eval(self, input, real_val, mask=None):
        """
        Evaluate the model.
        
        Args:
            input: Input data
            real_val: Ground truth values
            mask: Optional mask for loss calculation
            
        Returns:
            float: Evaluation loss
        """
        try:
            self.model.eval()
            with torch.no_grad():
                # Move input to device
                input = input.to(self.device)
                real_val = real_val.to(self.device)
                if mask is not None:
                    mask = mask.to(self.device)
                
                # Forward pass
                output = self.model(input)
                
                # Calculate loss
                if mask is not None:
                    loss = self.loss(output, real_val, mask)
                else:
                    loss = self.loss(output, real_val)
                
                return loss.item()
                
        except Exception as e:
            logger.error(f"Error in evaluation step: {str(e)}")
            raise

    def save(self, path):
        """
        Save the model state.
        
        Args:
            path: Path to save the model
        """
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler': self.scaler,
                'seq_out_len': self.seq_out_len,
                'task_level': self.task_level,
                'iter': self.iter
            }, path)
            logger.info(f"Model saved successfully to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load(self, path):
        """
        Load the model state.
        
        Args:
            path: Path to load the model from
        """
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scaler = checkpoint['scaler']
            self.seq_out_len = checkpoint['seq_out_len']
            self.task_level = checkpoint['task_level']
            self.iter = checkpoint['iter']
            logger.info(f"Model loaded successfully from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

class Optim:
    def __init__(self, params, method, lr, clip, lr_decay=1, start_decay_at=None, optimizer=None):
        """
        Initialize the optimizer.
        
        Args:
            params: Model parameters
            method: Optimization method
            lr: Learning rate
            clip: Gradient clipping value
            lr_decay: Learning rate decay factor
            start_decay_at: Epoch to start learning rate decay
            optimizer: Optional pre-configured optimizer
        """
        try:
            self.params = params
            self.last_ppl = None
            self.lr = lr
            self.clip = clip
            self.method = method
            self.lr_decay = lr_decay
            self.start_decay_at = start_decay_at
            self.start_decay = False
            
            if optimizer is None:
                self._makeOptimizer()
            else:
                self.optimizer = optimizer
                
            logger.info(f"Optimizer initialized with method: {method}")
            
        except Exception as e:
            logger.error(f"Error initializing optimizer: {str(e)}")
            raise

    def _makeOptimizer(self):
        """Create the optimizer based on the specified method."""
        try:
            if self.method == 'sgd':
                self.optimizer = optim.SGD(
                    self.params,
                    lr=self.lr,
                    weight_decay=self.lr_decay,
                    momentum=0.9  # Added momentum for better convergence
                )
            elif self.method == 'adagrad':
                self.optimizer = optim.Adagrad(
                    self.params,
                    lr=self.lr,
                    weight_decay=self.lr_decay
                )
            elif self.method == 'adadelta':
                self.optimizer = optim.Adadelta(
                    self.params,
                    lr=self.lr,
                    weight_decay=self.lr_decay
                )
            elif self.method == 'adam':
                self.optimizer = optim.Adam(
                    self.params,
                    lr=self.lr,
                    weight_decay=self.lr_decay,
                    eps=1e-8  # Added for numerical stability
                )
            else:
                raise ValueError(f"Invalid optimization method: {self.method}")
                
        except Exception as e:
            logger.error(f"Error creating optimizer: {str(e)}")
            raise

    def step(self):
        """
        Perform one optimization step.
        
        Returns:
            float: Gradient norm
        """
        try:
            grad_norm = 0
            if self.clip is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.clip)
            
            self.optimizer.step()
            return grad_norm
            
        except Exception as e:
            logger.error(f"Error in optimization step: {str(e)}")
            raise

    def updateLearningRate(self, ppl, epoch):
        """
        Update the learning rate based on perplexity and epoch.
        
        Args:
            ppl: Perplexity value
            epoch: Current epoch number
        """
        try:
            if self.start_decay_at is not None and epoch >= self.start_decay_at:
                self.start_decay = True
            if self.last_ppl is not None and ppl > self.last_ppl:
                self.start_decay = True

            if self.start_decay:
                self.lr = self.lr * self.lr_decay
                logger.info(f"Decaying learning rate to {self.lr}")
                self.start_decay = False

            self.last_ppl = ppl
            self._makeOptimizer()
            
        except Exception as e:
            logger.error(f"Error updating learning rate: {str(e)}")
            raise
