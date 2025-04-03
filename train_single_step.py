import argparse
import math
import time
import logging
from timeit import default_timer as timer
import torch
import torch.nn as nn
from net import gtnet
import numpy as np
import importlib
import pandas as pd
from util import *
from trainer import Optim, Trainer
from ranger21 import Ranger21

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    """
    Evaluate the model on given data.
    
    Args:
        data: DataLoader object
        X: Input features
        Y: Target values
        model: Neural network model
        evaluateL2: L2 loss criterion
        evaluateL1: L1 loss criterion
        batch_size: Batch size for evaluation
        
    Returns:
        tuple: (RSE, RAE, correlation)
    """
    try:
        model.eval()
        total_loss = 0
        total_loss_l1 = 0
        n_samples = 0
        predict = None
        test = None

        with torch.no_grad():
            for X, Y in data.get_batches(X, Y, batch_size, False):
                X = torch.unsqueeze(X, dim=1)
                X = X.transpose(2, 3)
                output = model(X)
                output = torch.squeeze(output)
                
                if len(output.shape) == 1:
                    output = output.unsqueeze(dim=0)
                    
                if predict is None:
                    predict = output
                    test = Y
                else:
                    predict = torch.cat((predict, output))
                    test = torch.cat((test, Y))

                scale = data.scale.expand(output.size(0), data.m)
                total_loss += evaluateL2(output * scale, Y * scale).item()
                total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
                n_samples += (output.size(0) * data.m)

        rse = math.sqrt(total_loss / n_samples) / data.rse
        rae = (total_loss_l1 / n_samples) / data.rae

        predict = predict.data.cpu().numpy()
        Ytest = test.data.cpu().numpy()
        sigma_p = predict.std(axis=0)
        sigma_g = Ytest.std(axis=0)
        mean_p = predict.mean(axis=0)
        mean_g = Ytest.mean(axis=0)
        index = (sigma_g != 0)
        correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        correlation = (correlation[index]).mean()
        
        return rse, rae, correlation
        
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        raise

def train(data, X, Y, model, criterion, optimizer, batch_size):
    """
    Train the model for one epoch.
    
    Args:
        data: DataLoader object
        X: Input features
        Y: Target values
        model: Neural network model
        criterion: Loss criterion
        optimizer: Optimizer
        batch_size: Batch size for training
        
    Returns:
        float: Average training loss
    """
    try:
        model.train()
        total_loss = 0
        n_samples = 0
        iter = 0

        for X, Y in data.get_batches(X, Y, batch_size, False):
            model.zero_grad()
            X = torch.unsqueeze(X, dim=1)
            X = X.transpose(2, 3)
            
            if iter % args.step_size == 0:
                perm = np.random.permutation(range(args.num_nodes))
            num_sub = int(args.num_nodes / args.num_split)

            for j in range(args.num_split):
                try:
                    if j != args.num_split - 1:
                        id = perm[j * num_sub:(j + 1) * num_sub]
                    else:
                        id = perm[j * num_sub:]
                    id = torch.LongTensor(id).to(device)
                    tx = X[:, :, id, :]
                    ty = Y[:, id]
                    
                    output = model(tx, id)
                    output = torch.squeeze(output)
                    scale = data.scale.expand(output.size(0), data.m)
                    scale = scale[:, id]
                    
                    loss = criterion(output * scale, ty * scale)
                    loss.backward()
                    
                    if args.clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                    n_samples += (output.size(0) * data.m)
                    
                except Exception as e:
                    logger.error(f"Error in training batch {j}: {str(e)}")
                    continue

            if iter % 100 == 0:
                logger.info(f'iter:{iter:3d} | loss: {loss.item()/(output.size(0) * data.m):.3f}')
            iter += 1
            
        return total_loss / n_samples
        
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        raise

def main():
    """
    Main training function.
    
    Returns:
        tuple: (validation metrics, test metrics)
    """
    try:
        # Set random seeds for reproducibility
        torch.manual_seed(1234)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Initialize data loader
        Data = DataLoaderS(args.data, 0.6, 0.2, device, args.horizon, args.seq_in_len, args.normalize)

        # Load or construct adjacency matrix
        if args.buildA_true:
            predefined_A = None
        else:
            try:
                predefined_A = pd.read_csv(args.predefinedA_path)
                predefined_A = torch.from_numpy(predefined_A.to_numpy().astype(np.float32))
                predefined_A = torch.tensor(predefined_A) + torch.eye(args.num_nodes)
                predefined_A = predefined_A.to(device)
            except Exception as e:
                logger.error(f"Error loading adjacency matrix: {str(e)}")
                raise

        # Calculate batches per epoch
        batch_per_epoch = sum(1 for _ in Data.get_batches(Data.train[0], Data.train[1], args.batch_size, False))

        # Initialize model
        model = gtnet(
            args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
            device, predefined_A=predefined_A, dropout=args.dropout,
            subgraph_size=args.subgraph_size, node_dim=args.node_dim,
            dilation_exponential=args.dilation_exponential,
            conv_channels=args.conv_channels, residual_channels=args.residual_channels,
            skip_channels=args.skip_channels, end_channels=args.end_channels,
            seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
            layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha,
            layer_norm_affline=False
        )
        model = model.to(device)

        logger.info(f"Model parameters: {sum(p.nelement() for p in model.parameters())}")
        logger.info(f"Receptive field size: {model.receptive_field}")

        # Initialize loss criteria
        if args.L1Loss:
            criterion = nn.L1Loss(size_average=False).to(device)
        else:
            criterion = nn.MSELoss(size_average=False).to(device)
        evaluateL2 = nn.MSELoss(size_average=False).to(device)
        evaluateL1 = nn.L1Loss(size_average=False).to(device)

        # Load pretrained model if specified
        if args.pretrained_model != 'no':
            try:
                model = torch.load(args.pretrained_model)
                logger.info("Loaded pretrained model")
            except Exception as e:
                logger.error(f"Error loading pretrained model: {str(e)}")
                raise

        # Initialize optimizers
        ranger_optimizer = Ranger21(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            use_warmup=True,
            lookahead_active=True,
            normloss_active=False,
            normloss_factor=6e-4,
            use_adaptive_gradient_clipping=True,
            use_madgrad=False,
            warmdown_active=True,
            use_cheb=False,
            num_epochs=args.epochs,
            warmup_pct_default=0.3,
            num_batches_per_epoch=batch_per_epoch
        )

        optim = Optim(
            model.parameters(),
            args.optim,
            args.lr,
            args.clip,
            lr_decay=args.weight_decay
        )

        # Training loop
        best_val = float('inf')
        try:
            logger.info('Starting training')
            for epoch in range(1, args.epochs + 1):
                epoch_start_time = time.time()
                
                # Training phase
                train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
                
                # Validation phase
                val_loss, val_rae, val_corr = evaluate(
                    Data, Data.valid[0], Data.valid[1],
                    model, evaluateL2, evaluateL1, args.batch_size
                )
                
                # Log epoch results
                logger.info(
                    f'| end of epoch {epoch:3d} | time: {time.time() - epoch_start_time:5.2f}s | '
                    f'train_loss {train_loss:5.4f} | valid rse {val_loss:5.4f} | '
                    f'valid rae {val_rae:5.4f} | valid corr {val_corr:5.4f}'
                )

                # Save best model
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(model, args.save)
                    logger.info(f"Saved best model with validation loss: {best_val:.4f}")

                # Evaluate on test set every 5 epochs
                if epoch % 5 == 0:
                    test_acc, test_rae, test_corr = evaluate(
                        Data, Data.test[0], Data.test[1],
                        model, evaluateL2, evaluateL1, args.batch_size
                    )
                    logger.info(
                        f"test rse {test_acc:5.4f} | test rae {test_rae:5.4f} | "
                        f"test corr {test_corr:5.4f}"
                    )

        except KeyboardInterrupt:
            logger.info('-' * 89)
            logger.info('Exiting from training early')

        # Load best model and evaluate
        model = torch.load(args.save)
        vtest_acc, vtest_rae, vtest_corr = evaluate(
            Data, Data.valid[0], Data.valid[1],
            model, evaluateL2, evaluateL1, args.batch_size
        )
        test_acc, test_rae, test_corr = evaluate(
            Data, Data.test[0], Data.test[1],
            model, evaluateL2, evaluateL1, args.batch_size
        )
        
        return vtest_acc, vtest_rae, vtest_corr, test_acc, test_rae, test_corr
        
    except Exception as e:
        logger.error(f"Error in main training loop: {str(e)}")
        raise

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
    parser.add_argument('--data', type=str, default='./data/solar_AL.txt',
                      help='location of the data file')
    parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                      help='report interval')
    parser.add_argument('--save', type=str, default='model/model.pt',
                      help='path to save the final model')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--L1Loss', type=bool, default=False)
    parser.add_argument('--normalize', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0', help='')
    parser.add_argument('--gcn_true', type=bool, default=True,
                      help='whether to add graph convolution layer')
    parser.add_argument('--buildA_true', type=bool, default=False,
                      help='whether to construct adaptive adjacency matrix')
    parser.add_argument('--pretrained_model', type=str, default='no',
                      help='path to pretrained model')
    parser.add_argument('--predefinedA_path', type=str, default='./data/solar_AL.txt',
                      help='location of adjacency matrix')
    parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
    parser.add_argument('--num_nodes', type=int, default=137, help='number of nodes/variables')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--subgraph_size', type=int, default=20, help='k')
    parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
    parser.add_argument('--dilation_exponential', type=int, default=2, help='dilation exponential')
    parser.add_argument('--conv_channels', type=int, default=16, help='convolution channels')
    parser.add_argument('--residual_channels', type=int, default=16, help='residual channels')
    parser.add_argument('--skip_channels', type=int, default=32, help='skip channels')
    parser.add_argument('--end_channels', type=int, default=64, help='end channels')
    parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
    parser.add_argument('--seq_in_len', type=int, default=24*7, help='input sequence length')
    parser.add_argument('--seq_out_len', type=int, default=1, help='output sequence length')
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--layers', type=int, default=5, help='number of layers')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay rate')
    parser.add_argument('--clip', type=int, default=5, help='clip')
    parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
    parser.add_argument('--tanhalpha', type=float, default=3, help='tanh alpha')
    parser.add_argument('--epochs', type=int, default=1, help='')
    parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')
    parser.add_argument('--step_size', type=int, default=100, help='step_size')

    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(3)

    # Run training multiple times for averaging
    vacc = []
    vrae = []
    vcorr = []
    acc = []
    rae = []
    corr = []

    for i in range(1):
        start = timer()
        try:
            val_acc, val_rae, val_corr, test_acc, test_rae, test_corr = main()
            end = timer()
            
            logger.info(f'\nTime taken per run: {end - start:.2f}s\n')
            
            vacc.append(val_acc)
            vrae.append(val_rae)
            vcorr.append(val_corr)
            acc.append(test_acc)
            rae.append(test_rae)
            corr.append(test_corr)
            
        except Exception as e:
            logger.error(f"Error in run {i}: {str(e)}")
            continue

    # Print final results
    logger.info('\n10 runs average')
    logger.info('\nvalid\trse\trae\tcorr')
    logger.info(f"mean\t{np.mean(vacc):5.4f}\t{np.mean(vrae):5.4f}\t{np.mean(vcorr):5.4f}")
    logger.info(f"std\t{np.std(vacc):5.4f}\t{np.std(vrae):5.4f}\t{np.std(vcorr):5.4f}")
    
    logger.info('\ntest\trse\trae\tcorr')
    logger.info(f"mean\t{np.mean(acc):5.4f}\t{np.mean(rae):5.4f}\t{np.mean(corr):5.4f}")
    logger.info(f"std\t{np.std(acc):5.4f}\t{np.std(rae):5.4f}\t{np.std(corr):5.4f}")
    
    
    
    #python train_single_step.py --save ./model-RE.pt  --pretrained_model ./model-RE.pt --data ./data/og_dataset.csv --num_nodes 1494 --batch_size 12 --epochs 30 --horizon 24 --predefinedA_path ./data/og_recon_adj.csv 

