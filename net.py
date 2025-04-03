from layer import *
import torch.nn.functional as F


class gtnet(nn.Module):                                                                                                                        # 2 def passed       16 if def passed   16 if def passed         32 def            64 def         24*7 def       1 def     1 def      5 def                                                         
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(gtnet, self).__init__()
        self.gcn_true = gcn_true 
        self.buildA_true = buildA_true 
        self.num_nodes = num_nodes 
        self.dropout = dropout
        self.predefined_A = predefined_A 
        self.device = device
        
        # Initialize module lists with proper error handling
        try:
            self.filter_convs = nn.ModuleList()  
            self.gate_convs = nn.ModuleList()
            self.residual_convs = nn.ModuleList()
            self.skip_convs = nn.ModuleList()
            self.gconv1 = nn.ModuleList()
            self.gconv2 = nn.ModuleList()
            self.norm = nn.ModuleList()
        except Exception as e:
            print(f"Error initializing module lists: {str(e)}")
            raise

        # Initialize start convolution with proper error handling
        try:
            self.start_conv = nn.Conv2d(
                in_channels=in_dim, 
                out_channels=residual_channels, 
                kernel_size=(1, 1)
            )
            self.start_conv.apply(init_weights)
        except Exception as e:
            print(f"Error initializing start convolution: {str(e)}")
            raise

        # Initialize graph constructor with proper error handling
        try:
            self.gc = graph_attention_constructor(
                num_nodes, 
                subgraph_size, 
                node_dim, 
                device, 
                alpha=tanhalpha, 
                static_feat=static_feat
            )
        except Exception as e:
            print(f"Error initializing graph constructor: {str(e)}")
            raise

        # Initialize decomposition layer
        self.decomp = decom_2D(seq_length, 1, 1) 
        self.seq_length = seq_length

        # Calculate receptive field size with proper error handling
        try:
            kernel_size = 7
            if dilation_exponential > 1:
                self.receptive_field = int(1 + (kernel_size-1) * (dilation_exponential**layers-1) / (dilation_exponential-1))
            else:
                self.receptive_field = layers * (kernel_size-1) + 1
        except Exception as e:
            print(f"Error calculating receptive field: {str(e)}")
            raise

        # Initialize network layers with proper error handling
        try:
            for i in range(1):
                if dilation_exponential > 1:
                    rf_size_i = int(1 + i * (kernel_size-1) * (dilation_exponential**layers-1) / (dilation_exponential-1))
                else:
                    rf_size_i = i * layers * (kernel_size-1) + 1
                
                new_dilation = 1
                for j in range(1, layers+1):
                    if dilation_exponential > 1:
                        rf_size_j = int(rf_size_i + (kernel_size-1) * (dilation_exponential**j-1) / (dilation_exponential-1))
                    else:
                        rf_size_j = rf_size_i + j * (kernel_size-1)

                    # Add layers with proper error handling
                    self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                    self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                    self.residual_convs.append(nn.Conv2d(
                        in_channels=conv_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1)
                    ))

                    # Add skip connections with proper error handling
                    if self.seq_length > self.receptive_field:
                        self.skip_convs.append(nn.Conv2d(
                            in_channels=conv_channels,
                            out_channels=skip_channels,
                            kernel_size=(1, self.seq_length-rf_size_j+1)
                        ))
                    else:
                        self.skip_convs.append(nn.Conv2d(
                            in_channels=conv_channels,
                            out_channels=skip_channels,
                            kernel_size=(1, self.receptive_field-rf_size_j+1)
                        ))

                    # Add graph convolutions if enabled
                    if self.gcn_true:
                        self.gconv1.append(MixPropAttention(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                        self.gconv2.append(MixPropAttention(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                    # Add normalization layers
                    if self.seq_length > self.receptive_field:
                        self.norm.append(LayerNorm(
                            (residual_channels, num_nodes, self.seq_length - rf_size_j + 1),
                            elementwise_affine=layer_norm_affline
                        ))
                    else:
                        self.norm.append(LayerNorm(
                            (residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),
                            elementwise_affine=layer_norm_affline
                        ))

                    new_dilation *= dilation_exponential
        except Exception as e:
            print(f"Error initializing network layers: {str(e)}")
            raise

        # Initialize end convolutions with proper error handling
        try:
            self.layers = layers
            self.end_conv_1 = nn.Conv2d(
                in_channels=skip_channels,
                out_channels=end_channels,
                kernel_size=(1,1),
                bias=True
            )
            self.end_conv_2 = nn.Conv2d(
                in_channels=end_channels,
                out_channels=out_dim,
                kernel_size=(1,1),
                bias=True
            )
            self.end_conv_1.apply(init_weights)
            self.end_conv_2.apply(init_weights)

            # Initialize skip connections
            if self.seq_length > self.receptive_field:
                self.skip0 = nn.Conv2d(
                    in_channels=in_dim, 
                    out_channels=skip_channels, 
                    kernel_size=(1, self.seq_length), 
                    bias=True
                )
                self.skipE = nn.Conv2d(
                    in_channels=residual_channels, 
                    out_channels=skip_channels, 
                    kernel_size=(1, self.seq_length-self.receptive_field+1), 
                    bias=True
                )
            else:
                self.skip0 = nn.Conv2d(
                    in_channels=in_dim, 
                    out_channels=skip_channels, 
                    kernel_size=(1, self.receptive_field), 
                    bias=True
                )
                self.skipE = nn.Conv2d(
                    in_channels=residual_channels, 
                    out_channels=skip_channels, 
                    kernel_size=(1, 1), 
                    bias=True
                )
            self.skip0.apply(init_weights)
            self.skipE.apply(init_weights)
        except Exception as e:
            print(f"Error initializing end convolutions: {str(e)}")
            raise

        # Initialize other components
        self.idx = torch.arange(self.num_nodes).to(device)
        self.act1 = torch.nn.SELU()
        self.act2 = torch.nn.SELU()
        self.act3 = torch.nn.SELU()

    def forward(self, input, idx=None):
        try:
            # Get input dimensions
            batch_size, feature_dim, num_nodes, input_window = input.shape
            
            # Get last cycle and reshape
            last_cycle = input[:, 0, :, -1:]
            outputs, r, tr = self.decomp(input)
            outputs = outputs + last_cycle
            outputs = outputs.reshape(batch_size, 1, num_nodes, 1)
            
            # Validate sequence length
            seq_len = input.size(3)
            if seq_len != self.seq_length:
                raise ValueError(f'Input sequence length {seq_len} not equal to preset sequence length {self.seq_length}')

            # Pad input if necessary
            if self.seq_length < self.receptive_field:
                input = F.pad(input, (self.receptive_field-self.seq_length, 0, 0, 0))

            # Get adjacency matrix
            if self.gcn_true:
                if self.buildA_true:
                    adp = self.gc(idx if idx is not None else self.idx)
                else:
                    adp = self.predefined_A
                    if adp is None:
                        raise ValueError("Predefined adjacency matrix is required when gcn_true is True and buildA_true is False")

            # Forward pass through network
            x = self.start_conv(input)
            skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
            
            for i in range(self.layers):
                residual = x
                filter_out = self.filter_convs[i](x)
                filter_out = F.dropout(filter_out, self.dropout, training=self.training)
                gate_out = self.gate_convs[i](x)
                gate_out = F.dropout(gate_out, self.dropout, training=self.training)
                x = filter_out * torch.sigmoid(gate_out)
                x = self.residual_convs[i](x)
                x = x + residual[:, :, :, -x.size(3):]
                
                if self.gcn_true:
                    x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1,0))
                
                if self.seq_length > self.receptive_field:
                    skip = self.skip_convs[i](F.dropout(x, self.dropout, training=self.training)) + skip[:, :, :, -(self.seq_length-rf_size_j+1):]
                else:
                    skip = self.skip_convs[i](F.dropout(x, self.dropout, training=self.training)) + skip[:, :, :, -(self.receptive_field-rf_size_j+1):]
                
                x = self.norm[i](x)
            
            x = F.relu(skip)
            x = F.relu(self.skipE(x))
            x = F.relu(self.end_conv_1(x))
            x = self.end_conv_2(x)
            
            return x
            
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            raise
