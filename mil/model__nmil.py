import torch

class NMILArchitecture(torch.nn.Module):
    """Neural Multiple Instance Learning architecture with clinical data integration."""
    
    def __init__(self, classes, aggregation, only_images, only_clinical, clinical_classes, 
                 neurons_1, neurons_2, neurons_3, neurons_att_1, neurons_att_2, dropout_rate):
        super().__init__()
        # Network configuration parameters
        self.n_classes = len(classes)
        self.neurons_1 = neurons_1 + (0 if only_images else len(clinical_classes))
        self.aggregation = aggregation
        
        # Build classifier based on input modality
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.neurons_1, neurons_2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(neurons_2, neurons_3),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(neurons_3, self.n_classes)
        )
        
        # MIL aggregation module
        self.mil_aggregation = MILAggregation(aggregation, neurons_att_1, neurons_att_2)

    def forward(self, features, clinical_data, region_info):
        """Process input through NMIL architecture."""
        if self.aggregation == 'attentionMIL':
            return self._process_with_attention(features, clinical_data, region_info)
        return self._process_basic(features, clinical_data)

    def _process_with_attention(self, features, clinical_data, region_info):
        """Handle attention-based MIL aggregation."""
        region_embeddings, instance_logits = [], []
        for region_id in range(int(region_info.max().item())+1):
            region_mask = (region_info == region_id)
            region_feats = features[region_mask]
            
            # Get region embedding and instance weights
            emb, weights = self.mil_aggregation(region_feats.squeeze())
            region_embeddings.append(emb)
            instance_logits.append(weights)
        
        # Aggregate regional features
        global_emb = torch.stack(region_embeddings).mean(dim=0)
        return self.classifier(global_emb), torch.cat(instance_logits)

    def _process_basic(self, features, clinical_data):
        """Handle basic aggregation (mean/max)."""
        agg_emb = self.mil_aggregation(features.squeeze())
        return self.classifier(agg_emb), None


class MILAggregation(torch.nn.Module):
    """Multiple Instance Learning aggregation module."""
    
    def __init__(self, aggregation, L, D):
        super().__init__()
        self.aggregation = aggregation
        if aggregation == 'attentionMIL':
            self.attention = AttentionModule(L, D)

    def forward(self, feats):
        """Apply selected aggregation method."""
        if self.aggregation == 'max':
            return feats.max(dim=0)[0]
        if self.aggregation == 'mean':
            return feats.mean(dim=0)
        return self.attention(feats)  # Attention-based aggregation


class AttentionModule(torch.nn.Module):
    """Attention mechanism for MIL (Ilse et al. 2018 implementation)."""
    
    def __init__(self, L, D):
        super().__init__()
        self.attention_V = torch.nn.Sequential(
            torch.nn.Linear(L, D), torch.nn.Tanh())
        self.attention_U = torch.nn.Sequential(
            torch.nn.Linear(L, D), torch.nn.Sigmoid())
        self.weights = torch.nn.Linear(D, 1)

    def forward(self, feats):
        """Compute attention-weighted features."""
        A_V = self.attention_V(feats)  # Attention features
        A_U = self.attention_U(feats)  # Gating mechanism
        attention = torch.softmax(self.weights(A_V * A_U), dim=0)
        return (feats * attention).sum(dim=0), attention.squeeze()
