"""GraphSAGE model for graph-based anomaly detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data, Batch
from typing import Dict, Any, List, Tuple
import numpy as np
from collections import defaultdict
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config

logger = setup_logger(__name__)


class GraphSAGEEncoder(nn.Module):
    """GraphSAGE encoder for learning node embeddings."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """Initialize GraphSAGE encoder."""
        super(GraphSAGEEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, output_dim))
        else:
            self.convs.append(SAGEConv(input_dim, output_dim))
        
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        """Forward pass through GraphSAGE layers."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class AnomalyDetector(nn.Module):
    """Anomaly detector using GraphSAGE embeddings."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        embedding_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """Initialize anomaly detector."""
        super(AnomalyDetector, self).__init__()
        
        # GraphSAGE encoder
        self.encoder = GraphSAGEEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Anomaly scoring network
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index):
        """Forward pass to compute anomaly scores."""
        # Get node embeddings
        embeddings = self.encoder(x, edge_index)
        
        # Compute anomaly scores
        scores = self.anomaly_scorer(embeddings)
        
        return embeddings, scores.squeeze()


class GraphSAGEModel:
    """GraphSAGE-based fraud detection model."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize GraphSAGE model."""
        if config is None:
            config = load_config()
        
        graph_config = config.get('graph', {})
        self.embedding_dim = graph_config.get('embedding_dim', 128)
        self.hidden_dim = graph_config.get('hidden_dim', 64)
        self.num_layers = graph_config.get('num_layers', 2)
        self.learning_rate = graph_config.get('learning_rate', 0.001)
        self.batch_size = graph_config.get('batch_size', 512)
        self.epochs = graph_config.get('epochs', 50)
        self.anomaly_threshold = graph_config.get('anomaly_threshold', 0.7)
        
        # Feature dimension (will be set when building graph)
        self.input_dim = None
        self.model = None
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initialized GraphSAGE model (device: {self.device})")
    
    def build_graph_data(
        self,
        graph_data: Dict[str, Any],
        node_features: Dict[str, torch.Tensor] = None
    ) -> Tuple[Data, List[str]]:
        """Build PyTorch Geometric Data object from graph data.
        
        Returns:
            Tuple of (Data object, list of node IDs in order)
        """
        nodes = graph_data['nodes']
        edges = graph_data['edges']
        
        # Create node to index mapping
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        num_nodes = len(nodes)
        
        # Build edge index
        edge_list = []
        edge_attrs = []
        
        for edge in edges:
            src_idx = node_to_idx.get(edge['src'])
            dst_idx = node_to_idx.get(edge['dst'])
            
            if src_idx is not None and dst_idx is not None:
                edge_list.append([src_idx, dst_idx])
                edge_attrs.append([edge.get('amount', 0.0)])
        
        if len(edge_list) == 0:
            # Create a dummy graph if no edges
            edge_list = [[0, 0]]
            edge_attrs = [[0.0]]
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        # Build node features
        if node_features is None:
            # Default features: one-hot encoding of node ID
            x = torch.eye(num_nodes)
            self.input_dim = num_nodes
        else:
            # Use provided features
            feature_list = []
            for node in nodes:
                if node in node_features:
                    feature_list.append(node_features[node])
                else:
                    # Default feature vector
                    feature_list.append(torch.zeros(self.input_dim or 10))
            x = torch.stack(feature_list)
            if self.input_dim is None:
                self.input_dim = x.shape[1]
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        return data, nodes
    
    def initialize_model(self):
        """Initialize the model architecture."""
        if self.input_dim is None:
            raise ValueError("Input dimension not set. Build graph data first.")
        
        self.model = AnomalyDetector(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
        
        logger.info(f"Initialized model with input_dim={self.input_dim}")
    
    def train_model(self, graph_data: Data, labels: torch.Tensor = None, node_ids: List[str] = None):
        """Train the GraphSAGE model."""
        if self.model is None:
            self.initialize_model()
        
        graph_data = graph_data.to(self.device)
        
        # If no labels provided, use unsupervised learning (reconstruction)
        if labels is None:
            # Self-supervised: predict node from neighbors
            labels = torch.ones(graph_data.x.size(0), device=self.device)
        
        self.model.train()
        
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            
            embeddings, scores = self.model(graph_data.x, graph_data.edge_index)
            
            # Reconstruction loss (simplified)
            loss = F.mse_loss(scores, labels)
            
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")
    
    def predict(self, graph_data: Data, node_ids: List[str] = None) -> Dict[str, float]:
        """Predict anomaly scores for nodes in the graph.
        
        Args:
            graph_data: PyTorch Geometric Data object
            node_ids: List of node IDs corresponding to graph nodes (optional)
        """
        if self.model is None:
            raise ValueError("Model not initialized. Train or load model first.")
        
        self.model.eval()
        graph_data = graph_data.to(self.device)
        
        with torch.no_grad():
            embeddings, scores = self.model(graph_data.x, graph_data.edge_index)
        
        # Convert to numpy
        scores_np = scores.cpu().numpy()
        
        # Map scores to node IDs
        node_scores = {}
        if node_ids is None:
            node_ids = [f"node_{i}" for i in range(len(scores_np))]
        
        for idx, node_id in enumerate(node_ids):
            node_scores[node_id] = float(scores_np[idx])
        
        return node_scores
    
    def detect_anomalies(
        self,
        graph_data: Dict[str, Any],
        node_features: Dict[str, torch.Tensor] = None
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in transaction graph."""
        # Build graph
        data, node_ids = self.build_graph_data(graph_data, node_features)
        
        # Initialize model if needed
        if self.model is None:
            self.initialize_model()
            # Quick training on the graph
            self.train_model(data, node_ids=node_ids)
        
        # Predict scores
        node_scores = self.predict(data, node_ids=node_ids)
        
        # Identify anomalies
        anomalies = []
        for node_id, score in node_scores.items():
            if score >= self.anomaly_threshold:
                anomalies.append({
                    "node_id": node_id,
                    "anomaly_score": score,
                    "is_anomaly": True
                })
        
        logger.info(f"Detected {len(anomalies)} anomalies out of {len(node_scores)} nodes")
        return anomalies
    
    def save_model(self, path: str):
        """Save model to file."""
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.input_dim,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.input_dim = checkpoint['input_dim']
        self.embedding_dim = checkpoint['embedding_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        self.num_layers = checkpoint['num_layers']
        
        self.initialize_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Model loaded from {path}")

