import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5  # 缩放因子

        
        self.q_linear = nn.Linear(dim, dim)  # Query
        self.k_linear = nn.Linear(dim, dim)  # Key
        self.v_linear = nn.Linear(dim, dim)  # Value
        self.gp_linear = nn.Linear(dim, dim)
        self.kp_linear = nn.Linear(dim, dim)
        self.norm=nn.LayerNorm(dim)
        

    def forward(self,g,g_p,W):
        
        W=W.squeeze(0)
        
        k_p = torch.matmul(F.softmax(torch.matmul(W, g.transpose(-1, -2)), dim=-1), g_p)
        k_p=self.kp_linear(k_p)
        g_p=self.gp_linear(g_p)
        Q = self.q_linear(g)+g_p
        K = self.k_linear(W)+k_p
        #Q = self.q_linear(g)
        #K = self.k_linear(W)
        V=self.v_linear(W)
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)  # [5000, x]
        output = torch.matmul(attention_weights, V)
        
        output=output+g
        output=self.norm(output)
        
        return output

class MLP1(nn.Module):
    def __init__(self, in_dim=1024, out_dim=128):
        super(MLP1, self).__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
       
        x = self.fc3(x)
        return x  

class MLP2(nn.Module):
    def __init__(self, in_dim=16, out_dim=128):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32 ,64)
        self.fc3 = nn.Linear(64, 128)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)
        return x 
    
class MLP3(nn.Module):
    def __init__(self, in_dim=3, out_dim=128):
        super(MLP3, self).__init__()
        self.fc1 = nn.Linear(in_dim, 16)
        self.fc2 = nn.Linear(16 ,64)
        self.fc3 = nn.Linear(64, out_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)
        return x    

class MLPFeatureConverter(nn.Module):
    def __init__(self, in_dim=512, out_dim=16):
        super(MLPFeatureConverter, self).__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, out_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP0(nn.Module):
    def __init__(self, in_dim=512, out_dim=16):
        super(MLP0, self).__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32 ,64)
        self.fc3 = nn.Linear(64, out_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)
        return x 

class InterObjectAttention(nn.Module):
    """物体间语义交互模块：让不同物体的referring features相互感知"""
    def __init__(self, dim=16, num_heads=4):
        super(InterObjectAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Multi-head attention for inter-object communication
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # Layer norm and residual connection
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )
        
        # Position encoder - properly initialized in __init__
        self.pos_encoder = nn.Linear(3, dim)
        
    def forward(self, object_features_list, object_positions):
        """
        Args:
            object_features_list: List[Tensor] - 每个物体的referring features, 每个tensor形状为[num_points_i, dim]
            object_positions: [num_objects, 3] - 每个物体的平均位置
        Returns:
            updated_features: List[Tensor] - 更新后的features列表
        """
        num_objects = len(object_features_list)
        if num_objects <= 1:
            return object_features_list, None
            
        dim = object_features_list[0].shape[1]
        
        # 强制类型转换，确保都是Python原生int
        num_objects = int(num_objects)
        max_neighbors = 4
        K = max_neighbors if num_objects > max_neighbors else num_objects - 1
        K = max(1, K)  # 至少有1个邻居
        
        # 计算物体间距离矩阵
        distances = torch.cdist(object_positions, object_positions, p=2)  # [num_objects, num_objects]
        
        # 初始化更新后的features (直接复制list中的每个tensor)
        updated_features = [feat.clone() for feat in object_features_list]

        all_attention_weights = {}
        
        # 为每个物体找到最近的K个邻居，并进行两两cross attention
        for i in range(num_objects):
            # 获取当前物体与其他物体的距离，排除自己
            current_distances = distances[i].clone()
            current_distances[i] = float('inf')  # 排除自己
            
            # 找到最近的K个邻居
            _, nearest_indices = torch.topk(current_distances, k=K, largest=False)
            
            current_object_features = object_features_list[i]  # [num_points_i, dim]
            
            # 对每个邻居进行cross attention
            for j_idx in nearest_indices:
                j = j_idx.item()
                neighbor_object_features = object_features_list[j]  # [num_points_j, dim]
                
                # 添加位置编码
                pos_encoding_i = self.pos_encoder(object_positions[i:i+1])  # [1, dim]
                pos_encoding_j = self.pos_encoder(object_positions[j:j+1])  # [1, dim]
                
                # Query来自当前物体，Key和Value来自邻居物体
                Q = current_object_features + pos_encoding_i  # [num_points_i, dim]
                Key = neighbor_object_features + pos_encoding_j  # [num_points_j, dim]
                Value = neighbor_object_features  # [num_points_j, dim]
                
                # Cross attention: 当前物体的所有点 attend to 邻居物体的所有点
                attention_scores = torch.matmul(Q, Key.transpose(0, 1)) / (dim ** 0.5)
                attention_weights = F.softmax(attention_scores, dim=-1)  # [num_points_i, num_points_j]
                
                # 加权融合邻居物体的features
                cross_attended_features = torch.matmul(attention_weights, Value)  # [num_points_i, dim]
                
                # 累积更新：使用较小的权重避免原始features被完全覆盖
                updated_features[i] += 0.2 * cross_attended_features
        
        # Layer norm保持训练稳定性
        for i in range(num_objects):
            updated_features[i] = self.norm1(updated_features[i])
        
        return updated_features, all_attention_weights
