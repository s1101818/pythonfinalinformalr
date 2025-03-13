import torch
import torch_cluster
import torch_scatter
import torch_sparse
import torch_geometric

def test_torch_cluster():
    try:
        # 創建測試數據
        x = torch.rand((100, 3))
        
        # 測試不同的cluster函數
        # 1. KNN Graph
        edge_index = torch_cluster.knn_graph(x, k=5)
        print("KNN Graph test passed")
        
        # 2. Radius Graph
        edge_index = torch_cluster.radius_graph(x, r=0.5)
        print("Radius Graph test passed")
        
        # 3. Grid Cluster
        # 修改：將 size 從 float 改為 tensor
        size = torch.tensor([0.2, 0.2, 0.2])  # 對應x的3個維度
        cluster = torch_cluster.grid_cluster(x, size=size)
        print("Grid Cluster test passed")
        
        print("\nAll tests passed! torch_cluster is working correctly!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    test_torch_cluster()