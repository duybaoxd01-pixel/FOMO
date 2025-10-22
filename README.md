| Thành phần          | Mô tả cụ thể                                            |
| ------------------- | ------------------------------------------------------- |
| **Backbone**        | MobileNetV2 / EfficientNet-Lite (CNN trích đặc trưng)   |
| **Head**            | Conv 1×1 → tạo heatmap có kích thước nhỏ (ví dụ 16×16)  |
| **Post-processing** | Tìm điểm cực đại trên heatmap để xác định tâm đối tượng |
| **Loss Function**   | Binary Cross Entropy (BCE) hoặc Focal Loss              |
| **Optimizer**       | Adam hoặc SGD                                           |
| **Input Data**      | Ảnh + tọa độ tâm (centroid) thay vì bounding box        |
