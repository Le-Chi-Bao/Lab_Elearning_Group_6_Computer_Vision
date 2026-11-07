# Lab 3 — Canny Edge Detector

Giải bài thực hành chương 2 (phần 2): lý thuyết, triển khai OpenCV/scikit-image, minh họa từng bước, khảo sát tham số, ảnh nhiễu/ít tương phản/nhiều chi tiết, so sánh Sobel/Laplacian, tìm vùng, nhận dạng hình dạng và video.

## Chạy

Python 3.10 trở lên, terminal tại `Lab_3`:

```powershell
python -m pip install -r requirements.txt
python canny_lab.py
python -m unittest -v
python canny_lab.py --input images/sample.jpg --sigma 1.5 --low 30 --high 90
python canny_lab.py --video input.mp4 --output outputs/video
```

Mở `Canny_Flow.ipynb`, chọn kernel đã cài requirements, chạy **Run All**. Notebook dùng lại `canny_lab.py`. Ảnh mẫu lấy từ Lab 1; không tải dataset ngoài. Kết quả ảnh, biểu đồ và benchmark được lưu vào `outputs/`.

## Các bước thuật toán

1. **Gaussian** giảm nhiễu trước đạo hàm. Sigma lớn giảm cạnh nhiễu nhưng có thể mất chi tiết và nhập các cạnh gần nhau.
2. **Gradient**: Sobel tính Gx, Gy; độ lớn sqrt(Gx²+Gy²), hướng atan2(Gy,Gx).
3. **Non-maximum suppression (NMS)**: chỉ giữ cực đại cục bộ theo hướng gradient để làm mảnh cạnh. Bản giáo dục lượng tử hóa thành bốn hướng; khác với các bản dùng nội suy.
4. **Ngưỡng kép**: >= high là cạnh mạnh; từ low đến high là ứng viên yếu; dưới low bị loại.
5. **Hysteresis**: giữ ứng viên yếu khi nối với cạnh mạnh qua đường liên thông 8 hướng; loại các thành phần không có cạnh mạnh. Không chỉ kiểm tra một pixel lân cận duy nhất.

`manual_canny` xuất tất cả bước trung gian, không gọi `cv2.Canny`. Sobel/Gaussian và connected components dùng OpenCV. Bản này phục vụ học thuật, không kỳ vọng trùng từng pixel với thư viện.

## Tham số và so sánh thư viện

OpenCV: Gaussian được áp dụng rõ ràng trước Canny, Sobel 3×3, L2gradient=True. scikit-image: đầu vào float [0,1], ngưỡng chia 255, Gaussian bên trong, mode reflect. Dù cùng sigma và giá trị ngưỡng quy đổi, các chi tiết chuẩn hóa đạo hàm, biên, NMS và làm tròn vẫn khác; không coi hai cấu hình tương đương hoàn toàn.

OpenCV yêu cầu truyền threshold1/threshold2, không có cặp ngưỡng mặc định của API. Project chọn baseline **sigma=1, low=50, high=100**. Notebook so sánh riêng `skimage.feature.canny(gray/255)` với mặc định thực của scikit-image: sigma=1, low=0.1, high=0.2 khi không dùng quantile.

Hạ ngưỡng thường tăng độ nhạy và cạnh giả; tăng ngưỡng có thể làm đứt hoặc mất cạnh yếu. Tăng sigma không luôn cải thiện kết quả. Notebook thay từng yếu tố và thể hiện số pixel cạnh; con số này không phải chỉ số chất lượng.

## So sánh và đánh giá

| Phương pháp | Đặc điểm | Giới hạn |
| --- | --- | --- |
| Sobel | Đạo hàm bậc một, tính gradient đơn giản | Cạnh có thể dày; cần chọn ngưỡng |
| Laplacian | Đạo hàm bậc hai, không ưu tiên hướng | Nhạy nhiễu; ngưỡng độ lớn có thể tạo biên kép |
| Canny | Gaussian, NMS, ngưỡng kép, liên kết cạnh | Nhiều bước và tham số, có thể mất cạnh yếu |

Benchmark dùng ảnh hình học có biên tham chiếu tạo trực tiếp từ tọa độ, độc lập với bộ phát hiện cạnh. Báo cáo precision/recall/F1 với dung sai Euclid 2 pixel: precision đếm pixel dự đoán gần biên thật; recall đếm biên thật gần dự đoán. Đây là phép ghép nhiều-với-nhiều, không phải matching một-một; cạnh dày vẫn có thể được ưu ái trong vùng dung sai. Ảnh thật không có nhãn nên không báo độ chính xác giả định.

Đo median thời gian của 5 lần sau warm-up, gồm bước lọc, không gồm đọc ảnh/vẽ hình. Không suy rộng thứ hạng tốc độ giữa các máy. Ngưỡng Sobel=100, Laplacian=40 là baseline minh họa, chưa được tối ưu công bằng cho mỗi phương pháp; Laplacian dùng ngưỡng độ lớn, không triển khai zero crossing. Muốn kết luận chính xác cần dữ liệu có annotation (có thể nhiều người gán nhãn), chia validation/test và chọn tham số trên validation. Pixel accuracy không phù hợp khi nền chiếm đa số.

## Kết hợp và câu hỏi mở rộng

- **Tìm vùng:** Canny → closing 3×3 → contour ngoài → tô vùng. Đây là phân đoạn dựa trên đường biên đóng; có thể thất bại nếu cạnh hở hoặc dính nhau. Có thể cải tiến bằng watershed/marker, region growing hoặc kết hợp màu.
- **Hình dạng:** approxPolyDP và circularity nhận diện tam giác, tứ giác, hình tròn trên ảnh tổng hợp. Đây là quy tắc hình học đơn giản, không phải bộ nhận diện đối tượng tổng quát.
- **Cải thiện:** điều chỉnh sigma/ngưỡng theo validation, tiền xử lý giảm nhiễu phù hợp, CLAHE cho tương phản thấp, ROI/resize để giảm chi phí, hoặc ngưỡng thích nghi. Hậu xử lý có thể mất chi tiết, cần đánh giá lại.
- **Ảnh màu:** có thể chuyển grayscale hoặc chạy từng kênh và OR các bản đồ cạnh. Cách OR giữ một số biên màu có cùng độ sáng nhưng cũng tăng cạnh giả; cần xử lý màu hoặc gradient vector cho bài toán nghiêm túc.
- **Video:** xử lý từng frame, giữ kích thước và FPS nguồn, giải phóng capture/writer bằng finally. Ngưỡng cố định có thể gây nhấp nháy khi ánh sáng thay đổi; cân nhắc làm mượt theo thời gian, theo dõi cạnh và đo FPS end-to-end. CLI kiểm tra lỗi mở video/writer.

Ứng dụng: tiền xử lý phân tích biên vật thể, kiểm tra hình học sản phẩm, trích đường kẻ tài liệu, và tạo ứng viên đường thẳng trong ảnh giao thông. Canny không tự hiểu ngữ nghĩa và không tự nhận diện đối tượng. Không khẳng định lĩnh vực nào dùng nhiều nhất khi chưa có khảo sát định lượng.
