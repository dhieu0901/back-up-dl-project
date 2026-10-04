# Chuẩn bị vấn đáp - Fresh and Rotten Fruit Classification

Mỗi câu hỏi kèm câu trả lời ngắn (nói trong 30 - 60 giây) và chỗ để tra cứu thêm.
Phần kết quả (mục G) sẽ được điền số liệu sau khi đánh giá trên tập test.

## A. Bài toán và dữ liệu

**1. Bài toán của nhóm là gì? Đầu vào, đầu ra?**
Đầu vào là một ảnh RGB 224x224 chụp một quả. Đầu ra gồm loại quả (8 lớp) và độ tươi (tươi / hỏng, kèm xác suất hỏng); tương đương 16 lớp kết hợp. Lớp dương tính là "hỏng" vì mục tiêu là phát hiện quả hỏng.

**2. Bộ dữ liệu có bao nhiêu ảnh, kích thước bao nhiêu, lấy ở đâu?**
FRUIT-16K trên Mendeley Data (DOI 10.17632/6ps7gtp2wg.1, CC BY 4.0): 16.000 ảnh, 8 loại quả x tươi/hỏng, mỗi lớp 1.000 ảnh, tất cả 224x224 RGB, khoảng 6 KB mỗi ảnh.

**3. "Làm sạch dữ liệu" của nhóm cụ thể là làm gì?**
Đọc thử 100 % ảnh (không có ảnh lỗi), kiểm tra kích thước và kênh màu, tính mã băm SHA-256 để tìm ảnh trùng: có 1.273 bản sao y hệt (luôn là cặp ảnh liền nhau). Dùng perceptual hash để tìm ảnh mâu thuẫn nhãn: `F_Banana/1.jpg` chính là `S_Banana/6.jpg`. Loại các ảnh này, còn 14.726 ảnh. Báo cáo chi tiết: `reports/data_quality_report.csv`.

**4. Tại sao không chia ngẫu nhiên train/val/test như bình thường?**
Ảnh được chụp theo loạt, các ảnh liền nhau gần như giống hệt. Chia ngẫu nhiên thì ảnh gần giống nhau rơi vào cả train và test: 9,7 % ảnh test có bản sao gần giống (pHash <= 6) trong train - kết quả test sẽ bị thổi phồng vì mô hình chỉ cần "nhớ". Nhóm gom 25 ảnh liên tiếp (và các ảnh gần trùng) thành một nhóm, đưa cả nhóm vào cùng một tập; tỉ lệ trên giảm còn 1,2 %.

**5. Perceptual hash (pHash) là gì?**
Thu nhỏ ảnh xám về 32x32, biến đổi cosin rời rạc (DCT), giữ 8x8 hệ số tần số thấp, so với trung vị để được 64 bit. Hai ảnh giống nhau về cấu trúc có pHash khác nhau ít bit (khoảng cách Hamming nhỏ); ảnh không liên quan khác khoảng 30/64 bit. Khác SHA-256 (chỉ bắt được ảnh trùng từng byte), pHash bắt được ảnh gần giống.

**6. Tỉ lệ 70/15/15 và vai trò từng tập?**
Train 10.320 ảnh để học trọng số; validation 2.203 ảnh để dừng sớm, chọn epoch tốt nhất, giảm learning rate và so sánh các thí nghiệm ablation; test 2.203 ảnh chỉ dùng một lần ở cuối. Chia phân tầng theo từng lớp trong 16 lớp nên tỉ lệ mỗi lớp gần như giống nhau ở ba tập.

**7. Dữ liệu có mất cân bằng không? Xử lý thế nào?**
Sau khi bỏ ảnh trùng, lớp ít nhất (Lemon_fresh) còn 555 ảnh, lớp nhiều nhất 1.000 ảnh - mất cân bằng nhẹ (khoảng 1:1,8). Nhóm dùng chia phân tầng và báo cáo chỉ số macro (precision/recall/F1 trung bình không trọng số theo lớp) để lớp ít mẫu được coi trọng như lớp nhiều mẫu.

## B. Tiền xử lý và tăng cường dữ liệu

**8. Chuẩn hoá ảnh như thế nào? Tại sao đặt trong mô hình?**
Mô hình 1, 2: chia 255 về [0, 1]. Mô hình 3: x/127,5 - 1 về [-1, 1] vì MobileNetV2 được huấn luyện trên ImageNet với miền giá trị này. Đặt chuẩn hoá thành lớp đầu tiên của mô hình để mô hình đã lưu nhận trực tiếp ảnh gốc, tránh quên hoặc làm sai tiền xử lý khi triển khai.

**9. Ví dụ code của cô chia 255 cho MobileNetV2, sao nhóm lại khác?**
`preprocess_input` của MobileNetV2 đưa ảnh về [-1, 1]. Nếu đưa ảnh [0, 1] vào, phân phối đầu vào khác hẳn lúc huấn luyện trước, đặc trưng ImageNet kém hiệu quả hơn. Nhóm dùng đúng miền giá trị của mạng huấn luyện sẵn.

**10. Tăng cường dữ liệu gồm những phép nào? Tại sao không đổi màu?**
Lật ngang/dọc, xoay tối đa +-36 độ, zoom +-15 %, độ sáng +-15 %, độ tương phản +-15 %, sinh ngẫu nhiên mỗi epoch, chỉ cho tập train. Không đổi sắc độ (hue) vì màu nâu, đốm thâm, nấm mốc là dấu hiệu chính của quả hỏng - đổi màu có thể làm sai nhãn.

**11. Tại sao val/test không tăng cường?**
Val/test phải phản ánh ảnh thật để đánh giá trung thực và so sánh công bằng giữa các epoch / mô hình. Notebook `01_data_validation` kiểm tra rằng đọc tập val hai lần cho kết quả giống hệt.

## C. Mô hình 1 - CNN tuần tự đơn giản

**12. Mô tả kiến trúc Mô hình 1.**
Sequential API: 5 khối [Conv 3x3 - ReLU - MaxPool 2x2] với 32-64-128-128-256 kênh (224 -> 7), Flatten 12.544, Dense 256 ReLU, Dropout 0,5, Dense 16 softmax. 3,75 triệu tham số, 86 % nằm ở lớp Dense sau Flatten.

**13. Tính số tham số của lớp Conv đầu tiên.**
(3 x 3 x 3 + 1) x 32 = 896. Tổng quát (k_h k_w c_in + 1) c_out, không phụ thuộc kích thước ảnh nhờ chia sẻ trọng số.

**14. Tính kích thước đầu ra sau lớp Conv 3x3 padding "same" rồi MaxPool 2x2?**
Padding same giữ 224x224; MaxPool 2x2 stride 2 giảm một nửa: 112x112, số kênh bằng số kernel (32).

**15. Mô hình 1 dự đoán 16 lớp, làm sao so sánh với mô hình 2 đầu ra?**
Từ 16 xác suất: P(quả) = P(quả, tươi) + P(quả, hỏng); P(hỏng) = tổng 8 xác suất lớp hỏng. Nhờ vậy cả ba mô hình được đánh giá trên cùng ba góc nhìn: loại quả, độ tươi và nhãn kết hợp (đúng cả hai).

## D. Mô hình 2 - CNN phức tạp đa nhiệm

**16. Mô hình 2 "phức tạp" ở chỗ nào so với Mô hình 1?**
Dùng Functional API; thân mạng ghép từ 4 khối phần dư kiểu ResNet có Batch Normalization; dùng Global Average Pooling thay Flatten + Dense; rẽ thành hai đầu ra song song (loại quả softmax 8 lớp, độ tươi sigmoid). 4,99 triệu tham số.

**17. Khối phần dư (residual block) hoạt động thế nào? Tại sao dùng?**
Đầu ra = g(x) + x: các lớp chỉ cần học phần dư g(x) = f(x) - x; nếu ánh xạ tốt nhất gần ánh xạ đồng nhất thì chỉ cần đẩy g về 0. Đường tắt giúp gradient truyền thẳng về lớp đầu, giảm triệt tiêu gradient, huấn luyện mạng sâu ổn định. Khi đổi số kênh/kích thước, đường tắt dùng Conv 1x1 (stride 2) + BN.

**18. Batch Normalization làm gì? Khi suy luận thì sao?**
Chuẩn hoá mỗi kênh theo trung bình, phương sai của mini-batch rồi nhân gamma, cộng beta (học được). Giúp hội tụ nhanh, ổn định, cho phép learning rate lớn hơn. Khi suy luận dùng trung bình/phương sai trượt tích luỹ lúc huấn luyện (đó là 5.824 tham số "non-trainable" của Mô hình 2).

**19. Học đa nhiệm là gì? Hàm mất mát của Mô hình 2?**
Một thân chung, nhiều đầu ra cho nhiều nhiệm vụ liên quan. Loss = 1,0 x cross-entropy (loại quả) + 1,0 x binary cross-entropy (độ tươi). Hai nhiệm vụ dùng chung đặc trưng hình dạng, màu, kết cấu; ít tham số hơn hai mô hình riêng.

**20. Tại sao đầu ra độ tươi dùng sigmoid chứ không phải softmax 2 lớp?**
Bài toán nhị phân chỉ cần một xác suất P(hỏng); sigmoid + binary cross-entropy tương đương softmax 2 lớp + cross-entropy nhưng gọn hơn. Ngưỡng quyết định 0,5.

**21. Global Average Pooling khác Flatten thế nào?**
GAP lấy trung bình mỗi kênh của bản đồ 7x7x512 thành vector 512 chiều; Flatten giữ toàn bộ 7x7x512 = 25.088 giá trị. GAP giảm mạnh số tham số của lớp Dense tiếp theo và ít quá khớp hơn (ý tưởng của NiN).

## E. Mô hình 3 - Học chuyển giao

**22. Transfer learning là gì? Tại sao dùng MobileNetV2?**
Dùng lại tri thức của mô hình huấn luyện trên ImageNet (1,28 triệu ảnh, 1.000 lớp). Các lớp đầu học đặc trưng tổng quát (cạnh, màu, kết cấu) dùng được cho ảnh trái cây. MobileNetV2 nhẹ (2,26 triệu tham số phần nền), nhanh trên CPU, là mạng dùng trong ví dụ của môn học.

**23. Hai giai đoạn huấn luyện của Mô hình 3?**
Giai đoạn 1: đóng băng toàn bộ mạng nền, chỉ học hai đầu ra (247 nghìn tham số), learning rate 1e-3. Giai đoạn 2: mở băng từ block 13 (38 lớp, 1,91 triệu tham số), learning rate 1e-4 nhỏ hơn 10 lần để không phá hỏng trọng số đã học.

**24. Tại sao phải huấn luyện đầu ra trước rồi mới tinh chỉnh?**
Đầu ra mới khởi tạo ngẫu nhiên tạo gradient lớn; nếu mở băng ngay, gradient này truyền vào mạng nền và phá hỏng đặc trưng ImageNet. Học đầu ra trước cho đến khi ổn định rồi mới tinh chỉnh với learning rate nhỏ.

**25. Tại sao giữ Batch Normalization đóng băng khi tinh chỉnh?**
Thống kê trung bình/phương sai của BN đã được ước lượng trên ImageNet rất ổn định; cập nhật chúng bằng các batch 32 ảnh của tập nhỏ sẽ nhiễu và làm giảm độ chính xác. Keras chạy BN ở chế độ suy luận khi `trainable = False`.

**26. Tại sao không mở băng toàn bộ mạng?**
Các khối đầu chứa đặc trưng tổng quát, ít cần thay đổi; mở hết làm tăng số tham số cần học (dễ quá khớp với 10 nghìn ảnh) và tốn thời gian. Thí nghiệm ablation so sánh mở từ: không mở, block 16, 13, 10, 6, toàn bộ - xem kết quả ở mục G.

**27. MobileNetV2 khác CNN thường ở đâu?**
Dùng tích chập tách biệt theo chiều sâu (depthwise 3x3 từng kênh + pointwise 1x1) giảm chi phí khoảng 9 lần; khối phần dư đảo ngược: 1x1 mở rộng kênh (x6) - depthwise 3x3 - 1x1 thu hẹp tuyến tính, đường tắt nối các đầu hẹp.

## F. Huấn luyện

**28. Dùng optimizer, learning rate, batch size nào? Tại sao?**
Adam (momentum + RMSProp, tự điều chỉnh learning rate từng tham số), learning rate 1e-3 cho mô hình học từ đầu và đầu ra mới, 1e-4 khi tinh chỉnh; batch 32 (mini-batch gradient descent cân bằng giữa độ ổn định và tốc độ).

**29. Các callback dùng để làm gì?**
ModelCheckpoint lưu mô hình có val_loss thấp nhất; EarlyStopping dừng sau 8 epoch không cải thiện và khôi phục trọng số tốt nhất; ReduceLROnPlateau giảm learning rate 0,3 lần sau 3 epoch không cải thiện. Cả ba mô hình dùng cùng cấu hình.

**30. Làm sao biết mô hình bị quá khớp?**
So sánh đường loss/accuracy của train và validation theo epoch (`figures/learning_curves/`): train tiếp tục tốt lên còn validation đi ngang hoặc xấu đi là quá khớp. Dropout, tăng cường dữ liệu, dừng sớm và BN là các biện pháp chống quá khớp đã dùng.

**31. Dropout hoạt động thế nào?**
Mỗi bước huấn luyện, ngẫu nhiên tắt một tỉ lệ nơ-ron (0,5 ở Mô hình 1, 0,3 ở hai đầu ra của Mô hình 2, 3), buộc mạng không phụ thuộc vài nơ-ron cụ thể; khi suy luận dropout tắt.

**32. Huấn luyện trên phần cứng gì, mất bao lâu?**
CPU Intel i9-13900H, TensorFlow 2.21 (TensorFlow trên Windows không hỗ trợ GPU). Thời gian từng mô hình xem `reports/experiment_log.csv`.

## G. Kết quả và đánh giá (điền sau khi có kết quả test)

**33. Precision, recall, F1 khác nhau thế nào? Tại sao báo cáo macro F1?**
Precision = TP/(TP+FP): trong các ảnh dự đoán là hỏng, bao nhiêu thực sự hỏng. Recall = TP/(TP+FN): trong các quả hỏng, mô hình bắt được bao nhiêu. F1 là trung bình điều hoà. Macro = trung bình không trọng số theo lớp, phù hợp khi lớp mất cân bằng nhẹ.

**34. Mô hình nào tốt nhất? Tốt hơn bao nhiêu?** - *điền từ `reports/results_summary.md`*

**35. Tăng cường dữ liệu có giúp không?** - *điền từ bảng ablation*

**36. Mở băng bao nhiêu lớp là tốt nhất?** - *điền từ bảng ablation*

**37. Mô hình hay nhầm ở đâu?** - *điền từ ma trận nhầm lẫn và `figures/misclassified_samples.png`*

**38. Grad-CAM cho thấy điều gì?**
Grad-CAM lấy gradient của điểm số quyết định (log-odds của xác suất hỏng) theo bản đồ đặc trưng lớp tích chập cuối, lấy trung bình theo không gian làm trọng số mỗi kênh, cộng có trọng số các bản đồ đặc trưng rồi qua ReLU. Vùng sáng là vùng ủng hộ quyết định. *Điền nhận xét từ `figures/gradcam/`.*

**39. Thời gian suy luận?** - *điền từ bảng so sánh (ms/ảnh trên CPU)*

## H. Hạn chế và hướng phát triển

**40. Hạn chế của đề tài?**
Ảnh chụp trong điều kiện phòng thí nghiệm (một quả, nền đơn giản), chỉ 8 loại quả; ảnh nén mạnh, kích thước nhỏ; chưa kiểm tra trên ảnh thực tế băng chuyền; nhãn tươi/hỏng nhị phân (thực tế có nhiều mức độ chín/hỏng).

**41. Hướng phát triển?**
Thu thêm ảnh thực tế nhiều quả trong một ảnh (bài toán phát hiện đối tượng - chương Object Detection), phân loại nhiều mức độ tươi, thử các backbone khác (EfficientNet), lượng tử hoá mô hình để chạy trên thiết bị nhúng.
