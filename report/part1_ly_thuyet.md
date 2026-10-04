# PHẦN 1. CƠ SỞ LÝ THUYẾT VỀ MẠNG NƠ-RON TÍCH CHẬP (CNN)

## 1.1. Từ mạng nơ-ron truyền thẳng đến mạng nơ-ron tích chập

**Nơ-ron nhân tạo.** Một nơ-ron nhận vector đầu vào $\mathbf{x}$, tính tổng có trọng số rồi đưa qua hàm kích hoạt $f$:

$$z = \mathbf{w}^T\mathbf{x} + b, \qquad a = f(z).$$

Các hàm kích hoạt thường dùng là sigmoid $\sigma(z) = \frac{1}{1+e^{-z}}$, tanh và ReLU $f(z) = \max(0, z)$. Sigmoid có đạo hàm $\sigma'(z) = \sigma(z)(1-\sigma(z))$ tiến về 0 khi $|z|$ lớn nên dễ gây hiện tượng triệt tiêu gradient (vanishing gradient); ReLU có đạo hàm bằng 1 với $z > 0$, tính toán nhanh và là lựa chọn mặc định cho các lớp ẩn của CNN hiện đại.

**Mạng nhiều lớp (MLP).** Các nơ-ron được xếp thành lớp; với lớp thứ $l$:

$$\mathbf{z}^{(l)} = \mathbf{W}^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}, \qquad \mathbf{a}^{(l)} = f\left(\mathbf{z}^{(l)}\right), \qquad \mathbf{a}^{(0)} = \mathbf{x}.$$

Toàn bộ trọng số được học bằng cách tối thiểu hoá hàm mất mát $J$ với thuật toán gradient descent, trong đó gradient của $J$ theo từng $\mathbf{W}^{(l)}, \mathbf{b}^{(l)}$ được tính bằng lan truyền ngược (backpropagation) dựa trên quy tắc chuỗi:

$$\boldsymbol{\delta}^{(L)} = \frac{\partial J}{\partial \mathbf{a}^{(L)}} \odot f'\left(\mathbf{z}^{(L)}\right), \quad
\boldsymbol{\delta}^{(l)} = \left(\mathbf{W}^{(l+1)}\right)^T \boldsymbol{\delta}^{(l+1)} \odot f'\left(\mathbf{z}^{(l)}\right), \quad
\frac{\partial J}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} \left(\mathbf{a}^{(l-1)}\right)^T.$$

**Hạn chế của MLP với dữ liệu ảnh.** Một ảnh màu $224 \times 224 \times 3$ có 150.528 giá trị; chỉ một lớp ẩn 1.000 nơ-ron đã cần khoảng 150 triệu trọng số. MLP còn bỏ qua cấu trúc không gian của ảnh (các điểm ảnh gần nhau có liên quan mật thiết) và phải học lại cùng một mẫu (ví dụ một vết thâm trên vỏ quả) ở mọi vị trí. Mạng nơ-ron tích chập khắc phục điều này bằng ba ý tưởng: **kết nối cục bộ** (mỗi nơ-ron chỉ nhìn một vùng nhỏ), **chia sẻ trọng số** (cùng một bộ lọc được trượt trên toàn ảnh) và **gộp (pooling)** để giảm kích thước. Một CNN điển hình gồm: Ảnh đầu vào → [Lớp tích chập + Lớp gộp] × n → Lớp kết nối đầy đủ → Đầu ra.

## 1.2. Lớp tích chập (convolutional layer)

**Phép tương quan chéo (cross-correlation).** Lớp tích chập trượt một mảng hạt nhân (kernel) $\mathbf{K}$ kích thước $k_h \times k_w$ trên mảng đầu vào $\mathbf{X}$; tại mỗi vị trí, nhân từng phần tử rồi cộng lại:

$$Y[i, j] = \sum_{a=0}^{k_h-1}\sum_{b=0}^{k_w-1} X[i+a,\, j+b]\, K[a, b] + b_0.$$

Ví dụ với đầu vào $3 \times 3$ gồm các số 0…8 và kernel $\begin{bmatrix}0 & 1\\ 2 & 3\end{bmatrix}$: phần tử đầu tiên của đầu ra là $0 \cdot 0 + 1 \cdot 1 + 3 \cdot 2 + 4 \cdot 3 = 19$, và toàn bộ đầu ra là $\begin{bmatrix}19 & 25\\ 37 & 43\end{bmatrix}$. Giá trị của kernel không được thiết kế thủ công mà được học từ dữ liệu bằng gradient descent; các kernel ở lớp đầu thường học được các bộ phát hiện cạnh, màu sắc, kết cấu, còn các lớp sâu hơn kết hợp chúng thành các mẫu phức tạp (hình dạng quả, vết thối, nấm mốc). Mảng đầu ra được gọi là bản đồ đặc trưng (feature map).

**Kích thước đầu ra, đệm (padding) và bước trượt (stride).** Với đầu vào $n_h \times n_w$ và kernel $k_h \times k_w$, đầu ra có kích thước $(n_h - k_h + 1) \times (n_w - k_w + 1)$. Đệm thêm tổng cộng $p_h$ hàng và $p_w$ cột (thường là số 0) quanh viền, và trượt với bước $s_h, s_w$, kích thước đầu ra là

$$\left\lfloor \frac{n_h - k_h + p_h + s_h}{s_h} \right\rfloor \times \left\lfloor \frac{n_w - k_w + p_w + s_w}{s_w} \right\rfloor.$$

Chọn $p_h = k_h - 1$, $p_w = k_w - 1$ (đệm "same") và $s = 1$ giữ nguyên chiều cao, chiều rộng; stride 2 giảm một nửa kích thước.

**Nhiều kênh đầu vào và đầu ra.** Ảnh màu có $c_i = 3$ kênh (R, G, B). Khi đó mỗi kernel có kích thước $c_i \times k_h \times k_w$; phép tương quan chéo được thực hiện trên từng kênh rồi cộng kết quả các kênh lại. Để tạo $c_o$ kênh đầu ra, lớp dùng $c_o$ kernel như vậy. Số tham số của một lớp tích chập là $(k_h k_w c_i + 1)\, c_o$ - **không phụ thuộc kích thước ảnh**, nhờ chia sẻ trọng số. Ví dụ lớp Conv $3 \times 3$ từ 3 lên 32 kênh có $(3 \cdot 3 \cdot 3 + 1) \cdot 32 = 896$ tham số (lớp đầu tiên của Mô hình 1).

**Tích chập $1 \times 1$.** Kernel $1 \times 1$ không nhìn lân cận không gian mà chỉ kết hợp các kênh tại từng điểm ảnh - tương đương một lớp kết nối đầy đủ áp dụng độc lập cho mỗi vị trí. Nó được dùng để thay đổi số kênh và thêm tính phi tuyến (NiN, GoogLeNet, ResNet, MobileNetV2).

**Vùng tiếp nhận (receptive field).** Mỗi phần tử ở lớp sâu phụ thuộc vào một vùng ngày càng lớn của ảnh gốc khi các lớp tích chập và gộp được xếp chồng; nhờ vậy mạng học được đặc trưng từ cục bộ (cạnh) đến toàn cục (cả quả).

## 1.3. Lớp gộp (pooling layer)

Lớp gộp thường đặt giữa các lớp tích chập, có hai tác dụng: (1) giảm độ nhạy của biểu diễn đối với vị trí chính xác của đặc trưng và (2) giảm kích thước biểu diễn, do đó giảm chi phí tính toán của các lớp sau. Cửa sổ gộp $p \times q$ trượt trên đầu vào như kernel nhưng **không có tham số**:

* **Gộp cực đại (max pooling)** lấy giá trị lớn nhất trong cửa sổ - với đầu vào 0…8 và cửa sổ $2 \times 2$ ta được $\max(0,1,3,4)=4$, $\max(1,2,4,5)=5$, $\max(3,4,6,7)=7$, $\max(4,5,7,8)=8$;
* **Gộp trung bình (average pooling)** lấy giá trị trung bình trong cửa sổ.

Đệm và bước trượt áp dụng như lớp tích chập; phổ biến nhất là cửa sổ $2 \times 2$, stride 2, giảm một nửa chiều cao và chiều rộng ($224 \times 224 \times 64 \rightarrow 112 \times 112 \times 64$). Với đầu vào nhiều kênh, lớp gộp xử lý **từng kênh riêng biệt**, nên số kênh đầu ra bằng số kênh đầu vào.

**Gộp trung bình toàn cục (global average pooling - GAP)** lấy trung bình toàn bộ bản đồ đặc trưng của mỗi kênh, biến tensor $H \times W \times D$ thành vector $D$ chiều. GAP được NiN đề xuất và dùng trong GoogLeNet, ResNet, MobileNetV2 vì giảm mạnh số tham số so với Flatten + Dense và ít bị quá khớp hơn.

## 1.4. Lớp kết nối đầy đủ và lớp đầu ra

Sau các khối tích chập - gộp, tensor đầu ra $H \times W \times D$ được duỗi phẳng (flatten) thành vector $H \cdot W \cdot D$ chiều (hoặc gộp bằng GAP), rồi đi qua các lớp kết nối đầy đủ (fully connected, Dense) - thực chất là một MLP - để kết hợp các đặc trưng và tính đầu ra.

**Phân loại nhiều lớp - softmax và cross-entropy.** Với $C$ lớp, lớp đầu ra tính $\mathbf{z} = \mathbf{W}^T\mathbf{x} + \mathbf{b}$ rồi chuẩn hoá thành phân phối xác suất bằng hàm softmax:

$$a_i = \frac{\exp(z_i)}{\sum_{j=1}^{C}\exp(z_j)}, \qquad i = 1, \dots, C.$$

Hàm mất mát là cross-entropy giữa nhãn one-hot $\mathbf{y}$ và dự đoán $\mathbf{a}$; với nhãn đúng $y$ của mẫu thì

$$J = -\sum_{j=1}^{C} y_j \log a_j = -\log a_y, \qquad \frac{\partial J}{\partial \mathbf{z}} = \mathbf{a} - \mathbf{y}.$$

Trong Keras, `SparseCategoricalCrossentropy` là cùng hàm mất mát này nhưng nhận nhãn dạng số nguyên thay vì one-hot.

**Phân loại nhị phân - sigmoid và binary cross-entropy.** Với hai lớp (ví dụ tươi / hỏng), chỉ cần một nơ-ron đầu ra $\hat{y} = \sigma(z) = P(y = 1 \mid \mathbf{x})$ và hàm mất mát

$$J = -\left[\, y\log\hat{y} + (1-y)\log(1-\hat{y}) \,\right],$$

chính là hàm log-likelihood âm của hồi quy logistic. Dự đoán nhãn 1 khi $\hat{y} \ge 0{,}5$.

## 1.5. Huấn luyện CNN

**Gradient descent và các biến thể.** Tham số được cập nhật ngược hướng gradient: $\mathbf{w} \leftarrow \mathbf{w} - \eta\nabla J(\mathbf{w})$ với tốc độ học (learning rate) $\eta$. Batch gradient descent dùng toàn bộ tập huấn luyện cho mỗi bước (chính xác nhưng chậm); stochastic gradient descent dùng một mẫu (rẻ nhưng nhiễu); **mini-batch gradient descent** dùng một nhóm nhỏ $B$ mẫu (ví dụ 32) - cân bằng giữa độ ổn định và tốc độ, tận dụng tính toán ma trận, và là cách huấn luyện mặc định:

$$\nabla J_B(\mathbf{w}) = \frac{1}{|B|}\sum_{(\mathbf{x},y) \in B}\nabla J_{\mathbf{x}y}(\mathbf{w}).$$

Tốc độ học quá nhỏ làm hội tụ chậm; quá lớn làm hàm mất mát dao động hoặc phân kỳ.

**Adam** (Kingma & Ba, 2015) kết hợp quán tính (momentum) và chuẩn hoá theo độ lớn gradient (RMSProp), tự điều chỉnh tốc độ học cho từng tham số:

$$\mathbf{m}_t = \beta_1\mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t, \quad \mathbf{v}_t = \beta_2\mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2, \quad
\mathbf{w}_t = \mathbf{w}_{t-1} - \eta\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon},$$

với $\hat{\mathbf{m}}_t = \mathbf{m}_t/(1-\beta_1^t)$, $\hat{\mathbf{v}}_t = \mathbf{v}_t/(1-\beta_2^t)$ (thường $\beta_1 = 0{,}9$, $\beta_2 = 0{,}999$).

**Triệt tiêu gradient và cách khắc phục.** Trong mạng sâu $o = f_L \circ f_{L-1} \circ \dots \circ f_1(\mathbf{x})$, gradient theo các lớp đầu là tích của nhiều ma trận đạo hàm; nếu các thừa số nhỏ, gradient gần như bằng 0 và các lớp đầu không học được. Các giải pháp: hàm kích hoạt ReLU / Leaky ReLU; khởi tạo trọng số phù hợp - Xavier/Glorot $W \sim U\left[-\frac{\sqrt{6}}{\sqrt{n_{in}+n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in}+n_{out}}}\right]$ giữ phương sai tín hiệu ổn định qua các lớp; chuẩn hoá theo lô (batch normalization) và kết nối tắt (residual connection).

**Chuẩn hoá theo lô (batch normalization - BN)** (Ioffe & Szegedy, 2015). Giá trị kích hoạt ở các lớp trung gian có thể biến thiên rất khác nhau giữa các lớp và theo thời gian huấn luyện, gây khó hội tụ. BN chuẩn hoá mỗi đặc trưng theo thống kê của mini-batch $\mathcal{B}$ rồi khôi phục bậc tự do bằng hai tham số học được $\gamma$ (scale) và $\beta$ (shift):

$$\hat{\mu}_\mathcal{B} = \frac{1}{|\mathcal{B}|}\sum_{\mathbf{x}\in\mathcal{B}}\mathbf{x}, \quad
\hat{\sigma}_\mathcal{B}^2 = \frac{1}{|\mathcal{B}|}\sum_{\mathbf{x}\in\mathcal{B}}(\mathbf{x}-\hat{\mu}_\mathcal{B})^2 + \epsilon, \quad
\mathrm{BN}(\mathbf{x}) = \boldsymbol{\gamma}\odot\frac{\mathbf{x}-\hat{\mu}_\mathcal{B}}{\hat{\sigma}_\mathcal{B}} + \boldsymbol{\beta}.$$

Khi suy luận, BN dùng trung bình và phương sai trượt (moving average) tích luỹ trong quá trình huấn luyện. Với lớp tích chập, thống kê được tính theo từng kênh trên toàn bộ batch và mọi vị trí không gian.

**Chống quá khớp (overfitting).**

* *Chính quy hoá L2 (ridge / weight decay)*: thêm $\frac{\lambda}{2}\lVert\mathbf{W}\rVert^2$ vào hàm mất mát để phạt trọng số lớn.
* *Dropout* (Srivastava et al., 2014): ở mỗi bước huấn luyện, ngẫu nhiên đặt về 0 một tỉ lệ $p$ (thường $\le 0{,}5$) nơ-ron của một lớp, buộc mạng không phụ thuộc vào một vài nơ-ron cụ thể; khi suy luận dropout bị tắt.
* *Dừng sớm (early stopping)*: theo dõi hàm mất mát trên tập kiểm định và dừng khi nó không còn giảm, giữ lại trọng số tốt nhất.
* *Giảm tốc độ học khi chững lại (ReduceLROnPlateau)*: nhân tốc độ học với một hệ số nhỏ hơn 1 khi hàm mất mát kiểm định không cải thiện sau vài epoch.
* *Tăng cường dữ liệu* (mục 1.8).

## 1.6. Các kiến trúc CNN tiêu biểu

**LeNet-5** (LeCun et al., 1998) - CNN đầu tiên được ứng dụng thực tế (nhận dạng chữ số viết tay): hai lớp tích chập $5 \times 5$ (6 và 16 kênh, kích hoạt sigmoid), mỗi lớp theo sau bởi gộp trung bình $2 \times 2$ stride 2, rồi ba lớp kết nối đầy đủ 120 - 84 - 10.

**AlexNet** (Krizhevsky et al., 2012) - sâu hơn nhiều (5 lớp tích chập, 2 lớp ẩn kết nối đầy đủ và 1 lớp đầu ra), dùng ReLU thay sigmoid, dropout cho các lớp kết nối đầy đủ, ảnh đầu vào $224 \times 224$; chiến thắng ImageNet 2012 và mở ra kỷ nguyên học sâu cho thị giác máy tính.

**VGG** (Simonyan & Zisserman, 2014) - xây dựng mạng từ các **khối (block)** lặp lại: mỗi khối VGG gồm vài lớp tích chập $3 \times 3$ (padding 1) và một lớp gộp cực đại $2 \times 2$ stride 2 (giảm một nửa độ phân giải). VGG-11 có 5 khối (8 lớp tích chập, số kênh tăng gấp đôi từ 64 đến 512) và 3 lớp kết nối đầy đủ. Ý tưởng "mạng = chồng các khối" được dùng trong hầu hết các kiến trúc sau này.

**NiN - Network in Network** (Lin et al., 2013) - mỗi khối NiN gồm một lớp tích chập theo sau bởi hai lớp tích chập $1 \times 1$ (đóng vai trò lớp kết nối đầy đủ tại từng điểm ảnh); lớp cuối có số kênh bằng số lớp và được gộp trung bình toàn cục thay cho các lớp kết nối đầy đủ - giảm đáng kể số tham số.

**GoogLeNet - mạng đa nhánh** (Szegedy et al., 2015) - khối Inception có bốn nhánh song song: tích chập $1 \times 1$; $1 \times 1$ rồi $3 \times 3$; $1 \times 1$ rồi $5 \times 5$; gộp cực đại $3 \times 3$ rồi $1 \times 1$. Các nhánh trích xuất thông tin ở nhiều kích thước không gian, được đệm để giữ nguyên chiều cao - chiều rộng và nối lại theo chiều kênh. GoogLeNet xếp 9 khối Inception thành ba nhóm, ngăn cách bởi gộp cực đại, và kết thúc bằng GAP.

**ResNet - mạng phần dư** (He et al., 2016) - trong một khối thông thường, các lớp phải học trực tiếp ánh xạ $f(\mathbf{x})$; trong **khối phần dư (residual block)** chúng chỉ cần học phần dư $g(\mathbf{x}) = f(\mathbf{x}) - \mathbf{x}$ và đầu ra là

$$f(\mathbf{x}) = g(\mathbf{x}) + \mathbf{x}.$$

Khi ánh xạ tối ưu gần với ánh xạ đồng nhất, mạng chỉ cần đẩy $g$ về 0 - dễ học hơn nhiều; đường tắt (shortcut) cũng giúp gradient truyền thẳng về các lớp đầu, khắc phục triệt tiêu gradient. Khối phần dư cơ bản gồm Conv $3 \times 3$ - BN - ReLU - Conv $3 \times 3$ - BN, cộng với đường tắt rồi qua ReLU; khi số kênh hoặc kích thước thay đổi, đường tắt dùng tích chập $1 \times 1$ (kèm stride) để biến đổi đầu vào cho khớp. ResNet-18 xếp các khối này thành bốn giai đoạn (stage) 64 - 128 - 256 - 512 kênh và chiến thắng ImageNet 2015.

**DenseNet** (Huang et al., 2017) - trong mỗi khối dày đặc (dense block), mỗi lớp nhận bản đồ đặc trưng của **tất cả** các lớp trước (nối theo chiều kênh) và tạo thêm $k$ kênh mới (growth rate), khuyến khích tái sử dụng đặc trưng; các khối được nối bởi lớp chuyển tiếp (BN, tích chập $1 \times 1$, gộp trung bình $2 \times 2$) để giảm số kênh và kích thước.

**MobileNetV2** (Sandler et al., 2018) - kiến trúc nhẹ cho thiết bị di động, được dùng làm mạng nền trong Mô hình 3. Hai ý tưởng chính:

* *Tích chập tách biệt theo chiều sâu (depthwise separable convolution)*: tách một lớp tích chập $k \times k$ thông thường thành tích chập theo chiều sâu (mỗi kênh một kernel $k \times k$ riêng) và tích chập $1 \times 1$ kết hợp các kênh; chi phí tính toán giảm khoảng $k^2$ lần (gần 9 lần với $k = 3$).
* *Khối phần dư đảo ngược với nút cổ chai tuyến tính (inverted residual, linear bottleneck)*: $1 \times 1$ mở rộng số kênh (thường gấp 6) - depthwise $3 \times 3$ - $1 \times 1$ thu hẹp số kênh **không có hàm kích hoạt**; đường tắt nối các đầu "hẹp" khi stride bằng 1 và số kênh không đổi.

Mạng gồm một lớp tích chập đầu, 17 khối như trên và một lớp $1 \times 1$ cuối 1.280 kênh; chỉ khoảng 3,5 triệu tham số (2,26 triệu khi bỏ phần phân loại) và khoảng 300 triệu phép nhân - cộng cho ảnh $224 \times 224$, được huấn luyện sẵn trên ImageNet (khoảng 1,28 triệu ảnh, 1.000 lớp; Deng et al., 2009).

## 1.7. Học chuyển giao (transfer learning) và tinh chỉnh (fine-tuning)

Học chuyển giao là quá trình áp dụng tri thức của một mô hình đã được huấn luyện trước (pretrained model) cho bài toán hiện tại. Các đặc trưng mà CNN học được trên tập dữ liệu lớn như ImageNet - đặc biệt ở các lớp đầu (cạnh, màu sắc, kết cấu) - mang tính tổng quát và có thể dùng lại cho nhiều bài toán khác (Yosinski et al., 2014). Lợi ích: hội tụ nhanh hơn, độ chính xác cao hơn, chi phí huấn luyện thấp hơn, hiệu quả khi dữ liệu ít.

Quy trình gồm ba pha:

1. **Huấn luyện trước (pretraining)**: mô hình được huấn luyện trên tập lớn (ImageNet) và học được đặc trưng tổng quát.
2. **Chuyển giao (transfer)**: lấy **mạng nền (base network)** - phần trích xuất đặc trưng sau khi bỏ các lớp kết nối đầy đủ ở đỉnh (`include_top=False`) - giữ nguyên trọng số đã huấn luyện, rồi gắn một MLP mới (khởi tạo ngẫu nhiên) để tính xác suất đầu ra. Có thể đóng băng toàn bộ mạng nền (trích xuất đặc trưng - feature extraction) và chỉ huấn luyện MLP.
3. **Tinh chỉnh (fine-tuning)**: mở băng một phần mạng nền (thường là các khối cuối, mang đặc trưng chuyên biệt của bài toán gốc) và huấn luyện tiếp trên dữ liệu mới với tốc độ học nhỏ.

Các lưu ý thực hành: (i) huấn luyện phần đầu ra mới trước rồi mới tinh chỉnh, tránh gradient lớn từ các lớp khởi tạo ngẫu nhiên phá hỏng trọng số đã học; (ii) dùng tốc độ học nhỏ ($10^{-4}$ - $10^{-5}$) khi tinh chỉnh; (iii) giữ các lớp batch normalization ở chế độ suy luận, vì thống kê của các mini-batch nhỏ trên tập dữ liệu mới kém ổn định; (iv) tiền xử lý ảnh **giống hệt** lúc huấn luyện trước - MobileNetV2 yêu cầu điểm ảnh trong khoảng $[-1, 1]$ ($x/127{,}5 - 1$).

## 1.8. Tăng cường dữ liệu (data augmentation)

Hiệu năng kém của mô hình thường đến từ dữ liệu không đủ hoặc không đại diện, mất cân bằng lớp, kiến trúc quá phức tạp so với dữ liệu, hoặc khó khăn trong tối ưu. Tăng cường dữ liệu tạo thêm dữ liệu huấn luyện một cách nhân tạo bằng các phép biến đổi ngẫu nhiên giữ nguyên nhãn: xoay, lật, cắt; phóng to - thu nhỏ, dịch chuyển; điều chỉnh độ sáng, độ tương phản, màu sắc; thêm nhiễu. Mỗi epoch, mô hình nhìn thấy một phiên bản biến đổi khác của từng ảnh ("không bao giờ thấy cùng một ảnh hai lần"), nhờ đó giảm quá khớp, tăng khả năng tổng quát hoá và độ bền vững. Tăng cường dữ liệu chỉ áp dụng cho **tập huấn luyện**; tập kiểm định và tập kiểm tra giữ nguyên để đánh giá trung thực. Phép biến đổi phải giữ nguyên nhãn - ví dụ với bài toán độ tươi, không nên thay đổi sắc độ màu vì màu nâu, vết thâm là dấu hiệu của quả hỏng.

## 1.9. Học đa nhiệm (multi-task learning)

Học đa nhiệm (Caruana, 1997) huấn luyện một mô hình giải đồng thời nhiều bài toán liên quan: một **thân chung (shared trunk)** trích xuất đặc trưng, rẽ thành nhiều **đầu ra (head)**, mỗi đầu cho một nhiệm vụ. Hàm mất mát tổng là tổng có trọng số các hàm mất mát thành phần

$$J = \sum_{t} \lambda_t J_t.$$

Với bài toán nhận diện loại quả và độ tươi: $J = \lambda_1 \cdot \mathrm{CE}(\text{loại quả}) + \lambda_2 \cdot \mathrm{BCE}(\text{độ tươi})$. Lợi ích: hai nhiệm vụ dùng chung đặc trưng cấp thấp (hình dạng, màu, kết cấu) nên cần ít tham số hơn hai mô hình riêng; nhiệm vụ này đóng vai trò chính quy hoá cho nhiệm vụ kia; và đầu ra có cấu trúc rõ ràng hơn so với gộp thành một bài toán 16 lớp. Trong Keras, mô hình nhiều đầu ra được xây dựng bằng **Functional API**, cho phép đồ thị tính toán rẽ nhánh mà API Sequential không biểu diễn được.

## 1.10. Đánh giá mô hình phân loại

Với ma trận nhầm lẫn (confusion matrix) của bài toán nhị phân gồm TP, FP, TN, FN:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}, \quad \text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}.$$

Precision cho biết trong các mẫu được dự đoán là dương tính có bao nhiêu phần đúng; recall cho biết mô hình tìm được bao nhiêu phần trong số các mẫu dương tính thực sự; $F_1$ là trung bình điều hoà của hai đại lượng. Với bài toán nhiều lớp, các chỉ số được tính cho từng lớp rồi lấy trung bình không trọng số (**macro average**), để các lớp ít mẫu được coi trọng như lớp nhiều mẫu. Diện tích dưới đường cong ROC (ROC-AUC) đo khả năng xếp hạng: xác suất một mẫu dương tính ngẫu nhiên được cho điểm cao hơn một mẫu âm tính ngẫu nhiên. Ma trận nhầm lẫn cho thấy cụ thể lớp nào bị nhầm sang lớp nào.

## 1.11. Giải thích mô hình bằng Grad-CAM

CNN thường bị xem là "hộp đen". Grad-CAM (Selvaraju et al., 2017) chỉ ra vùng ảnh nào ảnh hưởng mạnh nhất đến một quyết định. Gọi $A^k$ là bản đồ đặc trưng thứ $k$ của lớp tích chập cuối và $y^c$ là điểm số của quyết định cần giải thích (ví dụ log-odds của xác suất "hỏng"). Trọng số của từng kênh là gradient trung bình theo không gian

$$\alpha_k^c = \frac{1}{Z}\sum_i\sum_j \frac{\partial y^c}{\partial A^k_{ij}},$$

và bản đồ nhiệt (heatmap) là tổ hợp tuyến tính các bản đồ đặc trưng, giữ lại phần dương:

$$L^c_{\text{Grad-CAM}} = \mathrm{ReLU}\left(\sum_k \alpha_k^c A^k\right).$$

Bản đồ nhiệt được phóng to về kích thước ảnh và phủ lên ảnh gốc: vùng sáng là vùng ủng hộ mạnh cho quyết định. Grad-CAM giúp kiểm tra mô hình có nhìn vào đúng vùng quả (vết thâm, nấm mốc) hay đang dựa vào các yếu tố không liên quan như nền ảnh.
