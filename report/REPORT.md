# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:9/11/2026**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`)
class_id:468, class_name: "cab", rank: 1, score: 0.510915, taxonomy_name: "ImageNet-1K". 
- Record này mô tả toàn ảnh như thế nào?
class_id cho ta biết mã số duy nhất của lớp là 468, class_name cho ta biết ảnh được phân loại chủ yếu là "cab", cho thấy taxi/cab là đối tượng nổi bật nhất trong khung hình theo nhận định của mô hình, rank cho ta biết đây là dự đoán đứng đầu trong tất cả 1000 lớp của ImageNet-1K, score cho ta biết mô hình tin khoảng 51.09% rằng ảnh thuộc lớp "cab", taxonomy_name cho ta biết rằng mình đang sử dung mô hình phân loại ảnh dựa trên 1000 lớp của ImageNet.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
Class list thường được định nghĩa bởi dữ liệu huấn luyện và người huấn luyện mô hình.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
Tên lớp hoặc tên taxonomy có thể thay đổi, nhưng ID thì cố định. Tuy nhiên, ID quá khó nhớ với con người và không tốt cho SEO. Tên lớp giúp lập trình viên dễ gọi cấu trúc trong mã nguồn. Một tên lớp có thể xuất hiện ở nhiều nhóm khác nhau, tên taxonomy giúp hệ thống biết chính xác nhóm đó thuộc ngữ cảnh nào. Việc sử dụng cả 3 giúp giảm xung đột dữ liệu, tăng hiểu quả làm việc và tìm kiếm.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
Khi ảnh có nhiều chủ thể, guideline cần quy định bài toán và single label hay multi label, cách chọn chủ thể để có thể label ảnh, chọn chủ thể chiếm nhiều diện tích nhất, ở trung tâm hay theo tiêu chí dự án. ở đây mỗi ảnh được biểu diễn bởi Top-5 lớp có xác suất cao nhất trong taxonomy ImageNet-1K; lớp có rank=1 là mô tả phân loại chính của ảnh.
- Vì sao model score không phải ground truth?
Vì model score chỉ là mức độ tự tin của mô hình đối với dự đoán, còn ground truth là nhãn chuẩn được gán bởi dữ liệu hoặc con người, nên mô hình có thể dự đoán sai dù score rất cao hoặc dự đoán đúng dù score thấp.
## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
class_name: "bus", score: 0.912557, bbox_xyxy": [93.17, 187.95, 223.01, 320.91], bbox_width: 129.84, bbox_height": 132.96 
- Diễn giải vị trí box bằng lời:
Bounding box này bao quanh vật thể mà model định dạng là một xe buýt (bus) với độ tin cậy 91.26%, nằm ở nửa bên trái ảnh, bắt đầu khoảng 93 px từ mép trái và 188 px từ mép trên, kéo dài đến 223 px từ mép trái và 321 px từ mép trên, tạo thành một vùng có kích thước khoảng 130 × 133 px.
- So sánh số prediction ở hai threshold:
Khi tăng ngưỡng confidence (threshold), số lượng prediction giảm vì các dự đoán có độ tin cậy thấp bị loại bỏ:
0.20 → 17 vật thể: giữ lại nhiều đối tượng nhất, bao gồm cả các dự đoán có độ tin cậy thấp như spoon, bottle, potted plant, dining table.
0.35 → 11 vật thể: giảm 6 prediction (35.3%), chủ yếu loại bỏ các đối tượng ít chắc chắn; vẫn giữ lại person, bowl, oven, cup.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
Khi tăng threshold từ 0.20 → 0.60, độ bao phủ giảm vì nhiều đối tượng có confidence thấp bị loại bỏ, trong khi khối lượng reviewer cần xem giảm đáng kể (từ 17 xuống 6 prediction, giảm khoảng 65%).
Trong ví dụ, khi đặt ngưỡng là 0.6, hai cốc (0.45) và 1 bát (0.5) bị bỏ sót khiến cho reviewer phải xem lại, hạ ngưỡng thêm các vật thể có score thấp, tăng độ bao phủ nhưng có thể thêm các bounding box sai hoặc trùng lặp tăng khối lượng công việc cho reviewer.
- Đề xuất một quy tắc box chặt:
Box phải ôm sát toàn bộ phần nhìn thấy của object (sai số ≤ 2 px), không chứa nền thừa, không cắt vật thể; nếu bị che khuất thì chỉ box phần nhìn thấy.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
Ảnh kitchen cho thấy nhiều trường hợp biên cần guideline làm rõ: người bên trái được xác định chỉ còn phần cánh tay/bàn tay ở mép ảnh; một số bowl bị cắt khung hoặc chồng lấn nhau; cup bị che khuất một phần. Cần quy định: (1) mức độ hiển thị tối thiểu để vẫn gán nhãn object, (2) box chỉ bao phần nhìn thấy hay bao cả phần khuất/ngoài khung, (3) có đánh dấu trạng thái truncated và occluded hay không, và (4) cách xử lý vật thể không có lớp tương ứng trong COCO-80, chẳng hạn gán lớp gần nhất, dùng lớp other, hay chuyển sang escalation để review thủ công.
## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
instance_id: "traffic-001", class_name: "bus", score: 0.925745,  polygon_point_count: 120, "polygon_xy": [148.0, 189.0].
- Polygon bổ sung chi tiết gì so với box?
Polygon bổ sung hình dạng thực của object mà box không thể hiện được. Box chỉ xác định vùng chữ nhật chứa object nên thường bao gồm cả nền, trong khi polygon bám theo biên object và cho biết chính xác pixel nào thuộc object. Trong ảnh kitchen, polygon của person chỉ chiếm khoảng 57% diện tích box, phần còn lại là nền; với các object mảnh như spoon, tỷ lệ này còn thấp hơn (~24%). Polygon cũng giúp tách các object sát nhau hoặc chồng lấn mà box khó phân biệt. Đổi lại, polygon phức tạp hơn đáng kể, cần nhiều điểm để vẽ và tốn công gán nhãn cũng như kiểm định chất lượng hơn.
- `instance_id` dùng để làm gì và không phải loại ID nào?
instance_id là mã định danh duy nhất cho từng object instance trong kết quả của một ảnh, có dạng <sample_id>-NNN. Nó dùng để phân biệt nhiều object cùng lớp, ví dụ nhiều bowl hoặc nhiều person trong cùng ảnh. instance_id không phải class_id (nhãn lớp), không phải tracking ID giữa các frame hay các lần chạy model, và cũng không phải annotation ID gốc của COCO; nếu đổi model hoặc chạy lại, các mã này có thể thay đổi.
- Đề xuất một quy tắc biên mask:
Biên mask phải đi theo ranh giới nhìn thấy được của object ở độ chính xác pixel-level; mọi pixel thuộc object được đưa vào mask, mọi pixel nền bị loại bỏ. Không suy diễn phần bị che khuất hoặc nằm ngoài khung hình. Với biên mờ hoặc khó phân biệt, ưu tiên quy tắc nhất quán hơn là cố gắng đoán hình dạng thực.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
Với biên mờ, có ngưỡng nhất quán cho motion blur, tóc, lông, khói, phản chiếu không? Với 2 vật thể tiếp xúc nhau, nếu thấy ranh giới thì tách mask, nếu không thì thì gộp, bỏ qua hay escalation? Với vật bị che khuất thì chỉ lấy phần nhìn thấy.
## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Class (class_id, class_name, taxonomy_name) | Nhiều chủ thể, lớp gần giống nhau, taxonomy thiếu lớp | xác định chủ thể chính, gán nhãn và guideline| xác minh nhãn và ID đúng taxonomy, ưu tiên review các ảnh có score thấp hoặc chênh lệch top-1/top-2 nhỏ, kiểm tra việc áp dụng quy tắc trên ảnh nhiều chủ thể, đồng thời theo dõi mức độ đồng thuận giữa các annotator. |
| Phát hiện vật thể | Class (class_id, class_name, taxonomy_name), Bounding box (xyxy) + class | Box quá rộng/hẹp, object bị che hoặc cắt khung, lớp gần giống | Vẽ box chặt, gán đúng lớp, đánh dấu trường hợp mơ hồ  | Box chặt, đúng lớp, xử lý occlusion/truncation nhất quán |
| Instance segmentation | Polygon/mask + class | Biên mờ, vật chạm nhau, vật bị che khuất | Vẽ mask theo phần nhìn thấy, tách instance riêng | Biên mask chính xác, không thừa/thiếu, tách instance đúng |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
Chỉ sử dụng 3 ảnh COCO công khai (CC BY 2.0) đã được notebook tải xuống và xác minh checksum, đồng thời luôn giữ kèm file IMAGE_ATTRIBUTION.md. Không đưa ảnh cá nhân, dữ liệu khách hàng/VinFast hoặc tài liệu nội bộ lên Colab hay repository công khai, và không ghi thông tin định danh cá nhân (họ tên, MSSV, email) trong báo cáo hoặc output. Mặc dù ảnh là dữ liệu công khai, không thực hiện phóng to, cắt ghép hoặc phân tích nhằm nhận dạng người trong ảnh (kitchen, traffic) hay đọc biển số xe; trong môi trường sản xuất, các thông tin nhận dạng như khuôn mặt và biển số cần được làm mờ theo quy định bảo vệ dữ liệu.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
mentor/giảng viên phụ trách lớp hoặc người có trách nhiệm.
## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
