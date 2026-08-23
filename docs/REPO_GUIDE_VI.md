# T2ICount hiện tại: bản đồ kiến trúc, paper lineage và thay đổi so với upstream

> Ảnh chụp trạng thái ngày **2026-08-22**. Tài liệu này mô tả checkout hiện tại tại nhánh `DUMLO`, gồm cả code đã commit, thay đổi chưa commit và các artifact bị Git ignore đang có trong workspace.

## 1. Kết luận nhanh

Repo này hiện có ba lớp cần phân biệt:

1. **T2ICount gốc** từ `cha15yq/T2ICount`, mốc `upstream/main = 289d3fb`.
2. **Phần mở rộng đã commit** của repo này, từ `2addc67` đến `8706cf7` trên nhánh `DUMLO`.
3. **Trạng thái làm việc chưa commit/không track**: notebook Colab chứa một run DUMLO đã chạy, công cụ chẩn đoán gradient DUMLO, test của công cụ đó, và hai CSV IDCIA trong `results/`.

Về nghiên cứu, kiến trúc T2ICount gốc vẫn là trung tâm:

```text
ảnh + prompt
  -> VAE encoder + CLIP text encoder của Stable Diffusion v1.5
  -> U-Net tại timestep t = 0
  -> feature và cross-attention nhiều mức
  -> HSCM (SEM + SCM theo tầng)
  -> counter
  -> density map
```

Thay đổi nghiên cứu lớn nhất là **DUMLO loss tùy chọn**. Nó chỉ thay `L_reg` khi chạy với `--loss-mode dumlo`; mặc định vẫn là `baseline`. DUMLO không thay model, optimizer, RRC, HSCM, inference hay cách scale density map.

## 2. Mốc Git và cách đọc nhãn trong tài liệu

- Remote gốc: `upstream = https://github.com/cha15yq/T2ICount.git`.
- Remote triển khai này: `origin = https://github.com/bqa100507-spec/T2ICount-Implementation.git`.
- Nhánh hiện tại: `DUMLO`, tracking `origin/DUMLO`.
- `HEAD`: `8706cf7 Make count loss weight be able to change`.
- Merge-base/tip upstream dùng để so sánh: `289d3fb`.
- Diff đã commit so với upstream: **47 file, +5549/-549 dòng** (notebook JSON chiếm phần đáng kể).

Các nhãn dùng dưới đây:

- **[GỐC]**: kế thừa nguyên trạng hoặc gần như nguyên trạng từ upstream T2ICount.
- **[SỬA]**: file upstream đã được sửa trong repo này.
- **[MỚI]**: file mới được thêm sau upstream.
- **[WORKTREE]**: thay đổi/file hiện có nhưng chưa commit.
- **[IGNORED]**: artifact cục bộ bị `.gitignore` loại khỏi Git.

## 3. Các paper liên quan và phần code tương ứng

| Paper/nguồn | Ý tưởng được dùng | Vị trí trong repo |
| --- | --- | --- |
| [T2ICount, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Qian_T2ICount_Enhancing_Cross-modal_Understanding_for_Zero-Shot_Counting_CVPR_2025_paper.html) | Single-step diffusion feature, HSCM, SEM/SCM, RRC, FSC-147-S | `models/`, `utils/regression_trainer.py`, `datasets/dataset.py`, `FSC-147-S.json` |
| [Latent Diffusion Models](https://arxiv.org/abs/2112.10752) | VAE latent space, conditional U-Net, cross-attention | `ldm/`, `configs/v1-inference.yaml` |
| [CLIP](https://arxiv.org/abs/2103.00020) | Text representation cho prompt mở | `FrozenCLIPEmbedder`, prompt token mask, text-image similarity |
| [Learning To Count Everything / FSC-147](https://openaccess.thecvf.com/content/CVPR2021/html/Ranjan_Learning_To_Count_Everything_CVPR_2021_paper.html) | Dataset 147 lớp, point annotation và density map | `ObjectCount`, asset FSC-147 bên ngoài repo |
| [CUT, BMVC 2022](https://research-portal.st-andrews.ac.uk/en/publications/segmentation-assisted-u-shaped-multi-scale-transformer-for-crowd-/) | Regression loss mà T2ICount nói là kế thừa | `get_reg_loss`, `utils/ssim_loss.py` |
| [DUMLO](https://www.nature.com/articles/s41598-025-14056-2) | Annotation uncertainty, ba phân phối, Trihorn, OT/count/TV loss | `losses/dumlo.py`, point propagation trong dataset |
| [DM-Count](https://arxiv.org/abs/2009.13077) | OT giữa point distribution và predicted density; count + TV stabilization | Tiền thân trực tiếp về loss của DUMLO |
| [Zero-Shot Object Counting, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_Zero-Shot_Object_Counting_CVPR_2023_paper.html) | Định nghĩa bài toán ZSOC chỉ dùng tên lớp, không dùng exemplar | Bối cảnh bài toán; repo **không** implement patch-selection/ranking của paper này |

Điểm quan trọng: repo này là implementation của **T2ICount**, không phải một bộ tổng hợp các phương pháp ZSOC. Những paper như CLIP-Count, CounTX, VLCounter, DAVE hay ZSC patch-selection xuất hiện ở phần related work/baseline của paper, nhưng không có module tương ứng trong code này.

## 4. Đường chạy model thực tế

### 4.1 Xây model

`models.build.build_t2icount` là entry point thống nhất cho train, test, notebook và visualization:

1. Kiểm tra config, SD checkpoint và thư mục CLIP đều là đường dẫn local hợp lệ.
2. Ghi `clip_path` và device vào `cond_stage_config` trong OmegaConf.
3. Khởi tạo `models.reg_model.Count`.
4. Nếu có checkpoint T2ICount, nạp nó với `strict=True` sau khi xử lý đúng một buffer tương thích `position_ids`.
5. Chuyển model sang device và đặt train/eval mode.

Trong `Count.__init__`:

- Toàn bộ `LatentDiffusion` được dựng từ YAML và SD v1.5 checkpoint.
- Chỉ giữ VAE, U-Net và CLIP text encoder.
- Xóa `vae.decoder` vì đếm chỉ cần encoder.
- Freeze VAE và CLIP.
- Gắn decoder/counter của T2ICount.

### 4.2 Forward cho crop 384×384

Các shape chính với cấu hình mặc định:

| Tensor | Shape điển hình | Ý nghĩa |
| --- | --- | --- |
| Input | `B×3×384×384` | RGB đã normalize về khoảng gần `[-1, 1]` |
| VAE latent | `B×4×48×48` | posterior mode từ VAE encoder |
| CLIP tokens | `B×77×768` | hidden states của CLIP ViT-L/14 text encoder |
| U-Net features | xấp xỉ `320×48²`, `640×24²`, `1280×12²`, `1280×6²` | bốn mức decoder feature |
| Cross-attention | `77×12²`, `77×24²`, `77×48²` | attention trung bình theo head, sau đó chọn token prompt |
| Density output | `B×1×48×48` | density ở độ phân giải 1/8, scale huấn luyện ×60 |

`Count.extract_feat` thực hiện:

```python
posterior = vae.encode(img)
latent = posterior.mode()
text = clip.encode(prompt)
t = zeros(batch)
features = unet(latent, t, c_crossattn=[text])
```

Hai chi tiết dễ bị hiểu sai:

- Code không chạy chuỗi denoising nhiều bước; nó gọi U-Net đúng một lần tại `t=0`.
- Code cũng không tự thêm noise vào latent trước lần gọi này. Vì vậy “single denoising step” trong paper nên được hiểu theo implementation chính thức là **một lượt U-Net có conditioning tại timestep 0 để trích feature**, không phải một vòng sampling Stable Diffusion hoàn chỉnh.

Ngoài ra, active path dùng `posterior.mode()` trực tiếp, không gọi `LatentDiffusion.get_first_stage_encoding`; do đó `scale_factor: 0.18215` trong YAML không được áp dụng ở `Count.extract_feat`. Đây là hành vi upstream, không phải thay đổi của nhánh DUMLO.

### 4.3 Cách lấy feature và attention từ U-Net

`models/diff_unet.py` monkey-patch U-Net của Stable Diffusion theo hai hướng:

- `register_hier_output` thay `UNetModel.forward` để trả feature tại các output block `1, 4, 7` và feature cuối, thay vì trả noise prediction qua head `self.out`.
- `register_attention_control` thay `CrossAttention.forward` để lưu attention map trung bình theo head trong các nhánh `down`, `mid`, `up`.

`UNetWrapper.process_attn` chỉ dùng selector mặc định `down_cross+up_cross`, gom attention theo spatial size 12, 24, 48 và trung bình các layer cùng size. Đây là nguồn của ba bản đồ `attn12`, `attn24`, `attn48`.

### 4.4 Prompt mask và fused cross-attention

Prompt mask dài 77 đánh dấu các token nội dung tại vị trí `1..len(prompt_tokens)`, bỏ token đầu và padding/EOS. Mask này được dùng hai lần:

- Trung bình CLIP hidden states chỉ trên token của cụm prompt để có vector text `B×768`.
- Trung bình cross-attention chỉ trên cùng các token để có attention map theo prompt.

Ba attention map được resize về 48×48, min-max normalize riêng rồi trộn:

```text
A_cross = 0.6 * norm(A_12)
        + 0.3 * norm(A_24)
        + 0.1 * norm(A_48)
```

Code viết theo thứ tự `0.1*A48 + 0.3*A24 + 0.6*A12`, đúng bộ trọng số paper nêu cho `[12, 24, 48]`.

## 5. HSCM, SEM, SCM và Counter trong code

### 5.1 `models/decoder.py::ImgTxtFusion` — SEM

Đây là phần gần nhất với **Semantic Enhancement Module** của T2ICount:

1. Tạo image positional feature bằng depthwise convolution.
2. Text-to-image attention: text query đọc toàn bộ image tokens.
3. Residual + LayerNorm + feed-forward.
4. Image-to-text attention: image queries đọc text token đã cập nhật.
5. Trả lại cả text feature và image feature đã hiệu chỉnh.

Mỗi trong hai stage semantic của decoder gọi `ImgTxtFusion` hai lần.

### 5.2 `models/decoder.py::Upsample` — fusion và SCM

`Upsample.forward(low, high, text_feat, simi_map)`:

1. Upsample feature sâu hơn lên 2×.
2. Convolution và concatenate với skip feature cùng độ phân giải.
3. Nếu có similarity map từ stage trước, cộng `simi_map * low` vào fused feature. Đây là **Semantic Correction Module** trong code.
4. Nếu `sem=True`, chiếu text 768 chiều về số channel hiện tại và chạy hai SEM.

`models.reg_model.Decoder` xâu ba stage:

- Stage 1: 6→12, có SEM; tạo `sim_x2` ở 48×48 để tính RRC và resize nó về 24×24 cho stage sau.
- Stage 2: 12→24, có SCM + SEM; tạo `sim_x1` ở 48×48.
- Stage 3: 24→48, chỉ SCM, không có cross-modal SEM.
- Cuối cùng đưa feature 256 channel vào `Regressor`.

### 5.3 `models/decoder.py::Regressor` — Counter

Counter chạy hai block self-attention trên 48×48 image tokens. Mỗi block dùng depthwise convolution làm positional feature, multi-head attention, residual, LayerNorm. Head cuối gồm ba convolution và ReLU cuối để density không âm.

Đây là phần “Counter” trong Figure 2 của paper T2ICount; nó không phải DUMLO.

## 6. RRC và baseline regression loss

### 6.1 Tạo positive/negative/ambiguous supervision

Trong `Reg_Trainer.train_epoch`:

```python
fused_cross_attn_ = fused_cross_attn * gt_img_attn_mask
AN = fused_cross_attn_ >= 0.3
P = gt_den_maps >= 0.06  # 1e-3 * 60
```

- `P=True`: positive — vùng density GT vượt ngưỡng.
- `AN=False` và `P=False`: negative đáng tin.
- Phần còn lại: ambiguous, không bị ép làm background.

`RRC_loss` dùng:

```text
L_pos = mean_positive(1 - similarity)
L_neg = mean_negative(max(0, similarity))
L_RRC = 2 * L_pos + L_neg
```

Code áp loss ở `sim_x2` và `sim_x1`, rồi cộng:

```text
L_total = L_reg + 0.01 * L_RRC_stage1 + 0.01 * L_RRC_stage2
```

Hệ số `2` tương ứng λ của paper; `0.01` tương ứng γ. Dùng hai stage là cách code chính thức triển khai supervision nhiều tầng.

### 6.2 Một khác biệt paper-code đáng chú ý

Equation PNA trong bản CVPR mô tả pseudo-background theo attention thấp tương đối với trung bình fused attention. Upstream code lại dùng ngưỡng cố định `0.3` sau khi nhân `img_attn_mask`. Đây là khác biệt đã có trong repo gốc; nhánh hiện tại không tạo ra nó.

`img_attn_mask` còn mang thông tin augmentation: tile đúng lớp có 1, tile sai lớp có 0. Vì vậy RRC trong code vừa dùng cross-attention, vừa biết vùng nào của mosaic được xây từ lớp mục tiêu.

### 6.3 `get_reg_loss` — baseline `L_reg`

Baseline không phải MSE pixel thông thường:

1. Dùng mask `gt > 0.06` để tính `1 - SSIM` qua 3 mức pooling, chỉ tập trung vùng foreground.
2. Chuẩn hóa pred và GT để mỗi map có tổng mass gần 1.
3. Tính L1 giữa hai phân phối chuẩn hóa.
4. Trả `L_ssim + 0.1 * L_distribution`.

Hàm được T2ICount mô tả là kế thừa regression loss của CUT. Biến trong code gọi thành phần thứ hai là `tv_loss`, nhưng về mặt toán học nó là L1 giữa hai density distribution đã chuẩn hóa, không phải spatial total variation cổ điển.

## 7. Dataset và augmentation

### 7.1 `datasets/dataset.py::ObjectCount`

Đọc ba thành phần FSC-147:

- split từ `Train_Test_Val_FSC_147.json`;
- class name từ `ImageClasses_FSC147.txt`;
- point annotation từ `annotation_FSC147_384.json`;
- density target đã dựng sẵn từ `gt_density_map_adaptive_384_VarV2/*.npy`.

Với val/test, mỗi sample trả ảnh, `len(points)`, prompt, prompt-token mask và tên ảnh.

Với train, xác suất gần đúng của các nhánh augmentation là:

| Nhánh | Xác suất | Hành vi |
| --- | ---: | --- |
| Ảnh/prompt gốc | 50% | density và toàn vùng attention là positive candidate |
| Prompt ngẫu nhiên | 25% | nếu khác lớp: density=0, point rỗng, attention mask=0; nếu cùng lớp: giữ target |
| Mosaic 2×2 | 25% | crop ảnh hiện tại + 3 ảnh ngẫu nhiên, shuffle tile; chỉ tile cùng lớp prompt giữ density/point |

Sau đó `train_transform_density` có thể:

- resize ngẫu nhiên với factor trong `[1, 2)` và chia density cho `factor²` để bảo toàn mass;
- crop về 384×384;
- sum-pool density xuống 48×48;
- resize nearest-neighbor image-attention mask xuống 48×48;
- horizontal flip ngẫu nhiên.

### 7.2 Point propagation được thêm cho DUMLO

Các helper mới giữ point annotation đồng bộ với chính augmentation cũ:

- `as_point_tensor`: chuẩn hóa `[N,2]` theo thứ tự `[x,y]`.
- `crop_points`: lọc point nằm trong crop và chuyển sang tọa độ local.
- `resize_points`: scale theo kích thước integer thực sau resize.
- `horizontal_flip_points`: đổi `x -> width - 1 - x`.
- `assemble_2x2_points`: cộng offset theo vị trí tile sau shuffle.

`return_points=False` là mặc định nên đường baseline vẫn trả đúng 5 field như upstream. Chỉ train+DUMLO bật `return_points=True` và trả field thứ sáu là point tensor có độ dài thay đổi.

### 7.3 `datasets/carpk.py::CARPK`

- Chọn ảnh thuộc `ImageSets/test.txt`.
- Đếm số dòng annotation làm GT count.
- Prompt cố định là `cars`.
- Thay đổi local chỉ là tái sử dụng tokenizer của model hoặc load tokenizer từ đường dẫn local; dataset/inference semantics không đổi.

### 7.4 `datasets/dataset.py::IDCIA` — phần mở rộng ngoài paper T2ICount

IDCIA chỉ hỗ trợ official test split:

- tìm ảnh/CSV annotation không phân biệt hoa thường;
- xác định staining từ tên thư mục;
- đếm các hàng có tọa độ X,Y hữu hạn;
- đọc TIFF/grayscale, tùy chọn `raw` hoặc `autocontrast`, rồi convert RGB;
- trả ảnh, GT count, tên ảnh và staining.

Đây là adapter đánh giá cross-domain của repo hiện tại, không phải module được paper T2ICount công bố.

## 8. DUMLO implementation

### 8.1 Mục đích

DUMLO thay target Gaussian cố định bằng một bài toán OT ba phân phối:

```text
point GT gốc Z
  -> point uncertainty đã augment Z_tilde
  -> predicted density Z_hat
```

Trihorn mở rộng ý tưởng Sinkhorn để buộc hai transport liên tiếp cùng chia sẻ intermediate distribution.

### 8.2 Các hàm quan trọng trong `losses/dumlo.py`

#### `generate_discrete_map`

Chiếu point full-resolution xuống grid prediction. Dùng `scatter_add_`, nên nhiều point rơi cùng cell vẫn giữ đúng tổng mass. Map này phục vụ TV term, không thay density-map `.npy` của baseline.

#### `prediction_grid_coordinates`

Tạo tọa độ tâm cell prediction trong hệ tọa độ input 384×384. Nhờ đó cost OT giữa point và pixel có cùng đơn vị.

#### `derive_sampling_seed`

Trộn `base_seed`, epoch, step và sample index thành seed riêng. Không đọc hay thay global RNG.

#### `adaptive_augment_points`

- Với nhiều point: bán kính mỗi point = `radius_factor × nearest-neighbor distance`.
- Lấy mẫu đều theo diện tích bên trong disk bằng `r = sqrt(U) × radius`.
- Clamp point về biên ảnh.
- Với đúng một point: dùng `min(H,W)/4` làm surrogate nearest-neighbor distance.

Fallback một-point là quyết định triển khai riêng vì paper không định nghĩa rõ bandwidth cho trường hợp này.

#### `trihorn`

- Tạo kernel `K = exp(-C/epsilon)` cho original→augmented và `K_hat` cho augmented→prediction grid.
- Chạy Gauss-Seidel scaling `u, v, u_hat, v_hat` trong `num_iters` vòng.
- Toàn bộ scaling chạy dưới `torch.no_grad()`.
- Không materialize transport plan; cost diagnostic được tính theo chunk để giảm peak memory.
- Trả `beta_hat`, intermediate mass `z_tilde`, scale vectors và transport cost.

#### `analytical_ot_loss`

Trihorn bị detach, sau đó dùng closed-form gradient theo `beta_hat`. Hàm dựng surrogate có giá trị forward bằng transport cost nhưng backward đúng gradient giải tích mong muốn.

#### `DUMLOLoss.forward`

Với từng sample:

```text
pred_mass  = pred_den / 60
L_count    = |sum(pred_mass) - N_points|
L_TV       = 0.5 * L1(point_probability, predicted_probability)
L_OT       = analytical Trihorn OT surrogate

L_DUMLO = lambda_count * L_count
        + lambda_ot    * L_OT
        + lambda_tv    * N_points * L_TV
```

Batch loss là trung bình theo sample. Diagnostics giữ raw `count_loss`, `ot_loss`, `tv_loss`, mean count và signed count error.

Với negative prompt có 0 point, code chỉ dùng absolute count loss; OT và TV bằng 0. Đây là adaptation cần thiết để DUMLO tương thích negative-prompt augmentation của T2ICount.

### 8.3 Điểm diễn giải riêng so với paper DUMLO

Paper in công thức count/TV có chỗ dùng ký hiệu intermediate distribution, trong khi phần đạo hàm và diễn giải theo DM-Count lại phụ thuộc predicted distribution. Repo này chủ động dùng `pred_mass` cho count/TV vì:

- dùng intermediate mass-conserving `z_tilde` sẽ làm count loss triệt tiêu;
- gradient được paper nêu cũng hướng về predicted density.

Do đó implementation này là một **diễn giải có lý do**, không nên mô tả là bản chép literal mọi ký hiệu của paper.

Các default local (`augmentation_points=10`, `radius_factor=0.5`, `epsilon=10`, `iters=100`) cũng không tự động bảo đảm tái lập đúng mọi bảng của paper DUMLO; paper khảo sát nhiều cấu hình sampling/iteration.

### 8.4 Cách bật và phạm vi ảnh hưởng

```powershell
python train.py ... --loss-mode dumlo `
  --dumlo-lambda-count 1.0 `
  --dumlo-lambda-ot 0.1 `
  --dumlo-lambda-tv 0.01
```

- `--loss-mode baseline` vẫn là mặc định.
- `--dumlo-lambda-count` mới được thêm ở commit `8706cf7`; trước đó count weight ngầm là 1.
- HSCM/RRC vẫn chạy và vẫn cộng vào total training loss.
- Inference không biết model đã được train bằng baseline hay DUMLO; nó chỉ load checkpoint và chạy cùng graph.

## 9. Training loop, optimizer và checkpoint

### 9.1 `utils/regression_trainer.py::Reg_Trainer.setup`

- Yêu cầu đúng một GPU.
- Dựng model trước, rồi truyền tokenizer model vào datasets.
- Chỉ train split dùng batch size CLI; val/test luôn batch 1.
- Optimizer là AdamW với hai group:
  - U-Net: `lr × 0.1`, `weight_decay × 0.1`;
  - HSCM/counter decoder: `lr`, `weight_decay`.
- VAE và CLIP frozen.
- Không có LR scheduler hoặc AMP scaler.

### 9.2 Hai loại subset không được trộn

- `--smoke-train-samples`: lấy **N sample đầu**, chỉ dành cho kiểm tra hạ tầng.
- `--train-samples`: chọn N index ngẫu nhiên nhưng deterministic bằng private Torch generator và `--train-subset-seed`.
- Hai option mutually exclusive.
- Val/test object không bị wrap hay giảm size.

### 9.3 Epoch và metric

- Train log hiển thị regression/RRC/MAE; DUMLO thêm count/OT/TV/pred_count/signed_error.
- Validation bắt đầu từ `start_val`, chạy mỗi `val_epoch`.
- “Best” được chọn theo `MAE + RMSE`, không chỉ MAE.
- Các biến/tin log tên `mse` thực tế chứa **RMSE** vì code đã lấy căn bậc hai.

### 9.4 Checkpoint

- Mỗi 5 epoch lưu `*_ckpt.tar` full-state: model, optimizer, completed/next epoch, best metrics, Python/NumPy/Torch/CUDA RNG.
- Save dùng file `.tmp` rồi `os.replace` để atomic trong cùng thư mục.
- `.pth` là weights-only cho inference, không được dùng `--resume`.
- Legacy `.tar` thiếu metrics/RNG vẫn load được với default phù hợp.
- Checkpoint được load lên CPU trước rồi optimizer/model chuyển theo runtime, tránh phụ thuộc device serialization.

### 9.5 Hyperparameter paper, upstream CLI và run local không đồng nhất

Paper T2ICount ghi cấu hình chính là 400 epoch, batch 16, base learning rate `5e-5`, weight decay `1e-4`, λ của RRC bằng 2 và γ bằng `0.01`. Trong khi đó:

- CLI upstream/current mặc định là 300 epoch, batch 4, `lr=5e-5`, `weight_decay=5e-4`;
- lệnh minh họa trong README upstream/current lại truyền batch 16 và `weight_decay=5e-5`;
- notebook hiện tại chạy các pilot batch 1, subset 500/1000 và 10 epoch.

Vì vậy “chạy default” không đồng nghĩa “tái lập cấu hình paper”. Khi so sánh thí nghiệm phải lưu nguyên command, không chỉ tên checkpoint.

## 10. Inference và metric

### 10.1 `utils/inference.py`

`predict_density` là shared path cho test, validation và notebook:

1. `extract_patches` tách ảnh thành crop 384, cho phép overlap theo stride.
2. Chạy model theo chunk `batch_size`.
3. `reassemble_patches` bilinear-upsample density 48→384 và chia 64 để bảo toàn mass khi scale không gian 8×8.
4. Các vùng patch overlap được lấy trung bình bằng `norm_map`.
5. Chia tiếp `DENSITY_SCALE=60` để đưa density về đơn vị count.

`predict_count` chỉ sum toàn bộ density kết quả.

### 10.2 `test.py`

- FSC-147/CARPK dùng legacy metric MAE và RMSE.
- IDCIA có generic prompt `cell` và prompt riêng theo staining; có thể chạy `generic`, `specific` hoặc `both`.
- IDCIA báo MAE, RMSE, mean APE trên GT>0, WAPE, median APE, số sample GT=0 bị loại khỏi percentage metric và breakdown theo staining.
- CSV lưu prediction/error từng ảnh.

`evaluate_legacy` vẫn chia `abs_error / gt`; nếu một dataset legacy có GT=0 thì sẽ chia 0. FSC-147/CARPK test thông thường không có trường hợp đó, còn IDCIA đã có xử lý riêng.

## 11. Bản đồ từng thư mục và file

### 11.1 Root

| File | Nhãn | Vai trò |
| --- | --- | --- |
| `train.py` | [SỬA] | CLI train, resolve asset/save path, subset controls và DUMLO hyperparameters |
| `test.py` | [SỬA] | shared model builder/inference, FSC/CARPK/IDCIA evaluation |
| `visualize.py` | [SỬA] | script một ảnh FSC-147, xuất `den.jpg` |
| `simple_subset_test.py` | [SỬA] | chạy FSC-147-S JSON bằng shared inference path |
| `FSC-147-S.json` | [GỐC] | annotation FSC-147-S v2 từ upstream |
| `environment.yaml` | [SỬA] | env Windows/legacy Python 3.8, Torch 1.11, dependency source pin |
| `requirements-colab.txt` | [MỚI] | bridge cho Colab Python/Torch mới, không thay Torch do Colab cấp |
| `pytest.ini` | [MỚI] | đặt `pythonpath = .` |
| `.gitignore` | [MỚI] | loại model/data/checkpoint/log/result/local dependency khỏi Git |
| `README.md` | [SỬA] | attribution, portable assets, Windows/Colab hướng dẫn |
| `THIRD_PARTY_NOTICES.md` | [MỚI] | provenance/license của LDM, OpenAI, x-transformers và dependency |

`train.py --beta` hiện không được dùng ở đâu trong trainer. Nó là CLI thừa/stale từ upstream. Tương tự, code bề ngoài cho phép đổi `--downsample-ratio`, nhưng một số chỗ vẫn hard-code 8; xem mục 14.

### 11.2 `asset/` và `configs/`

- `asset/teaser.jpg`, `asset/visualization.jpg`: hình từ upstream README/paper.
- `asset/FSC-147-S-v1.json`: bản subset cũ dùng trong paper/review stage.
- `configs/v1-inference.yaml`: kiến trúc SD v1.5 gồm `LatentDiffusion`, 4-channel latent U-Net, `AutoencoderKL`, `FrozenCLIPEmbedder`. File này giữ nguyên upstream; builder chỉ inject CLIP local path/device vào memory.

### 11.3 `models/`

- `build.py` **[MỚI]**: canonical construction, local asset validation, strict T2ICount checkpoint load và `position_ids` reconciliation.
- `reg_model.py` **[SỬA]**: `Count`, four-level decoder, fused attention; kiến trúc số học chính giữ như upstream, chỉ thay loader/config plumbing.
- `decoder.py` **[GỐC]**: HSCM/SEM/SCM và Counter; không bị sửa so với upstream.
- `diff_unet.py` **[SỬA rất hẹp]**: feature/attention hooks của T2ICount; chỉ bỏ import private Transformers không dùng để tương thích version mới.
- `__init__.py` **[MỚI]**: buộc local package resolution ổn định trên Colab.

### 11.4 `losses/`

- `dumlo.py` **[MỚI]**: toàn bộ point uncertainty, Trihorn, analytical gradient và composite DUMLO loss.
- `__init__.py` **[MỚI]**: export `DUMLOLoss`.

### 11.5 `datasets/`

- `dataset.py` **[SỬA]**: ObjectCount gốc + tokenizer local + IDCIA + point propagation cho DUMLO.
- `carpk.py` **[SỬA hẹp]**: tokenizer reuse/local-only.
- `__init__.py` **[MỚI]**: local package marker.

### 11.6 `utils/`

| File | Nhãn | Vai trò |
| --- | --- | --- |
| `regression_trainer.py` | [SỬA lớn] | train/val/test, RRC, baseline/DUMLO selection, subset, progress, checkpoint/resume |
| `inference.py` | [MỚI] | shared patch inference và `/60` scaling |
| `paths.py` | [MỚI] | external asset layout và fail-fast absolute errors |
| `checkpoints.py` | [MỚI] | trusted legacy `torch.load`, tương thích PyTorch trước/sau 2.6 |
| `clip.py` | [MỚI] | local-only CLIP tokenizer |
| `tools.py` | [GỐC] | patch extraction/reassembly |
| `ssim_loss.py` | [GỐC] | multi-scale SSIM component của baseline regression |
| `helper.py` | [GỐC] | rolling checkpoint list và average meter |
| `trainer.py` | [SỬA hẹp] | base trainer ghi vào explicit save root |
| `logger.py` | [GỐC] | file + console logging |
| `__init__.py` | [MỚI] | package marker |

### 11.7 `ldm/`

Đây chủ yếu là vendored/adapted Stable Diffusion/Latent Diffusion code, không phải các module mới của T2ICount.

| File/nhóm | Active trong counting path? | Vai trò |
| --- | --- | --- |
| `ldm/util.py` | Có | instantiate object từ target string trong YAML |
| `models/autoencoder.py` | Có một phần | `AutoencoderKL.encode`; VQ training và decoder path không dùng |
| `models/diffusion/ddpm.py` | Có khi dựng model | `LatentDiffusion`, `DiffusionWrapper`; phần diffusion training/sampling gần như không dùng trong T2ICount forward |
| `models/diffusion/ddim.py`, `plms.py` | Không trong active path | sampler nhiều bước kế thừa từ LDM |
| `models/diffusion/classifier.py` | Không | classifier utilities kế thừa |
| `modules/attention.py` | Có | SpatialTransformer/CrossAttention trong SD U-Net; forward bị hook để thu attention |
| `modules/diffusionmodules/openaimodel.py` | Có | SD U-Net blocks; forward bị `register_hier_output` thay |
| `modules/diffusionmodules/model.py` | Có một phần | VAE Encoder/Decoder blocks; encoder dùng, VAE decoder bị xóa sau load |
| `modules/diffusionmodules/util.py` | Có | timestep embedding, gradient checkpoint và layer helpers |
| `modules/distributions/distributions.py` | Có | `DiagonalGaussianDistribution.mode()` cho latent |
| `modules/encoders/modules.py` | Có một class | `FrozenCLIPEmbedder`; các BERT/OpenAI CLIP image/text alternative không nằm trong YAML active |
| `modules/ema.py` | Không | LDM EMA helper |
| `modules/x_transformer.py` | Không trong active config | x-transformers implementation kế thừa |

Hai sửa đổi local trong `ldm/` chỉ phục vụ compatibility:

- `ddpm.py`: import `rank_zero_only` được fallback giữa Lightning cũ/mới.
- `encoders/modules.py`: `FrozenCLIPEmbedder` buộc explicit local path và `local_files_only=True`.

### 11.8 `scripts/` và `tools/`

- `scripts/check_assets.py` **[MỚI]**: kiểm tra asset tree, CUDA/device, và tùy chọn thử load CLIP offline.
- `tools/diagnose_dumlo_gradients.py` **[WORKTREE]**: đo gradient riêng của weighted Count/OT/TV đối với `pred_den`, cosine giữa các gradient, cancellation ratio và thống kê aggregate; không gọi optimizer và xác nhận model parameter không tích gradient.
- `tools/__init__.py` **[WORKTREE]**: package marker.

Diagnostic dùng batch 1 có chủ đích vì decomposition từ scalar batch mean trở về từng sample sẽ khác nếu batch>1.

### 11.9 `notebooks/`

- `train_colab.ipynb` **[MỚI + WORKTREE]**: mount Drive, copy/extract asset ZIP sang local SSD, clone/checkout branch, install, validate, optional smoke, stream training output và lưu status. Notebook hiện chứa output thật của một run DUMLO.
- `visualize_results.ipynb` **[MỚI]**: đọc CSV IDCIA, so sánh raw/autocontrast và generic/specific, chọn failure cases, chỉ re-inference các ảnh đã chọn để vẽ density map.

Notebook là orchestration; model/loss/inference logic vẫn nằm trong module `.py`.

### 11.10 `tests/`

| File | Phạm vi |
| --- | --- |
| `test_infrastructure.py` | local package, asset layout, save/read root separation, prompt mask, scaling, trusted checkpoint loader |
| `test_checkpoint_resume.py` | full-state/legacy resume, RNG và CPU map-location |
| `test_t2icount_checkpoint_compatibility.py` | allowlist duy nhất cho text `position_ids`, vẫn strict với key lạ/weight thiếu |
| `test_smoke_train_samples.py` | first-N smoke limiter chỉ tác động train |
| `test_train_samples.py` | deterministic random subset, private RNG, mutual exclusion |
| `test_progress_reporting.py` | tqdm không đổi order/length |
| `test_notebook_workflow.py` | notebook flags, stdout/stderr và live Popen contract |
| `test_dumlo.py` | discrete mass, sampling, Trihorn, gradient, weights, zero-point, baseline unchanged |
| `test_dumlo_dataset.py` | point transform, mosaic, negative prompt, RNG sequence |
| `test_dumlo_gradient_diagnostics.py` | **[WORKTREE]** helper math và CLI default của diagnostic |

### 11.11 `docs/` và `third_party/`

- `docs/provenance.md`: audit nguồn code và tình trạng license upstream chưa rõ.
- `docs/dependency_audit.md`: dependency versions/risk khi upgrade.
- `docs/offline_asset_audit.md`: active offline loading chain.
- `docs/training_checkpoints.md`: checkpoint format và Colab/Drive split.
- `third_party/licenses/`: bản license của Latent Diffusion, OpenAI và x-transformers.

Repo upstream không có license grant rõ tại mốc audit; các notice local không tự cấp quyền cho phần code/asset kế thừa.

## 12. Toàn bộ thay đổi so với repo gốc, theo nhóm

### 12.1 Thay đổi nghiên cứu

1. Thêm DUMLO loss opt-in và toàn bộ point propagation.
2. Thêm `lambda_count` cấu hình được, mặc định 1.0 để giữ hành vi DUMLO trước đó.
3. Thêm diagnostic gradient Count/OT/TV trong worktree.

Không có thay đổi local đối với `models/decoder.py`, tức HSCM/Counter vẫn là code upstream.

### 12.2 Portability/offline

1. Asset root ngoài repo qua `T2ICOUNT_ASSET_ROOT`/`--asset-root`.
2. Local-only CLIP tokenizer/text encoder; thiếu asset thì fail ngay bằng đường dẫn tuyệt đối.
3. Shared `build_t2icount` và shared inference API.
4. Tách runtime reads ở local SSD khỏi checkpoint writes trên Drive.
5. Package markers để tránh import nhầm package `datasets` trên Colab.

### 12.3 Checkpoint compatibility và resume

1. Explicit `weights_only=False` cho trusted legacy checkpoint trên PyTorch 2.6+, fallback cho Torch 1.11.
2. Reconcile hai chiều đúng buffer text `position_ids`; không lọc rộng state dict và vẫn `strict=True`.
3. Full-state training resume kèm RNG và atomic save.

### 12.4 Experiment controls/observability

1. `--smoke-train-samples` first-N cho hạ tầng.
2. `--train-samples` random deterministic cho thí nghiệm giới hạn compute.
3. tqdm cho train/val/test.
4. Notebook dùng `python -u` + `subprocess.Popen`, stream stdout/stderr chung, lưu command/return code/time.

### 12.5 Dataset/evaluation

1. Giữ FSC-147 và CARPK nhưng chuyển sang tokenizer/build/inference chung.
2. Thêm IDCIA raw/autocontrast, generic/specific prompt, metric an toàn với GT=0 và CSV export.
3. Thêm notebook qualitative analysis.

### 12.6 Repository hygiene

1. Xóa tracked `logs/train.log` của upstream khỏi branch hiện tại; link lại upstream log trong README.
2. Ignore model/data/log/result/local checkout lớn.
3. Thêm provenance, notices, dependency audit và tests.

## 13. Lịch sử 15 commit sau upstream

| Commit | Nội dung chính |
| --- | --- |
| `2addc67` | Portable offline assets, Colab train, IDCIA/shared build/inference, resume infrastructure |
| `121e494` | Attribution, provenance, third-party notices, dependency source pin |
| `62ed1d9` | Dùng system `unzip` trong Colab bootstrap |
| `8291e9a` | Sửa bug trong notebook bootstrap |
| `57288d5` | Local package `__init__.py` và import regression test |
| `3ed71de` | PyTorch 2.6 trusted legacy checkpoint compatibility |
| `65c1d04` | Tách local runtime assets khỏi Drive output root |
| `1b24642` | Xử lý checkpoint-only CLIP `position_ids` |
| `e16c2e9` | `--smoke-train-samples` |
| `8da9936` | Sửa restore CPU Torch RNG khi resume |
| `e607f5f` | `--train-samples` + deterministic subset seed |
| `50644bd` | tqdm/live Colab experiment workflow |
| `9105e4d` | Reconcile `position_ids` hai chiều giữa Transformers versions |
| `742914b` | DUMLO loss + point propagation + tests |
| `8706cf7` | Configurable DUMLO count-loss weight |

## 14. Trạng thái thí nghiệm/artifact hiện có

### 14.1 Run nằm trong notebook chưa commit

Cell training hiện cấu hình:

```text
name/content: dumlo_lot001_1000x10
loss-mode: dumlo
train samples: 1000, subset seed 3407
epochs: 10, batch size 1
lambda_count: 1.0
lambda_ot: 0.01
lambda_tv: 0.01
epsilon: 10
Trihorn iterations: 100
augmented points: 10
radius factor: 0.5
```

Output đã lưu trong notebook báo `RETURN CODE: 0`, tổng thời gian 9710.9 giây (161.8 phút). Validation:

| Epoch | Val MAE | Val RMSE | Ghi chú |
| ---: | ---: | ---: | --- |
| 7 | **51.33** | **130.60** | best; test MAE 50.19, RMSE 153.73 |
| 8 | 55.18 | 135.38 | không cải thiện |
| 9 | 58.05 | 135.33 | không cải thiện |

Đây là run giới hạn 1000 sample/10 epoch, không phải reproduction paper T2ICount 400 epoch/full train.

### 14.2 Hai CSV IDCIA bị Git ignore

`results/idcia_baseline_500x10.csv` và `results/idcia_dumlo_500x10.csv` đều có 53 ảnh. Metric được tính lại trực tiếp từ CSV:

| File | Prompt | MAE | RMSE | WAPE | Mean prediction |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline_500x10 | generic `cell` | 325.17 | 378.72 | 357.18% | 416.21 |
| baseline_500x10 | staining-specific | 285.83 | 338.36 | 313.96% | 376.86 |
| dumlo_500x10 | generic `cell` | 90.91 | 134.15 | 99.86% | 0.137 |
| dumlo_500x10 | staining-specific | 90.82 | 134.09 | 99.76% | 0.248 |

Mean GT là 91.04. Vì DUMLO CSV dự đoán gần 0 cho gần như mọi ảnh, MAE thấp hơn baseline ở đây **không có nghĩa model DUMLO tốt**; nó phản ánh baseline overcount rất mạnh còn DUMLO bị density collapse. Chênh lệch generic/specific của DUMLO gần như bằng 0 cũng cho thấy checkpoint đó chưa có prompt sensitivity hữu ích trên IDCIA.

Tên file gợi ý 500 sample × 10 epoch, nhưng CSV không chứa command, commit SHA hay checkpoint path. Không nên dùng chúng làm bằng chứng tái lập cấu hình nếu thiếu log đi kèm.

## 15. Kiểm chứng hiện tại

Các lệnh read-only/validation đã chạy ở trạng thái tài liệu này:

- `python -m py_compile` cho train/test/DUMLO/gradient diagnostic: **pass**.
- `git diff --check`: **pass**.
- Pytest sau khi bỏ `SSLKEYLOGFILE=C:\ssl-keys.log`: **62 passed, 1 failed**.
- Failure duy nhất: `test_notebook_workflow.py` vẫn tìm cell có `baseline_500x10`, trong khi worktree notebook đã đổi cell đó sang DUMLO `lambda_ot_001_1000x10`.

Lần pytest đầu tiên không collect được vì biến môi trường `SSLKEYLOGFILE` trỏ tới `C:\ssl-keys.log` không có quyền ghi. Đây là lỗi shell/environment, không phải lỗi import của repo.

Test CPU không phải bằng chứng full GPU training. Bằng chứng GPU hiện có là output run Colab được embed trong notebook; checkpoint thật nằm trên Drive và không nằm trong Git checkout.

## 16. Các điểm dễ gây “lạc” hoặc dễ cấu hình sai

1. **Paper và code không hoàn toàn literal**: single-step, PNA threshold và một số DUMLO ký hiệu có khác biệt như đã nêu.
2. **Crop/downsample trông có vẻ configurable nhưng thực chất gần như khóa ở 384/8**:
   - `Decoder.forward` hard-code resize similarity về `(24,24)`;
   - `train_transform_density` hard-code attention map `/8`;
   - `reassemble_patches` hard-code upsample 8× và chia 64;
   - `test.py` hard-code crop 384.
   Đổi `--crop-size` hay `--downsample-ratio` cần audit toàn bộ graph, không chỉ đổi CLI.
3. **`--beta` không được dùng**; không nên xem nó là hyperparameter đang hoạt động.
4. **`MSE` trong log là RMSE**.
5. **Notebook chứa output rất lớn**: worktree diff hiện tăng hơn 15 nghìn dòng JSON chủ yếu do output/run state. Source và experiment record nên được tách nếu muốn Git diff dễ đọc.
6. **Diagnostic gradient đang untracked**: nó chưa thuộc lịch sử nhánh cho tới khi được commit.
7. **CSV results bị ignore**: dễ mất provenance. Mỗi run nên có manifest gồm commit SHA, command, checkpoint path/hash, dataset/preprocess, seed và metric summary.
8. **Environment có hai mục tiêu khác nhau**: `environment.yaml` là legacy Windows path; `requirements-colab.txt` là compatibility bridge. Không nên coi chúng là hai bản cài tương đương số học.
9. **SD checkpoint load và T2ICount checkpoint load có strictness khác nhau**:
   - SD v1.5 được load `strict=False` vào full LatentDiffusion theo kiểu upstream;
   - T2ICount weights được load `strict=True`, chỉ reconcile buffer `position_ids` allowlisted.
10. **Density scale 60 là invariant xuyên suốt**: train target ×60, count/prediction ÷60. Bỏ một phía sẽ làm metric lệch khoảng 60×.

## 17. Mental model ngắn để tiếp tục nghiên cứu

Khi thay đổi repo, hãy xác định thay đổi thuộc đúng một trong bốn lớp:

```text
[Data/annotation]
    ObjectCount augmentation, point propagation, prompt semantics

[Backbone/architecture]
    VAE + CLIP + U-Net hooks + HSCM + Counter

[Training objective]
    baseline CUT-like L_reg OR DUMLO L_reg
    + RRC stage 1 + RRC stage 2

[Infrastructure/evaluation]
    assets, checkpoints, subset, notebook, inference, IDCIA, tests
```

Nếu mục tiêu chỉ là ablation DUMLO, thay đổi nên nằm ở lớp **Training objective** và test point/loss tương ứng. Nếu chạm `models/decoder.py`, `models/diff_unet.py`, prompt mask, `/60`, density preprocessing hoặc inference reassembly thì đó không còn là ablation loss thuần túy nữa.

## 18. Gợi ý tối thiểu để repo bớt rối từ đây

1. Tạo `experiments/<run-name>/manifest.yaml` nhỏ, track bằng Git, nhưng vẫn để checkpoint/log lớn ngoài Git.
2. Manifest ghi: commit SHA, dirty diff flag, command, asset/checkpoint hashes, train indices/seed, metric và link output Drive.
3. Tách notebook “template sạch” khỏi notebook “executed output”; không dùng chính notebook output làm nguồn duy nhất của lịch sử thí nghiệm.
4. Cập nhật hoặc parameterize `test_notebook_workflow.py` khi đổi pilot cell từ baseline sang DUMLO.
5. Trước mỗi ablation, ghi rõ invariant: architecture, RRC, sampling, optimizer, data transforms, subset indices, inference và metric.
