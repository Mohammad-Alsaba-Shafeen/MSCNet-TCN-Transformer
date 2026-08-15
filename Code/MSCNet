#For 2a

import os
import numpy as np
import pandas as pd
import random, datetime, time, warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn, Tensor
from torch.autograd import Variable
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange

from pandas import ExcelWriter
from torchsummary import summary
from torch.backends import cudnn

# Make sure utils.py is in your working directory
from utils import numberClassChannel, load_data_evaluate, calMetrics, calculatePerClass

gpu_number = 0
os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
cudnn.benchmark     = False
cudnn.deterministic = True

# ═══════════════════════════════════════════════════════════════════
# Attention Blocks (CBAM)
# ═══════════════════════════════════════════════════════════════════
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.se(x).view(x.shape[0], x.shape[1], 1, 1)

class CBAMSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg     = torch.mean(x, dim=1, keepdim=True)
        mx, _   = torch.max(x,  dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

class CBAMBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.se      = SEBlock(channels)
        self.spatial = CBAMSpatialAttention()
    def forward(self, x):
        return self.spatial(self.se(x))

# ═══════════════════════════════════════════════════════════════════
# Spatial Extractor (MSCNet)
# ═══════════════════════════════════════════════════════════════════
class MSCNet(nn.Module):
    def __init__(self, f1=16, D=2, pooling_size=50, dropout_rate=0.5, number_channel=22):
        super().__init__()
        self.D = D
        def make_branch(kernel):
            conv_spatial = nn.Conv2d(f1, f1 * D, (number_channel, 1), (1, 1), groups=f1)
            conv_spatial.register_forward_pre_hook(self._max_norm_spatial)
            return nn.Sequential(
                nn.Conv2d(1, f1, (1, kernel), (1, 1), padding='same'),
                conv_spatial,
                nn.BatchNorm2d(f1 * D),
                nn.ELU(),
                CBAMBlock(f1 * D),
                nn.AvgPool2d((1, pooling_size)),
                nn.Dropout(dropout_rate),
            )
        self.cnn1 = make_branch(125)
        self.cnn2 = make_branch(62)
        self.cnn3 = make_branch(31)
        self.proj = Rearrange('b e (h) (w) -> b (h w) e')

    def _max_norm_spatial(self, module, input):
        with torch.no_grad():
            norm = module.weight.data.norm(2, dim=(1,2,3), keepdim=True)
            desired = torch.clamp(norm, max=1.0)
            module.weight.data *= desired / (norm + 1e-8)

    def forward(self, x):
        return self.proj(torch.cat([self.cnn1(x), self.cnn2(x), self.cnn3(x)], dim=1))

# ═══════════════════════════════════════════════════════════════════
# Temporal Extractor (TCN)
# ═══════════════════════════════════════════════════════════════════
class CausalConv1d(nn.Conv1d):
    def forward(self, x):
        return super().forward(F.pad(x, ((self.kernel_size[0] - 1) * self.dilation[0], 0)))

class _TCNBlock(nn.Module):
    def __init__(self, input_dimension, depth, kernel_size, filters, drop_prob):
        super().__init__()
        self.activation = nn.ELU()
        self.downsample = nn.Conv1d(input_dimension, filters, 1, bias=False) if input_dimension != filters else None
        self.layers = nn.ModuleList([
            nn.Sequential(
                CausalConv1d(input_dimension if i == 0 else filters, filters, kernel_size, dilation=2**i, bias=False),
                nn.BatchNorm1d(filters), self.activation, nn.Dropout(drop_prob),
                CausalConv1d(filters, filters, kernel_size, dilation=2**i, bias=False),
                nn.BatchNorm1d(filters), self.activation, nn.Dropout(drop_prob),
            ) for i in range(depth)
        ])
    def forward(self, x):
        # 1. Compute features across all 3 temporal scales
        out1 = self.cnn1(x)
        out2 = self.cnn2(x)
        out3 = self.cnn3(x)

        # 2. Concatenate them along the channel dimension (dim=1)
        # This turns 32 + 32 + 32 channels into the expected 96 channels!
        out = torch.cat([out1, out2, out3], dim=1)

        return out
# ═══════════════════════════════════════════════════════════════════
# Transformer Blocks
# ═══════════════════════════════════════════════════════════════════
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size  = emb_size
        self.num_heads = num_heads
        self.keys      = nn.Linear(emb_size, emb_size)
        self.queries   = nn.Linear(emb_size, emb_size)
        self.values    = nn.Linear(emb_size, emb_size)
        self.att_drop  = nn.Dropout(dropout)
        self.proj      = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        q = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(self.keys(x),    "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(self.values(x),  "b n (h d) -> b h n d", h=self.num_heads)
        att = F.softmax(torch.einsum('bhqd,bhkd->bhqk', q, k) / self.emb_size**0.5, dim=-1)
        out = rearrange(torch.einsum('bhal,bhlv->bhav', self.att_drop(att), v), "b h n d -> b n (h d)")
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, emb_size, expansion=4, drop_p=0.5):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size * expansion),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(emb_size * expansion, emb_size),
            nn.Dropout(drop_p),
        )
    def forward(self, x):
        return self.ff(x)

class ResidualAdd(nn.Module):
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn   = fn
        self.drop = nn.Dropout(drop_p)
        self.norm = nn.LayerNorm(emb_size)
    def forward(self, x, **kwargs):
        return self.norm(self.drop(self.fn(x, **kwargs)) + x)

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=4, drop_p=0.5):
        super().__init__(
            ResidualAdd(MultiHeadAttention(emb_size, num_heads, drop_p), emb_size, drop_p),
            ResidualAdd(FeedForward(emb_size, drop_p=drop_p), emb_size, drop_p),
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, heads, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size, heads) for _ in range(depth)])
#
class AdaptiveFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, tcm_feat, tr_feat):

        concat = torch.cat([tcm_feat, tr_feat], dim=-1)

        alpha = self.gate(concat)

        fused = alpha * tcm_feat + (1 - alpha) * tr_feat

        return fused
# ═══════════════════════════════════════════════════════════════════
# TCANet (Shallow Transformer Variant)
# ═══════════════════════════════════════════════════════════════════
class TCANet_ShallowTransformer(nn.Module):
    def __init__(self, n_chans=22, out_features=4, drop_prob=0.5, depth=2, kernel_size=4, filters=16, max_norm_const=0.25):
        super().__init__()
        self.max_norm_const = max_norm_const
        D = 2

        # 1. Spatial Extractor
        self.mseegnet = MSCNet(number_channel=n_chans, dropout_rate=drop_prob, pooling_size=POOLING_SIZE, f1=F1, D=D)

        # 2. Temporal Extractor (Shallow)
        self.tcn_block = _TCNBlock(3 * (F1 * D), depth, kernel_size, filters, drop_prob=0.25)

        # 3. Shallow Transformer
        self.sa = TransformerEncoder(heads=HEADS, depth=DEPTH, emb_size=filters)
        self.fusion = AdaptiveFusion(filters)

        # Positional Embedding
        seq_len = 1000 // POOLING_SIZE
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, filters))

        self.drop    = nn.Dropout(0.25)
        self.flatten = nn.Flatten()

        # Multiply filters * 2 because we concatenate TCN and Transformer outputs
        self.classifier = nn.Linear(filters * seq_len, out_features)
        self.classifier.register_forward_pre_hook(self._max_norm)

       def _max_norm(self, module, input):
        with torch.no_grad():
            norm = self.classifier.weight.data.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, max=self.max_norm_const)
            self.classifier.weight.data *= desired / (norm + 1e-8)

    def forward(self, x):

        x = self.mseegnet(x)

        x_tcn = self.tcn_block(x)

        x_pos = x_tcn + self.pos_embedding

        sa = self.sa(x_pos)

        fused = self.fusion(
            x_tcn,
            sa
        )

        features = self.flatten(
            self.drop(
                fused
            )
        )

        return self.classifier(features), features

class EEGTransformer(nn.Module):
    def __init__(self, parameters, database_type='A'):
        super().__init__()
        self.number_class, self.number_channel = numberClassChannel(database_type)
        parameters.number_channel = self.number_channel
        self.cnn = TCANet_ShallowTransformer(n_chans=self.number_channel, out_features=self.number_class)
    def forward(self, x):
        out, features = self.cnn(x)
        return features, out

# ═══════════════════════════════════════════════════════════════════
# Augmentations
# ═══════════════════════════════════════════════════════════════════
def mixup_data(x, y, alpha=0.2):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0)).cuda()
    return lam * x + (1 - lam) * x[index], y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ═══════════════════════════════════════════════════════════════════
# Experiment class
# ═══════════════════════════════════════════════════════════════════
class ExP():
    def __init__(self, nsub, data_dir, result_name, parameters, evaluate_mode='LOSO-no', dataset_type='A', n_fold=0):
        self.n_fold = n_fold
        self.dataset_type = dataset_type
        self.batch_size = parameters.batch_size
        self.lr = parameters.learning_rate
        self.n_epochs = parameters.epochs
        self.nSub = nsub
        self.nFold = n_fold
        self.number_augmentation = parameters.number_aug
        self.number_seg = parameters.number_seg
        self.root = data_dir
        self.result_name = result_name
        self.evaluate_mode = evaluate_mode
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_cls = torch.nn.CrossEntropyLoss(label_smoothing=0.05).cuda()
        self.number_class, self.number_channel = numberClassChannel(dataset_type)
        self.model = EEGTransformer(database_type=self.dataset_type, parameters=parameters).cuda()
        self.model_filename = (self.result_name + f'/model_nsub_{self.nSub}_nfold_{n_fold+1}.pth')

    def interaug(self, timg, label):
        aug_data, aug_label = [], []
        n_per = self.number_augmentation * int(self.batch_size / self.number_class)
        seg   = 1000 // self.number_seg
        for cls in range(self.number_class):
            idx  = np.where(label == cls + 1)
            tmp  = timg[idx]
            aug  = np.zeros((n_per, 1, self.number_channel, 1000))
            for ri in range(n_per):
                ri_idx = np.random.randint(0, tmp.shape[0], self.number_seg)
                for rj in range(self.number_seg):
                    aug[ri, :, :, rj*seg:(rj+1)*seg] = tmp[ri_idx[rj], :, :, rj*seg:(rj+1)*seg]
            aug_data.append(aug)
            aug_label.append(label[idx][:n_per])
        aug_data  = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        shuf      = np.random.permutation(len(aug_data))
        return (torch.from_numpy(aug_data[shuf]).cuda().float(),
                torch.from_numpy(aug_label[shuf] - 1).cuda().long())

    def get_source_data(self):
        (self.train_data, self.train_label, self.test_data,  self.test_label) = load_data_evaluate(
            self.root, self.dataset_type, self.nSub, mode_evaluate=self.evaluate_mode)

        self.train_data  = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.transpose(self.train_label)
        self.allData     = self.train_data
        self.allLabel    = self.train_label[0]

        shuf = np.random.permutation(len(self.allData))
        self.allData  = self.allData[shuf]
        self.allLabel = self.allLabel[shuf]

        self.test_data  = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)
        self.testData   = self.test_data
        self.testLabel  = self.test_label[0]

        mu  = np.mean(self.allData)
        std = np.std(self.allData)
        self.allData  = (self.allData  - mu) / std
        self.testData = (self.testData - mu) / std
        return self.allData, self.allLabel, self.testData, self.testLabel

    def test_model(self, model, dataloader):
        model.eval()
        out_list, lbl_list = [], []
        with torch.no_grad():
            for img, label in dataloader:
                img   = img.type(self.Tensor).cuda()
                label = label.type(self.LongTensor).cuda()
                _, cls = model(img)
                out_list.append(cls); lbl_list.append(label)
        cls_all  = torch.cat(out_list)
        lbl_all  = torch.cat(lbl_list)
        val_loss = self.criterion_cls(cls_all, lbl_all)
        val_pred = torch.max(cls_all, 1)[1]
        val_acc  = float((val_pred == lbl_all).cpu().numpy().astype(int).sum()) / float(lbl_all.size(0))
        return val_acc, val_loss, val_pred

    def train(self):
        timg, label, test_data, test_label = self.get_source_data()

        data_pc, lbl_pc = [], []
        for cls in range(self.number_class):
            idx = np.where(label == cls + 1)
            data_pc.append(timg[idx]); lbl_pc.append(label[idx])

        tr_d, tr_l, val_d, val_l = [], [], [], []
        seed = 1234 + self.nSub
        for cls in range(self.number_class):
            n = len(data_pc[cls]); n_val = n // 5
            np.random.seed(seed + cls)
            idx_s = np.random.permutation(n)
            idx_v = list(range(self.n_fold * n_val, n if self.n_fold == 4 else (self.n_fold+1)*n_val))
            idx_t = [i for i in range(n) if i not in idx_v]
            tr_d.append(data_pc[cls][idx_s[idx_t]])
            tr_l.append(lbl_pc[cls][idx_s[idx_t]])
            val_d.append(data_pc[cls][idx_s[idx_v]])
            val_l.append(lbl_pc[cls][idx_s[idx_v]])

        img       = np.concatenate(tr_d)
        label     = np.concatenate(tr_l)
        val_data  = np.concatenate(val_d)
        val_label = np.concatenate(val_l)

        self.dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(img), torch.from_numpy(label - 1)),
            batch_size=self.batch_size, shuffle=True)

        self.val_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(val_data), torch.from_numpy(val_label - 1)),
            batch_size=self.batch_size, shuffle=True)

        test_t     = torch.from_numpy(test_data)
        test_lbl_t = torch.from_numpy(test_label - 1)
        self.test_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_t, test_lbl_t),
            batch_size=self.batch_size, shuffle=False)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=1e-4)

        warmup = 20
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.05, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.n_epochs - warmup, eta_min=1e-5)
            ],
            milestones=[warmup])

        test_var = Variable(test_t.type(self.Tensor))
        tlbl_var = Variable(test_lbl_t.type(self.LongTensor))

        best_epoch = 0
        min_loss   = 100.0
        result_process = []

        for e in range(self.n_epochs):
            self.model.train()
            out_list, lbl_list = [], []

            for img_b, lbl_b in self.dataloader:
                img_v = Variable(img_b.type(self.Tensor))
                lbl_v = Variable(lbl_b.type(self.LongTensor))

                # Segment-mixing augmentation ONLY
                aug_data, aug_label = self.interaug(self.allData, self.allLabel)

                img_v = torch.cat((img_v, aug_data))
                lbl_v = torch.cat((lbl_v, aug_label))

                if e > warmup:
                    img_mix, lbl_a, lbl_b_mix, lam = mixup_data(img_v, lbl_v, alpha=0.2)
                    _, outputs = self.model(img_mix)
                    loss = mixup_criterion(self.criterion_cls, outputs, lbl_a, lbl_b_mix, lam)
                else:
                    _, outputs = self.model(img_v)
                    loss = self.criterion_cls(outputs, lbl_v)

                out_list.append(outputs.detach()); lbl_list.append(lbl_v)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

            self.model.eval()
            val_acc, val_loss, _ = self.test_model(self.model, self.val_dataloader)

            all_out = torch.cat(out_list)
            all_lbl = torch.cat(lbl_list)
            tr_pred = torch.max(all_out, 1)[1]
            tr_acc  = float((tr_pred == all_lbl).cpu().numpy().astype(int).sum()) / float(all_lbl.size(0))

            ep = {'epoch': e, 'val_acc': val_acc, 'val_loss': val_loss.detach().cpu().numpy(),
                  'train_acc': tr_acc, 'train_loss': loss.detach().cpu().numpy()}

            if min_loss > val_loss:
                min_loss   = val_loss
                best_epoch = e
                torch.save(self.model, self.model_filename)
                test_acc, _, _ = self.test_model(self.model, self.test_dataloader)
                print(f"sub{self.nSub} ep{e:4d} | tr:{tr_acc:.4f} val_acc:{val_acc:.4f} val_loss:{val_loss:.6f} test:{test_acc:.4f}")

            result_process.append(ep)

        self.model = torch.load(self.model_filename, weights_only=False).cuda()
        self.model.eval()

        out_list = []
        feat_list = []

        with torch.no_grad():
            for img_b, _ in self.test_dataloader:
                features, outputs = self.model(Variable(img_b.type(self.Tensor)).cuda())
                out_list.append(outputs)
                feat_list.append(features)

        outputs = torch.cat(out_list)
        features_all = torch.cat(feat_list)

        y_pred = torch.max(outputs, 1)[1]

        test_acc = float((y_pred == tlbl_var).cpu().numpy().astype(int).sum()) / float(tlbl_var.size(0))

        # ✅ RETURN FEATURES INSTEAD OF OUTPUTS
        return test_acc, tlbl_var, y_pred, pd.DataFrame(result_process), best_epoch, features_all

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main(dirs, parameters, evaluate_mode='subject-dependent', dataset_type='A'):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    result_write  = ExcelWriter(dirs + "/result_metric.xlsx")
    process_write = ExcelWriter(dirs + "/process_train.xlsx")
    pred_write    = ExcelWriter(dirs + "/pred_true.xlsx")

    subjects_result = []

    # ===== GLOBAL STORAGE =====
    all_true = []
    all_pred = []
    all_features = []

    for i in range(parameters.subject_number):
        starttime = datetime.datetime.now()

        seed_n = np.random.randint(2024)
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)

        print(f'\nSubject {i+1}  seed={seed_n}')

        subjects_result_fold = []

        for n_fold in range(1):
            exp = ExP(i + 1, DATA_DIR, dirs, parameters,
                      evaluate_mode=evaluate_mode,
                      dataset_type=dataset_type,
                      n_fold=n_fold)

            testAcc, Y_true, Y_pred, df_proc, best_epoch, features_all = exp.train()

            true_cpu = Y_true.cpu().numpy().astype(int)
            pred_cpu = Y_pred.cpu().numpy().astype(int)
            features_cpu = features_all.cpu().numpy()

            # ===== STORE GLOBAL =====
            all_true.extend(true_cpu)
            all_pred.extend(pred_cpu)
            all_features.append(features_cpu)

            # (Optional: still save per-subject if needed)
            pd.DataFrame({'pred': pred_cpu, 'true': true_cpu}).to_excel(pred_write, sheet_name=str(i+1))
            df_proc.to_excel(process_write, sheet_name=str(i+1))

            acc, prec, rec, f1, kappa = calMetrics(true_cpu, pred_cpu)

            result = {
                'accuray': acc*100,
                'precision': prec*100,
                'recall': rec*100,
                'f1': f1*100,
                'kappa': kappa*100
            }

            print(f'Subject {i+1}: acc={acc*100:.2f}%  kappa={kappa:.4f}')
            subjects_result_fold.append(result)

        df = pd.DataFrame(subjects_result_fold)
        pd.DataFrame([result]).to_excel(result_write, index=False, sheet_name=str(i+1))
        subjects_result.append(df.mean())

    # ===== GLOBAL METRICS =====
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_features = np.concatenate(all_features, axis=0)

    # 🔥 CONFUSION MATRIX
    cm = confusion_matrix(all_true, all_pred)

    print("\n==============================")
    print("FINAL CONFUSION MATRIX (ALL SUBJECTS)")
    print("==============================")
    print(cm)

    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    labels = ['Left', 'Right', 'Foot', 'Tongue']

    plt.figure(figsize=(6,5))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=labels, yticklabels=labels)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (All Subjects - Normalized)")
    plt.show()
    pd.DataFrame(cm).to_excel(dirs + "/confusion_matrix_all_subjects.xlsx", index=False)
    pd.DataFrame(cm_norm).to_excel(dirs + "/confusion_matrix_normalized.xlsx", index=False)

    # 🔥 COVARIANCE MATRIX (FEATURES)
    cov_matrix = np.cov(all_features, rowvar=False)
    np.save(dirs + "/covariance_matrix.npy", cov_matrix)

    print("\nCovariance Matrix Shape:", cov_matrix.shape)

    # ===== FINAL RESULT SAVE =====
    df_result = pd.DataFrame(subjects_result)

    mean = df_result.mean()
    mean.name = 'mean'

    std = df_result.std()
    std.name = 'std'

    df_result = pd.concat([df_result, pd.DataFrame(mean).T, pd.DataFrame(std).T])
    df_result.to_excel(result_write, index=False, sheet_name='mean')

    result_write.close()
    process_write.close()
    pred_write.close()

    print(f'\nAverage accuracy: {df_result["accuray"]["mean"]:.2f}%')

    return df_result

# ═══════════════════════════════════════════════════════════════════
# Parameters & Entry
# ═══════════════════════════════════════════════════════════════════
class Parameters():
    def __init__(self):
        self.subject_number = 9
        self.learning_rate  = 0.001
        self.batch_size     = 72
        self.epochs         = 1000
        self.number_aug     = 2
        self.number_seg     = 10

if __name__ == "__main__":
    DATA_DIR      = r'/content/mymat_raw/'
    EVALUATE_MODE = 'LOSO-No'
    TYPE          = 'A'
    parameters    = Parameters()

    POOLING_SIZE  = 50
    F1            = 16

    # ★ SOTA FIX: Shallow Transformer parameters
    HEADS         = 2
    DEPTH         = 2

    number_class, number_channel = numberClassChannel(TYPE)
    RESULT_NAME = f"TCANet_ShallowTransformer_Fast_{TYPE}"

    sModel = EEGTransformer(database_type=TYPE, parameters=parameters).cuda()
    summary(sModel, (1, number_channel, 1000))

    result = main(RESULT_NAME, parameters, evaluate_mode=EVALUATE_MODE, dataset_type=TYPE)

#For 2b

import os
import numpy as np
import pandas as pd
import random, datetime, warnings
import scipy.io as sio
warnings.filterwarnings("ignore")

import torch
from torch import nn, Tensor
from torch.autograd import Variable
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange

from pandas import ExcelWriter
from torchsummary import summary
from torch.backends import cudnn

from utils import numberClassChannel, load_data_evaluate, calMetrics, calculatePerClass

gpu_number = 0
os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
cudnn.benchmark     = False
cudnn.deterministic = True

# ═══════════════════════════════════════════════════════════════════
# Attention Blocks (CBAM)
# ═══════════════════════════════════════════════════════════════════
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.se(x).view(x.shape[0], x.shape[1], 1, 1)

class CBAMSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg   = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x,  dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.se      = SEBlock(channels, reduction)
        self.spatial = CBAMSpatialAttention()
    def forward(self, x):
        x = self.se(x)
        x = self.spatial(x)
        return x

# ═══════════════════════════════════════════════════════════════════
# Spatial Extractor (MSCNet)
# ═══════════════════════════════════════════════════════════════════
class MSCNet(nn.Module):
    def __init__(self, f1=16, D=2, pooling_size=50, dropout_rate=0.5, number_channel=3):
        super().__init__()
        self.D = D
        def make_branch(kernel):
            conv_spatial = nn.Conv2d(f1, f1 * D, (number_channel, 1), (1, 1), groups=f1)
            conv_spatial.register_forward_pre_hook(self._max_norm_spatial)
            return nn.Sequential(
                nn.Conv2d(1, f1, (1, kernel), (1, 1), padding='same'),
                conv_spatial,
                nn.BatchNorm2d(f1 * D),
                nn.ELU(),
                CBAMBlock(f1 * D),
                nn.AvgPool2d((1, pooling_size)),
                nn.Dropout(dropout_rate),
            )
        self.cnn1 = make_branch(125)
        self.cnn2 = make_branch(62)
        self.cnn3 = make_branch(31)
        self.proj = Rearrange('b e (h) (w) -> b (h w) e')

    def _max_norm_spatial(self, module, input):
        with torch.no_grad():
            norm    = module.weight.data.norm(2, dim=(1,2,3), keepdim=True)
            desired = torch.clamp(norm, max=1.0)
            module.weight.data *= desired / (norm + 1e-8)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.cat([self.cnn1(x), self.cnn2(x), self.cnn3(x)], dim=1))

# ═══════════════════════════════════════════════════════════════════
# Temporal Extractor (TCN)
# ═══════════════════════════════════════════════════════════════════
class CausalConv1d(nn.Conv1d):
    def forward(self, x):
        return super().forward(F.pad(x, ((self.kernel_size[0] - 1) * self.dilation[0], 0)))

class _TCNBlock(nn.Module):
    def __init__(self, input_dimension, depth, kernel_size,
                 filters, drop_prob, activation=nn.ELU):
        super().__init__()
        self.activation = activation()
        self.downsample = (
            nn.Conv1d(input_dimension, filters, 1, bias=False)
            if input_dimension != filters else None)
        self.layers = nn.ModuleList([
            nn.Sequential(
                CausalConv1d(input_dimension if i == 0 else filters,
                             filters, kernel_size, dilation=2**i, bias=False),
                nn.BatchNorm1d(filters), activation(), nn.Dropout(drop_prob),
                CausalConv1d(filters, filters, kernel_size,
                             dilation=2**i, bias=False),
                nn.BatchNorm1d(filters), activation(), nn.Dropout(drop_prob),
            ) for i in range(depth)
        ])

    def forward(self, x):
        x   = x.permute(0, 2, 1)
        res = x if self.downsample is None else self.downsample(x)
        for layer in self.layers:
            out = self.activation(layer(x) + res)
            res = out;  x = out
        return out.permute(0, 2, 1)

# ═══════════════════════════════════════════════════════════════════
# Transformer Blocks
# ═══════════════════════════════════════════════════════════════════
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size  = emb_size
        self.num_heads = num_heads
        self.keys      = nn.Linear(emb_size, emb_size)
        self.queries   = nn.Linear(emb_size, emb_size)
        self.values    = nn.Linear(emb_size, emb_size)
        self.att_drop  = nn.Dropout(dropout)
        self.proj      = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        q = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(self.keys(x),    "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(self.values(x),  "b n (h d) -> b h n d", h=self.num_heads)
        att = F.softmax(
            torch.einsum('bhqd,bhkd->bhqk', q, k) / self.emb_size**0.5, dim=-1)
        out = rearrange(
            torch.einsum('bhal,bhlv->bhav', self.att_drop(att), v),
            "b h n d -> b n (h d)")
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, emb_size, expansion=4, drop_p=0.5):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size * expansion),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(emb_size * expansion, emb_size),
            nn.Dropout(drop_p),
        )
    def forward(self, x):
        return self.ff(x)

class ResidualAdd(nn.Module):
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn   = fn
        self.drop = nn.Dropout(drop_p)
        self.norm = nn.LayerNorm(emb_size)
    def forward(self, x, **kwargs):
        return self.norm(self.drop(self.fn(x, **kwargs)) + x)

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=4, drop_p=0.5):
        super().__init__(
            ResidualAdd(MultiHeadAttention(emb_size, num_heads, drop_p),
                        emb_size, drop_p),
            ResidualAdd(FeedForward(emb_size, drop_p=drop_p),
                        emb_size, drop_p),
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, heads, depth, emb_size):
        super().__init__(
            *[TransformerEncoderBlock(emb_size, heads) for _ in range(depth)])

# ═══════════════════════════════════════════════════════════════════
# TCANet (Shallow Transformer Variant)
# ★ SOTA FIXES: Positional Encoding, Skip-Concat to Classifier,
#               Learnable Input Normalization
# ═══════════════════════════════════════════════════════════════════
class TCANet_ShallowTransformer(nn.Module):
    def __init__(self, n_chans=3, out_features=2, drop_prob=0.5, depth=2,
                 kernel_size=4, filters=16, max_norm_const=0.25):
        super().__init__()
        self.max_norm_const = max_norm_const
        D = 2

        # Learnable Input Normalization for subject-specific scaling
        self.input_norm = nn.BatchNorm2d(1)

        # 1. Spatial Extractor
        self.mseegnet  = MSCNet(
            number_channel=n_chans,
            dropout_rate=drop_prob,
            pooling_size=POOLING_SIZE,
            f1=F1,
            D=D)

        # 2. Temporal Extractor (Shallow)
        self.tcn_block = _TCNBlock(
            3 * (F1 * D), depth, kernel_size, filters, drop_prob=0.25)

        # 3. Shallow Transformer
        self.sa = TransformerEncoder(heads=HEADS, depth=DEPTH, emb_size=filters)

        # Positional Embedding
        seq_len = 1000 // POOLING_SIZE
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, filters))

        self.drop    = nn.Dropout(0.25)
        self.flatten = nn.Flatten()

        # Multiply filters * 2 because we concatenate TCN and Transformer outputs
        self.classifier = nn.Linear((filters * 2) * seq_len, out_features)
        self.classifier.register_forward_pre_hook(self._max_norm)

    def _max_norm(self, module, input):
        with torch.no_grad():
            norm    = self.classifier.weight.data.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, max=self.max_norm_const)
            self.classifier.weight.data *= desired / (norm + 1e-8)

    def forward(self, x):
        x        = self.input_norm(x)           # Learnable subject-specific scaling
        x        = self.mseegnet(x)
        x_tcn    = self.tcn_block(x)
        x_pos    = x_tcn + self.pos_embedding   # Add positional embedding before Transformer
        sa       = self.sa(x_pos)
        # Combine local TCN features with global Transformer features
        features = self.flatten(self.drop(torch.cat([x_tcn, sa], dim=-1)))
        return self.classifier(features), features


class EEGTransformer(nn.Module):
    def __init__(self, database_type, parameters):
        super(EEGTransformer, self).__init__()
        self.database_type = database_type

        # 1. Dynamically fetch class and channel configurations
        self.num_classes, self.num_channels = numberClassChannel(database_type)

        # 2. Link directly to your multi-branch MSCNet (Fixes the 16 vs 96 channel mismatch)
        # This automatically uses your cnn1, cnn2, cnn3 branches and outputs 96 channels total!
        self.spatial_extractor = MSCNet(
            number_channel=self.num_channels,
            dropout_rate=parameters.dropout_rate,
            pooling_size=POOLING_SIZE,
            f1=F1,
            D=2
        )

        # 3. Temporal Block (TCN) - Expects exactly 96 input channels from MSCNet
        self.temporal_block = nn.Sequential(
            nn.Conv2d(96, 16, kernel_size=(1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU()
        )

        # 4. Shallow Transformer Encoder Module
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=HEADS,
            dim_feedforward=32,
            dropout=parameters.dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=DEPTH)

        # 5. Feature Aggregation & Dense Classifier Head
        self.flatten = nn.Flatten()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 20))
        self.classifier = nn.Linear(16 * 20, self.num_classes)

    def forward(self, x):
        # x shape: [Batch, 1, Channels, Time_Samples]

        # Run through the multi-scale branches + CBAM blocks
        x = self.spatial_extractor(x)  # Outputs shape: [Batch, 96, 1, 20]

        # Run through temporal feature extraction
        x = self.temporal_block(x)     # Outputs shape: [Batch, 16, 1, 20]

        # Format dimensions for Transformer Sequence Length [Batch, Seq, Features]
        b, c, h, w = x.size()
        x_trans = x.permute(0, 3, 1, 2).reshape(b, w, c * h)
        x_trans = self.transformer_encoder(x_trans)
        x_trans = x_trans.reshape(b, c, h, w)

        # Flatten and classify
        x_fused = self.global_pool(x_trans)
        x_flat = self.flatten(x_fused)

        out = self.classifier(x_flat)
        return out

# ═══════════════════════════════════════════════════════════════════
# Augmentations
# ═══════════════════════════════════════════════════════════════════
def mixup_data(x, y, alpha=0.2):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0)).cuda()
    return lam * x + (1 - lam) * x[index], y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def freq_shift_augment(data, max_shift=2):
    shift = random.randint(-max_shift, max_shift)
    if shift == 0:
        return data
    return torch.roll(data, shifts=shift, dims=-1)

# ═══════════════════════════════════════════════════════════════════
# Experiment class
# ═══════════════════════════════════════════════════════════════════
class ExP():
    def __init__(self, nsub, data_dir, result_name, parameters,
                 evaluate_mode='LOSO-no', dataset_type='B', n_fold=0):
        super().__init__()
        self.n_fold              = n_fold
        self.dataset_type        = dataset_type
        self.batch_size          = parameters.batch_size
        self.lr                  = parameters.learning_rate
        self.b1                  = 0.5
        self.b2                  = 0.999
        self.n_epochs            = parameters.epochs
        self.nSub                = nsub
        self.nFold               = n_fold
        self.number_augmentation = parameters.number_aug
        self.number_seg          = parameters.number_seg
        self.root                = data_dir
        self.result_name         = result_name
        self.evaluate_mode       = evaluate_mode
        self.Tensor              = torch.cuda.FloatTensor
        self.LongTensor          = torch.cuda.LongTensor

        self.criterion_cls = torch.nn.CrossEntropyLoss(
            label_smoothing=0.05).cuda()

        self.number_class, self.number_channel = numberClassChannel(dataset_type)
        self.model = EEGTransformer(
            database_type=self.dataset_type,
            parameters=parameters).cuda()
        self.model_filename = (self.result_name +
            f'/model_nsub_{self.nSub}_nfold_{n_fold+1}.pth')

    def interaug(self, timg, label):
        aug_data, aug_label = [], []
        n_per = self.number_augmentation * int(self.batch_size / self.number_class)
        seg   = 1000 // self.number_seg
        for cls in range(self.number_class):
            idx  = np.where(label == cls + 1)
            tmp  = timg[idx]
            aug  = np.zeros((n_per, 1, self.number_channel, 1000))
            for ri in range(n_per):
                ri_idx = np.random.randint(0, tmp.shape[0], self.number_seg)
                for rj in range(self.number_seg):
                    aug[ri, :, :, rj*seg:(rj+1)*seg] = \
                        tmp[ri_idx[rj], :, :, rj*seg:(rj+1)*seg]
            aug_data.append(aug)
            aug_label.append(label[idx][:n_per])
        aug_data  = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        shuf      = np.random.permutation(len(aug_data))
        return (torch.from_numpy(aug_data[shuf]).cuda().float(),
                torch.from_numpy(aug_label[shuf] - 1).cuda().long())

    def get_source_data(self):
        # Load via utils for consistency (uses load_data_evaluate)
        train_path = os.path.join(self.root, f'B0{self.nSub}T.mat')
        train_mat  = sio.loadmat(train_path)
        self.train_data  = train_mat['data']
        self.train_label = train_mat['label']

        test_path = os.path.join(self.root, f'B0{self.nSub}E.mat')
        test_mat  = sio.loadmat(test_path)
        self.test_data  = test_mat['data']
        self.test_label = test_mat['label']

        self.train_data  = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.transpose(self.train_label)
        self.allData     = self.train_data
        self.allLabel    = self.train_label[0]

        shuf = np.random.permutation(len(self.allData))
        self.allData  = self.allData[shuf]
        self.allLabel = self.allLabel[shuf]

        self.test_data  = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)
        self.testData   = self.test_data
        self.testLabel  = self.test_label[0]

        # Global z-score using train statistics
        mu  = np.mean(self.allData)
        std = np.std(self.allData)
        self.allData  = (self.allData  - mu) / std
        self.testData = (self.testData - mu) / std

        print('-'*20, "train:", self.train_data.shape,
              "test:", self.test_data.shape,
              "sub:", self.nSub, "fold:", self.nFold+1)
        return self.allData, self.allLabel, self.testData, self.testLabel

    def test_model(self, model, dataloader):
        model.eval()
        out_list, lbl_list = [], []
        with torch.no_grad():
            for img, label in dataloader:
                img   = img.type(self.Tensor).cuda()
                label = label.type(self.LongTensor).cuda()
                _, cls = model(img)
                out_list.append(cls); lbl_list.append(label)
                del img, cls; torch.cuda.empty_cache()
        cls_all  = torch.cat(out_list)
        lbl_all  = torch.cat(lbl_list)
        val_loss = self.criterion_cls(cls_all, lbl_all)
        val_pred = torch.max(cls_all, 1)[1]
        val_acc  = float((val_pred == lbl_all).cpu().numpy().astype(int).sum()) / \
                   float(lbl_all.size(0))
        return val_acc, val_loss, val_pred

    def train(self):
        timg, label, test_data, test_label = self.get_source_data()

        # Per-class 5-fold split
        data_pc, lbl_pc = [], []
        for cls in range(self.number_class):
            idx = np.where(label == cls + 1)
            data_pc.append(timg[idx]); lbl_pc.append(label[idx])

        tr_d, tr_l, val_d, val_l = [], [], [], []
        seed = 1234 + self.nSub
        for cls in range(self.number_class):
            n = len(data_pc[cls]); n_val = n // 5
            np.random.seed(seed + cls)
            idx_s = np.random.permutation(n)
            idx_v = list(range(self.n_fold * n_val,
                               n if self.n_fold == 4
                               else (self.n_fold+1)*n_val))
            idx_t = [i for i in range(n) if i not in idx_v]
            tr_d.append(data_pc[cls][idx_s[idx_t]])
            tr_l.append(lbl_pc[cls][idx_s[idx_t]])
            val_d.append(data_pc[cls][idx_s[idx_v]])
            val_l.append(lbl_pc[cls][idx_s[idx_v]])

        img       = np.concatenate(tr_d)
        label     = np.concatenate(tr_l)
        val_data  = np.concatenate(val_d)
        val_label = np.concatenate(val_l)
        print('-'*20, "train:", img.shape, "val:", val_data.shape)

        self.dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(img), torch.from_numpy(label - 1)),
            batch_size=self.batch_size, shuffle=True)

        self.val_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(val_data), torch.from_numpy(val_label - 1)),
            batch_size=self.batch_size, shuffle=True)

        test_t     = torch.from_numpy(test_data)
        test_lbl_t = torch.from_numpy(test_label - 1)
        self.test_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_t, test_lbl_t),
            batch_size=self.batch_size, shuffle=False)

        # AdamW + Warmup + Cosine LR
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr,
            betas=(self.b1, self.b2), weight_decay=1e-4)

        warmup = 20
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.05, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.n_epochs - warmup, eta_min=1e-5)
            ],
            milestones=[warmup])

        test_var = Variable(test_t.type(self.Tensor))
        tlbl_var = Variable(test_lbl_t.type(self.LongTensor))

        best_epoch = 0
        min_loss   = 100.0
        result_process = []

        for e in range(self.n_epochs):
            self.model.train()
            out_list, lbl_list = [], []

            for img_b, lbl_b in self.dataloader:
                img_v = Variable(img_b.type(self.Tensor))
                lbl_v = Variable(lbl_b.type(self.LongTensor))

                # Segment-mixing augmentation
                aug_data, aug_label = self.interaug(img, label)

                # Gaussian noise
                aug_data = aug_data + torch.randn_like(aug_data) * 0.05

                # Amplitude scaling
                aug_data = aug_data * torch.FloatTensor(
                    aug_data.shape[0], 1, 1, 1).uniform_(0.8, 1.2).cuda()

                # Frequency shift
                aug_data = freq_shift_augment(aug_data, max_shift=2)

                img_v = torch.cat((img_v, aug_data))
                lbl_v = torch.cat((lbl_v, aug_label))

                # MixUp only after warmup — protects early training
                if e > warmup:
                    img_mix, lbl_a, lbl_b_mix, lam = mixup_data(
                        img_v, lbl_v, alpha=0.2)
                    _, outputs = self.model(img_mix)
                    loss = mixup_criterion(
                        self.criterion_cls, outputs, lbl_a, lbl_b_mix, lam)
                else:
                    _, outputs = self.model(img_v)
                    loss = self.criterion_cls(outputs, lbl_v)

                out_list.append(outputs.detach())
                lbl_list.append(lbl_v)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            torch.cuda.empty_cache()

            # Validation — model selection on val_loss (NO test set leakage)
            self.model.eval()
            val_acc, val_loss, _ = self.test_model(self.model, self.val_dataloader)

            all_out = torch.cat(out_list)
            all_lbl = torch.cat(lbl_list)
            tr_pred = torch.max(all_out, 1)[1]
            tr_acc  = float((tr_pred == all_lbl).cpu().numpy().astype(int).sum()) / \
                      float(all_lbl.size(0))

            ep = {'epoch':      e,
                  'val_acc':    val_acc,
                  'val_loss':   val_loss.detach().cpu().numpy(),
                  'train_acc':  tr_acc,
                  'train_loss': loss.detach().cpu().numpy()}

            if min_loss > val_loss:
                min_loss   = val_loss
                best_epoch = e
                torch.save(self.model, self.model_filename)
                test_acc, _, y_pred = self.test_model(
                    self.model, self.test_dataloader)
                lr_now = self.scheduler.get_last_lr()[0]
                print(f"sub{self.nSub} ep{e:4d} | "
                      f"tr:{tr_acc:.4f} val_acc:{val_acc:.4f} "
                      f"val_loss:{val_loss:.6f} "
                      f"test:{test_acc:.4f} lr:{lr_now:.6f}")

            result_process.append(ep)

        # Final evaluation with best saved model
        self.model = torch.load(self.model_filename, weights_only=False).cuda()
        self.model.eval()
        out_list = []
        with torch.no_grad():
            for img_b, _ in self.test_dataloader:
                _, outputs = self.model(
                    Variable(img_b.type(self.Tensor)).cuda())
                out_list.append(outputs)
        outputs  = torch.cat(out_list)
        y_pred   = torch.max(outputs, 1)[1]
        test_acc = float((y_pred == tlbl_var).cpu().numpy().astype(int).sum()) / \
                   float(tlbl_var.size(0))
        print(f"Best epoch: {best_epoch}  Final test acc: {test_acc:.4f}")
        return test_acc, tlbl_var, y_pred, pd.DataFrame(result_process), \
               best_epoch, outputs

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main(dirs, parameters, evaluate_mode='subject-dependent', dataset_type='B'):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    result_write  = ExcelWriter(dirs + "/result_metric.xlsx")
    process_write = ExcelWriter(dirs + "/process_train.xlsx")
    pred_write    = ExcelWriter(dirs + "/pred_true.xlsx")
    softmax_write = ExcelWriter(dirs + "/pred_softmax.xlsx")
    subjects_result, best_epochs = [], []

    for i in range(parameters.subject_number):
        starttime = datetime.datetime.now()
        seed_n = np.random.randint(2024)
        random.seed(seed_n); np.random.seed(seed_n)
        torch.manual_seed(seed_n); torch.cuda.manual_seed_all(seed_n)
        print(f'\nSubject {i+1}  seed={seed_n}')

        exp = ExP(i + 1, DATA_DIR, dirs, parameters,
                  evaluate_mode=evaluate_mode,
                  dataset_type=dataset_type,
                  n_fold=2)

        testAcc, Y_true, Y_pred, df_proc, best_epoch, pred_out = exp.train()

        pd.DataFrame(
            torch.softmax(pred_out, dim=1).cpu().numpy()
        ).to_excel(softmax_write, sheet_name=str(i+1))

        true_cpu = Y_true.cpu().numpy().astype(int)
        pred_cpu = Y_pred.cpu().numpy().astype(int)
        pd.DataFrame({'pred': pred_cpu, 'true': true_cpu}).to_excel(
            pred_write, sheet_name=str(i+1))
        df_proc.to_excel(process_write, sheet_name=str(i+1))
        best_epochs.append(best_epoch)

        acc, prec, rec, f1, kappa = calMetrics(true_cpu, pred_cpu)
        result = {'accuray':   acc*100,
                  'precision': prec*100,
                  'recall':    rec*100,
                  'f1':        f1*100,
                  'kappa':     kappa*100}
        print(f'Subject {i+1}: acc={acc*100:.2f}%  kappa={kappa:.4f}')
        print(f'Duration: {datetime.datetime.now()-starttime}')

        if i == 0: yt = Y_true;  yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))

        pd.DataFrame([result]).to_excel(
            result_write, index=False, sheet_name=str(i+1))
        subjects_result.append(result)

    df_result = pd.DataFrame(subjects_result)
    process_write.close(); pred_write.close(); softmax_write.close()

    mean = df_result.mean(); mean.name = 'mean'
    std  = df_result.std();  std.name  = 'std'
    df_result = pd.concat([df_result,
                            pd.DataFrame(mean).T, pd.DataFrame(std).T])
    df_result.to_excel(result_write, index=False, sheet_name='mean')
    result_write.close()

    print(f'\n{"="*40}')
    print(f'Average accuracy: {df_result["accuray"]["mean"]:.2f}%')
    print(f'Average kappa:    {df_result["kappa"]["mean"]:.4f}')
    print(f'Best epochs: {best_epochs}')
    return df_result

    cm_sub = confusion_matrix(true_cpu, pred_cpu)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_sub, annot=True, fmt='d', cmap='YlGnBu')
    plt.title(f'Confusion Matrix - Subject {i+1}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f"{dirs}/CM_Subject_{i+1}.png")
    plt.close() # Close to save memory

# ═══════════════════════════════════════════════════════════════════
# Parameters & Entry
# ═══════════════════════════════════════════════════════════════════
class Parameters():
    def __init__(self, dropout_rate):
        self.subject_number = 9
        self.learning_rate  = 0.001
        self.batch_size     = 72
        self.epochs         = 1000
        self.number_aug     = 2
        self.number_seg     = 10
        self.dropout_rate   = dropout_rate

if __name__ == "__main__":
    DATA_DIR      = r'/content/mymat_aligned/'
    EVALUATE_MODE = 'LOSO-No'
    TYPE          = 'B'

    CNN_DROPOUT_RATE = 0.5
    parameters = Parameters(CNN_DROPOUT_RATE)

    POOLING_SIZE  = 50
    F1            = 16
    HEADS         = 2
    # ★ SOTA FIX: Reduced Transformer depth to 2 to prevent rapid overfitting
    DEPTH         = 2

    number_class, number_channel = numberClassChannel(TYPE)
    RESULT_NAME = f"TCANet_ShallowTransformer_Fixed_{TYPE}"

    sModel = EEGTransformer(database_type=TYPE, parameters=parameters).cuda()
    summary(sModel, (1, number_channel, 1000))

    result = main(RESULT_NAME, parameters,
                  evaluate_mode=EVALUATE_MODE, dataset_type=TYPE)
