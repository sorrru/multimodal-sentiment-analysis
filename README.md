# 实验一：英文新闻文本十分类

课程实验代码：TF-IDF + 逻辑回归（LR）/ 线性 SVM，并比较 C 参数与 RBF 核。仓库名沿用 `multimodal-sentiment-analysis`，本项目实际任务是英文新闻分类，不包含多模态情感分析。

## 环境与运行

```bash
conda create -n project1 python=3.10
conda activate project1
pip install -r requirements.txt
```

将课程提供的 `train_data.csv`（text、target 两列）与 `test_data_unlabeled.csv`（text 一列）放在仓库根目录，再从根目录运行：

```bash
mkdir -p results
python run_experiment.py
```

数据集不随仓库发布。训练数据 7368 条，无标签测试数据 2457 条；需要自行取得课程原始 CSV，不应以其他数据代替来核验本报告成绩。

已验证环境：Python 3.10.21、scikit-learn 1.5.2、NumPy 1.23.5、pandas 2.2.3、SciPy 1.10.1、Matplotlib 3.7.0，macOS ARM64。`gensim` 与 `nltk` 为可选扩展依赖，主脚本不使用。

## 实验流程

1. 用固定正则处理邮件头，并提取、保留 Subject；不删除样本或改变标签。
2. 使用 seed=42 的 80/20 分层划分，得到 5894 条训练、1474 条验证样本。
3. 仅在训练子集拟合 TF-IDF（30000 特征、英文停用词、默认 unigram / 小写 / L2 归一化）；验证集只 transform。
4. LR 与线性 SVC 分别搜索 C=0.1、1、10、100；RBF 固定 C=1，比较 gamma=scale、auto、0.1、1。
5. 按验证准确率选优，准确率持平时选择 LR；macro-F1 和逐类指标作为辅助分析。
6. 锁定配置后，在全部训练数据上重新拟合向量化器和分类器，预测测试数据并保存 `predictions.csv`。

## 验证结果

| 模型 | C | 验证准确率 | 宏平均 F1 |
|---|---|---|---|
| LogisticRegression | 100 | 0.939620 | 0.939899 |
| SVC(linear) | 10 | 0.939620 | 0.939925 |

RBF 在本次配置中最佳准确率为 0.936906（gamma=scale 与 1 并列）。由于两类核的搜索范围不对等，不能据此推断普遍优势。测试集无标签，上述成绩均不代表测试集准确率。

## 文件与产物

- `run_experiment.py`：完整训练、调参、评估和预测入口。
- `requirements.txt`：依赖版本。
- `results/`：调参表、RBF 对照表、最优 LR 分类报告及混淆矩阵。
- `audit/check_evidence.py`：补充去头对照、LR/SVM 逐类评估、清洗残留及重复样本检查；可执行 `python audit/check_evidence.py`。
- `predictions.csv`：运行后生成，2457 行、单列、无表头、整数 0-9；提交时复制并命名为 `prediction.csv`。该文件不上传仓库。

## 已知局限与复现说明

当前代码保留与已复现报告一致的处理方式。头部正则遇到 `Article-I.D.` 或折行头字段可能提前停止，至少 503 条训练文本仍有部分元数据残留，不能将其称为完整去头。当前处理相对原始全文的 LR C=1 准确率增益约为 0.61 个百分点，不能据此证明数据泄露。

清洗后有两组相同文本跨训练/验证集；只采用单次 hold-out，未量化分组或不同划分的波动。RBF 的 C、TF-IDF 参数未系统搜索。修复清洗规则后应重新生成全部结果，不能直接沿用当前分数。

主脚本首次写结果发生在其内部创建目录之前，因此运行前需执行上面的 `mkdir -p results`。复跑已确认全部调参准确率、分类报告、混淆矩阵和最终预测可重现；耗时随硬件及线程设置变化。

## 作业提交

GitHub 只保存代码、依赖说明和小型评估结果。另行提交的 ZIP 应在根层只包含 `实验报告.pdf` 与 `prediction.csv`；报告不超过两页，并在首页列出本仓库地址。
