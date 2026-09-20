# -*- coding: utf-8 -*-
"""实验一：文本分类（新闻 10 分类）。

流程：
  1. 加载数据
  2. 数据清洗：剥掉元数据邮件头（From/Organization/Lines/Reply-To/主机名等身份泄漏噪声），
     但保留 Subject 标题（真实主题信号），最终用 "Subject + 正文" 作为文本
  3. 划分训练集 / 验证集（80/20 分层，固定随机种子 42）
  4. 特征工程：TF-IDF（只在训练集上 fit，验证集/测试集只 transform）
  5. 模型一：逻辑回归，C 网格搜索
  6. 模型二：SVM（线性核），C 网格搜索
  7. 评估对比：准确率 / 分类报告 / 混淆矩阵
  8. 锁定最优模型，用全部训练集重训，预测测试集并保存 predictions.csv

可复现性：所有随机种子固定为 42；只用相对路径（脚本需在数据目录内运行）。
"""

import os
import re
import time

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # 无图形界面也能保存图片
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

RANDOM_STATE = 42
C_GRID = [0.1, 1, 10, 100]  # 正则化参数搜索范围（按数量级）
TFIDF_KW = dict(max_features=30000, stop_words="english")

# 匹配文本开头连续出现的 "Key: value" 邮件头行（整块头部）
HEADER_RE = re.compile(r"(?:(?:[A-Za-z][\w-]*):[^\n]*\n)+")


def load_data():
    train_df = pd.read_csv("train_data.csv")
    test_df = pd.read_csv("test_data_unlabeled.csv")
    X_train = train_df["text"].astype(str).tolist()
    y_train = train_df["target"].values
    X_test = test_df["text"].astype(str).tolist()
    return X_train, y_train, X_test


def clean_text(text):
    """清洗新闻组文本：剥掉元数据邮件头，但保留 Subject，返回 "Subject + 正文"。

    依据归因实验：Subject 是真实主题信号（保留能提升验证集准确率约 2 个点），
    而 From/Organization/Lines/Reply-To/主机名 等是身份泄漏噪声，应去掉。
    """
    m = HEADER_RE.match(text)
    header_block = m.group(0) if m else ""

    # 从头部块里提取 Subject，并去掉 "Re:" / "Re[2]:" 前缀
    subj_m = re.search(r"^Subject:\s*(.*)$", header_block, re.M)
    subject = subj_m.group(1).strip() if subj_m else ""
    subject = re.sub(r"^[Rr][Ee]\d*:\s*", "", subject)

    # 正文 = 去掉头部块之后的部分
    body = text[m.end():].lstrip("\n") if m else text

    return f"{subject} {body}".strip() if subject else body


def grid_search_c(model_cls, X_tr, y_tr, X_val, y_val, **fixed_kwargs):
    """在 C_GRID 上逐点训练，返回每组参数在验证集上的指标表。"""
    rows = []
    for C in C_GRID:
        model = model_cls(C=C, random_state=RANDOM_STATE, **fixed_kwargs)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        rows.append(
            {
                "C": C,
                "val_accuracy": accuracy_score(y_val, pred),
                "val_macro_f1": f1_score(y_val, pred, average="macro"),
            }
        )
    return pd.DataFrame(rows)


def main():
    print("=" * 62)
    print("实验一：文本分类（新闻 10 分类）")
    print("=" * 62)

    # ---- 1. 加载 ----
    X_train, y_train, X_test = load_data()
    print(f"[1] 数据加载：训练集 {len(X_train)} 条，测试集 {len(X_test)} 条")

    # ---- 2. 清洗 ----
    X_clean = [clean_text(t) for t in X_train]
    X_test_clean = [clean_text(t) for t in X_test]
    print("[2] 数据清洗：剥元数据头，保留 Subject（只改内容，不删样本）")
    print(f"    清洗前样本数 = {len(X_train)}，清洗后样本数 = {len(X_clean)}（无样本被删/标签改变）")

    # ---- 3. 划分（按索引，raw/clean 共用同一划分）----
    idx = np.arange(len(y_train))
    idx_tr, idx_val = train_test_split(
        idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )
    y_tr, y_val = y_train[idx_tr], y_train[idx_val]
    print(f"[3] 划分(80/20 分层)：新训练集 {len(idx_tr)} 条，验证集 {len(idx_val)} 条")
    dist = pd.DataFrame(
        {
            "train%": pd.Series(y_tr).value_counts(normalize=True).sort_index(),
            "val%": pd.Series(y_val).value_counts(normalize=True).sort_index(),
        }
    ).round(4)
    print("    各类别占比（应基本一致）：")
    print(dist.to_string())

    # ---- 4. TF-IDF ----
    print(f"\n[4] 特征工程：TF-IDF {TFIDF_KW}")
    vec = TfidfVectorizer(**TFIDF_KW)
    X_tr = vec.fit_transform([X_clean[i] for i in idx_tr])
    X_val = vec.transform([X_clean[i] for i in idx_val])
    print(f"    TF-IDF: 训练 {X_tr.shape}, 验证 {X_val.shape}")

    # ---- 4b. 清洗价值对比（可选，展示剥邮件头是否有效）----
    print("\n[4b] 清洗价值对比（逻辑回归默认参数，验证集准确率）")
    for name, txts in [("不清洗(含全部头部)", X_train), ("清洗(剥元数据头+保留Subject)", X_clean)]:
        v = TfidfVectorizer(**TFIDF_KW)
        a = v.fit_transform([txts[i] for i in idx_tr])
        b = v.transform([txts[i] for i in idx_val])
        m = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE).fit(a, y_tr)
        print(f"    {name}: {accuracy_score(y_val, m.predict(b)):.6f}")

    # ---- 5. 逻辑回归 ----
    print("\n[5] 逻辑回归：C 网格搜索")
    lr_tab = grid_search_c(
        LogisticRegression, X_tr, y_tr, X_val, y_val, max_iter=2000
    )
    print(lr_tab.to_string(index=False))

    # ---- 6. SVM ----
    print("\n[6] SVM(线性核)：C 网格搜索")
    svm_tab = grid_search_c(SVC, X_tr, y_tr, X_val, y_val, kernel="linear")
    print(svm_tab.to_string(index=False))

    # ---- 6b. 可选扩展：RBF 核 vs 线性核（验证线性核更适合稀疏文本）----
    print("\n[6b] 可选扩展：RBF 核 vs 线性核（C=1.0，gamma 网格搜索）")
    rbf_rows = []
    for gamma in ["scale", "auto", 0.1, 1]:
        t0 = time.time()
        model = SVC(kernel="rbf", C=1.0, gamma=gamma, random_state=RANDOM_STATE)
        model.fit(X_tr, y_tr)
        dt = time.time() - t0
        pred = model.predict(X_val)
        acc = accuracy_score(y_val, pred)
        rbf_rows.append(
            {"gamma": str(gamma), "val_accuracy": acc, "train_time_s": round(dt, 1)}
        )
        print(f"    gamma={str(gamma):<6} acc={acc:.6f}  训练耗时 {dt:.1f}s")
    rbf_tab = pd.DataFrame(rbf_rows)
    rbf_best_acc = rbf_tab["val_accuracy"].max()
    lin_best_C = svm_tab.loc[svm_tab["val_accuracy"].idxmax(), "C"]
    lin_best_acc = svm_tab["val_accuracy"].max()
    print(f"    线性核最优 (C={lin_best_C}): acc={lin_best_acc:.6f}")
    print(f"    RBF 核最优:                 acc={rbf_best_acc:.6f}")
    print(
        "    → 结论："
        + (
            "线性核更优（且更快），符合 TF-IDF 稀疏文本的预期"
            if lin_best_acc >= rbf_best_acc
            else "RBF 核更优"
        )
    )
    rbf_tab.to_csv("results/rbf_vs_linear.csv", index=False)

    # ---- 7. 选优 ----
    lr_best = lr_tab.loc[lr_tab["val_accuracy"].idxmax()]
    svm_best = svm_tab.loc[svm_tab["val_accuracy"].idxmax()]
    print("\n[7] 模型选优")
    print(
        f"    逻辑回归最优: C={lr_best['C']}, "
        f"acc={lr_best['val_accuracy']:.6f}, macroF1={lr_best['val_macro_f1']:.6f}"
    )
    print(
        f"    SVM 最优:     C={svm_best['C']}, "
        f"acc={svm_best['val_accuracy']:.6f}, macroF1={svm_best['val_macro_f1']:.6f}"
    )

    # 整体最优：验证集准确率更高者；持平则选更简单的逻辑回归
    if lr_best["val_accuracy"] >= svm_best["val_accuracy"]:
        best_name, best_cls, best_C, best_kw = (
            "LogisticRegression",
            LogisticRegression,
            lr_best["C"],
            dict(max_iter=2000),
        )
    else:
        best_name, best_cls, best_C, best_kw = (
            "SVC(linear)",
            SVC,
            svm_best["C"],
            dict(kernel="linear"),
        )
    print(f"    → 整体最优: {best_name}, C={best_C}")

    # ---- 7b. 最优模型的分类报告 + 混淆矩阵（在验证集上）----
    best_model = best_cls(C=best_C, random_state=RANDOM_STATE, **best_kw)
    best_model.fit(X_tr, y_tr)
    y_val_pred = best_model.predict(X_val)
    report = classification_report(y_val, y_val_pred, digits=4)
    print("\n[7b] 最优模型在验证集上的分类报告：")
    print(report)

    os.makedirs("results", exist_ok=True)
    with open("results/classification_report.txt", "w") as f:
        f.write(f"best model: {best_name}, C={best_C}\n")
        f.write(report)

    cm = confusion_matrix(y_val, y_val_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({best_name}, C={best_C})")
    for i in range(10):
        for j in range(10):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=7,
            )
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig("results/confusion_matrix_best.png", dpi=150)
    plt.close(fig)
    print("    已保存 results/confusion_matrix_best.png")

    # 保存调参表
    lr_tab = lr_tab.copy()
    svm_tab = svm_tab.copy()
    lr_tab.insert(0, "model", "LogisticRegression")
    svm_tab.insert(0, "model", "SVC(linear)")
    pd.concat([lr_tab, svm_tab]).to_csv("results/tuning_results.csv", index=False)
    print("    已保存 results/tuning_results.csv")

    # ---- 8. 最终预测：用全部训练集重训最优模型 ----
    print("\n[8] 最终预测：在全部训练集上重训最优模型，预测测试集")
    vec_final = TfidfVectorizer(**TFIDF_KW)
    X_full = vec_final.fit_transform(X_clean)
    X_test_final = vec_final.transform(X_test_clean)
    final_model = best_cls(C=best_C, random_state=RANDOM_STATE, **best_kw)
    final_model.fit(X_full, y_train)
    preds = final_model.predict(X_test_final)
    print(f"    测试集预测完成，共 {len(preds)} 条")

    pd.DataFrame(preds.astype(int)).to_csv("predictions.csv", index=False, header=False)
    print("    已保存 predictions.csv（无表头、单列、顺序对应测试集）")

    print("\n" + "=" * 62)
    print("实验完成 ✅")
    print("=" * 62)


if __name__ == "__main__":
    main()
