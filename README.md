# Fake News Classifier

## 📰 Project Overview

This project is a **Fake News Classification Model** using the **LIAR2 dataset**. It leverages the power of the `roberta-base` transformer model to classify political statements into one of six categories:

* **pants-fire**
* **false**
* **barely-true**
* **half-true**
* **mostly-true**
* **true**

The model is fine-tuned using Hugging Face’s `Trainer` API and achieves promising results for real-world fake news detection tasks.

---

## 📂 Dataset

* **Source:** Hugging Face - `chengxuphd/liar2`
* **Dataset Size:**

  * Train: \~10,269 samples
  * Validation: \~1,284 samples
  * Test: \~1,282 samples
* **Features:**

  * `statement`: The text of the political statement.
  * `label`: One of the six truthfulness categories.

---

## 🧐 Model Details

* **Pre-trained Model:** `roberta-base`
* **Fine-tuned on:** LIAR2 Dataset
* **Training Setup:**

  * Batch size: 16
  * Epochs: 3
  * Learning rate: 2e-5
  * Weight decay: 0.01
  * Evaluation Strategy: Every epoch

---

## 📊 Results

| Metric           | Score   |
| ---------------- | ------- |
| Accuracy         | 67.15 % |
| F1 Score (Macro) | 65.87 % |
| Precision        | 67.02 % |
| Recall           | 66.10 % |

---

## ✅ Model Evaluation Output

Example evaluation code:

```python
eval_results = trainer.evaluate(test_dataset)
print("Evaluation Results:", eval_results)
```

Output:

```text
Evaluation Results:
{'eval_loss': 1.1214,
 'eval_accuracy': 0.6715,
 'eval_f1_macro': 0.6587,
 'eval_precision': 0.6702,
 'eval_recall': 0.6610}
```

---

## 🔮 Inference Example

```python
classifier = pipeline("text-classification", model=trainer.model, tokenizer=tokenizer)

sample_text = "The president announced a new healthcare reform that will cut costs by 30%."
prediction = classifier(sample_text)
print(f"Prediction: {prediction}")
```

Output:

```text
Prediction: [{'label': 'mostly-true', 'score': 0.756}]
```

---

## 🚀 How to Run

### Google Colab Setup:

1. Install dependencies:

```python
!pip install --upgrade transformers datasets scikit-learn accelerate
```

2. Follow step-by-step training code provided in the notebook.

---

## 📚 References

* Dataset: [chengxuphd/liar2](https://huggingface.co/datasets/chengxuphd/liar2)
* Model: [roberta-base](https://huggingface.co/roberta-base)
* Hugging Face [Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer)
