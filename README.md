# Neural Machine Translation with Attention (English → French)

This project implements an end-to-end **Neural Machine Translation (NMT)** pipeline that translates English sentences into French using **recurrent neural networks (RNNs)**. The focus of the project is to explore and compare different sequence-to-sequence architectures and demonstrate how architectural choices impact translation quality.

The final model incorporates an **attention mechanism**, addressing common limitations of vanilla encoder–decoder models such as repetition and poor word alignment.

---

## 📌 Project Objectives

- Build a complete English → French translation pipeline
- Explore multiple RNN-based sequence modeling architectures
- Understand the limitations of fixed-context encoder–decoder models
- Improve translation quality using an attention mechanism
- Demonstrate preprocessing, training, and inference in a real NMT workflow

---

## 🗂 Dataset

The project uses a provided **small English–French parallel corpus** with a limited vocabulary, enabling efficient training and experimentation without requiring large computational resources.

---

## 🔧 Preprocessing

- **Word-level tokenization** using Keras `Tokenizer`
- Integer encoding of tokens
- **Post-padding** to ensure uniform sequence lengths
- Labels reshaped to support sparse categorical cross-entropy
- Tokenizers retained for decoding predictions back to text

---

## 🧠 Model Architectures

Five progressively more expressive models were implemented:

1. **Simple RNN Model**  
   A baseline GRU-based model without embeddings.

2. **Embedding RNN Model**  
   Introduces trainable word embeddings for improved semantic representation.

3. **Bidirectional RNN Model**  
   Uses a bidirectional GRU to incorporate both past and future context.

4. **Encoder–Decoder Model**  
   Compresses the input sequence into a fixed-length context vector and decodes it into a translated sentence.

5. **Final Model: Encoder–Decoder with Attention**  
   Extends the encoder–decoder architecture with an attention mechanism, allowing the decoder to dynamically focus on relevant encoder hidden states at each time step. This significantly improves alignment, reduces repetition, and achieves substantially higher accuracy.

All models are trained using the **Adam optimizer** and **sparse categorical cross-entropy**, which is appropriate for multi-class sequence prediction with integer labels.

---

## 📈 Results

- The attention-based final model achieves **~95% training accuracy**
- Produces coherent translations with correct word order and alignment
- Successfully eliminates common failure modes such as repeated words
- Demonstrates clear performance improvements over baseline architectures

## 🧪 Evaluation Notes

For consistency with the original project specification, models were trained on the full dataset without an explicit train/test split. While this leads to overstated accuracy values, the primary goal of the project is **architectural exploration and understanding**, not generalization benchmarking.

A train/test split and additional evaluation metrics could be introduced in future work.

---

## 🚀 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Jupyter Notebook

---

## 📚 What This Project Demonstrates

- Understanding of sequence-to-sequence learning
- Practical implementation of attention mechanisms
- Ability to diagnose and fix architectural limitations
- Clear progression from simple baselines to advanced models
- End-to-end ML pipeline design and execution

---

## 📝 Future Improvements

- Add a formal train/test split for generalization evaluation
- Experiment with different attention mechanisms
- Extend the model to larger vocabularies
- Compare against Transformer-based architectures
