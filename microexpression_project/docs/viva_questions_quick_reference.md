# Viva Questions & Answers - Quick Reference

## 🎯 Core Project Questions

### 1. **Project Overview**
**Q: Can you briefly explain your project?**
**A**: I developed a micro-expression detection system using CNN and computer vision techniques. The system analyzes subtle facial expressions from the CASME-II dataset to classify four emotions: happiness, surprise, disgust, and repression using a hybrid approach combining optical flow, strain analysis, and deep learning features.

### 2. **Motivation**
**Q: Why is micro-expression detection important?**
**A**: Micro-expressions reveal genuine emotions that people cannot fake, making them valuable for clinical psychology, security applications, lie detection, and understanding human behavior in various fields.

### 3. **Dataset Choice**
**Q: Why did you choose CASME-II dataset?**
**A**: CASME-II provides high-resolution (640x480) videos at 200fps, which is essential for capturing subtle micro-expressions. It's well-labeled with precise onset/offset frames and is a standardized benchmark in the research community.

---

## 🔧 Technical Questions

### 4. **Feature Extraction**
**Q: What features did you extract and why?**
**A**: I used a hybrid approach:
- **48-D handcrafted features**: Motion magnitude statistics
- **40-D AU-aligned features**: Strain patterns in specific facial regions (AU4, AU6, AU9, AU10, AU12)
- **128-D CNN features**: Deep learning features from optical flow and strain maps
This combines domain knowledge with automatic feature learning.

### 5. **Optical Flow**
**Q: Explain optical flow and its role in your system.**
**A**: Optical flow computes motion vectors between consecutive frames. It's crucial for micro-expressions because it captures subtle facial movements, provides direction information, is robust to lighting changes, and reveals temporal dynamics of expressions.

### 6. **Model Architecture**
**Q: Why use SVM instead of end-to-end deep learning?**
**A**: SVM was chosen because CASME-II is relatively small for deep learning, it provides better interpretability, is less prone to overfitting, and our hybrid features provide strong input that works well with SVM's decision boundaries.

### 7. **Validation Method**
**Q: What is LOSO validation and why is it important?**
**A**: LOSO (Leave-One-Subject-Out) validation trains on all subjects except one, then tests on the left-out subject. It's important because it evaluates generalization to unseen individuals and prevents subject-specific bias, providing more realistic performance estimates.

---

## 🧠 Advanced Technical Questions

### 8. **Action Units**
**Q: What are Action Units and why AU-aligned features?**
**A**: Action Units (AUs) are fundamental facial muscle movements defined in the FACS system. AU-aligned features focus on anatomically relevant regions, making them more interpretable and reducing noise by concentrating on emotion-relevant areas.

### 9. **Temporal Processing**
**Q: How do you handle the temporal aspect of micro-expressions?**
**A**: I process 10-frame sequences, use optical flow to capture motion dynamics, extract statistics across temporal dimensions, and identify key frames (onset, apex, offset) for analysis.

### 10. **Preprocessing Pipeline**
**Q: Describe your preprocessing steps.**
**A**: Raw video → Face detection → Face cropping → Resizing to 64×64 → Normalization (pixel/255.0) → Frame selection → Feature extraction.

---

## 📊 Evaluation & Performance

### 11. **Performance Metrics**
**Q: How do you evaluate your system's performance?**
**A**: I use accuracy, confusion matrix, F1-score, and LOSO validation. Subject-dependent accuracy typically reaches 85-90%, while LOSO provides more realistic 65-75% accuracy.

### 12. **Challenges**
**Q: What are the main challenges in micro-expression detection?**
**A**: Key challenges include subtlety of expressions, individual variability, data scarcity, sensitivity to lighting/pose, and computational efficiency for real-time applications.

### 13. **Limitations**
**Q: What are the limitations of your current approach?**
**A**: Limited dataset size, dependency on accurate face detection, computational complexity of optical flow, and lower LOSO performance compared to subject-dependent evaluation.

---

## 🚀 Future Work & Improvements

### 14. **System Improvements**
**Q: How would you improve this system?**
**A**: I'd explore 3D CNNs for spatial-temporal features, attention mechanisms to focus on relevant regions, transfer learning from larger datasets, multi-modal fusion with physiological signals, and ensemble methods.

### 15. **Real-world Applications**
**Q: Where could this system be applied?**
**A**: Clinical psychology for emotion assessment, security and lie detection, human-computer interaction, market research, and educational applications for understanding emotional responses.

### 16. **Ethical Considerations**
**Q: What ethical considerations are important?**
**A**: Privacy concerns, consent for data collection, potential misuse in surveillance, cultural differences in expression, and ensuring responsible deployment.

---

## 💻 Implementation Questions

### 17. **Technical Stack**
**Q: What technologies did you use?**
**A**: Python with OpenCV for computer vision, PyTorch for CNN features, scikit-learn for SVM, Flask for web interface, and matplotlib for visualizations.

### 18. **Web Interface**
**Q: How does your web application work?**
**A**: The Flask-based web app allows users to upload videos or use webcam for real-time processing. It preprocesses the video, extracts features, runs prediction, and displays results with confidence scores.

### 19. **Performance Optimization**
**Q: How do you optimize for real-time performance?**
**A**: I use efficient face detection, optimized optical flow computation, feature caching, and model quantization to achieve processing times of ~0.1 seconds per frame.

---

## 🎯 Research Context Questions

### 20. **Literature Review**
**Q: How does your work compare to existing research?**
**A**: My hybrid approach combines strengths of traditional methods (interpretability) and deep learning (automatic feature learning). Performance is competitive with state-of-the-art while maintaining better explainability.

### 21. **Novel Contributions**
**Q: What is novel about your approach?**
**A**: The AU-aligned feature extraction provides domain knowledge integration, the hybrid feature combination balances interpretability and performance, and the comprehensive evaluation includes both subject-dependent and LOSO protocols.

### 22. **Reproducibility**
**Q: How do you ensure reproducibility?**
**A**: I use deterministic frame selection, fixed random seeds, comprehensive documentation, standardized preprocessing pipeline, and provide complete code and trained models.

---

## 🔥 Quick Fire Questions

### 23. **Dataset Size**
**Q: How many samples in your dataset?**
**A**: ~247 micro-expression samples from 26 subjects in CASME-II.

### 24. **Emotion Classes**
**Q: How many emotion classes?**
**A**: 4 classes: Happiness, Surprise, Disgust, Repression.

### 25. **Feature Dimension**
**Q: What's your feature vector size?**
**A**: 216-dimensional total (48 + 40 + 128).

### 26. **Processing Time**
**Q: How fast is your system?**
**A**: Approximately 0.1 seconds per frame for real-time processing.

### 27. **Model Size**
**Q: What's your model size?**
**A**: Around 500MB including all components.

---

## 🎓 Viva Success Tips

### Key Points to Emphasize:
1. **Hybrid Approach**: Best of both worlds (traditional + deep learning)
2. **Domain Knowledge**: AU alignment shows understanding of facial anatomy
3. **Rigorous Evaluation**: LOSO validation demonstrates real-world applicability
4. **Complete System**: From preprocessing to web deployment
5. **Reproducibility**: Deterministic pipeline ensures reliable results

### Questions to Ask Examiners:
1. "Would you like me to demonstrate the web interface?"
2. "Should I explain any specific component in more detail?"
3. "Are you interested in the performance metrics or technical implementation?"
4. "Would you like to see the confusion matrix results?"

### Common Follow-ups:
- Be prepared to explain any confusion matrix patterns
- Know your best and worst performing emotion classes
- Understand why certain emotions are harder to classify
- Be ready to discuss specific technical implementation choices

---

## 📋 Last Minute Checklist

### Before Viva:
- [ ] Review code structure and key functions
- [ ] Test web application demo
- [ ] Prepare confusion matrix visualizations
- [ ] Know your performance numbers
- [ ] Practice explaining technical concepts

### During Viva:
- [ ] Speak clearly and confidently
- [ ] Use visualizations when explaining
- [ ] Acknowledge limitations honestly
- [ ] Show enthusiasm for your work
- [ ] Ask for clarification if needed

Good luck! You've built a comprehensive system that demonstrates strong technical skills and research understanding.
