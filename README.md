# Model Canvas v2 🎨

An interactive machine learning visualization platform that brings algorithms to life through real-time decision boundary visualization and performance metrics.

![Model Canvas](https://img.shields.io/badge/ML-Visualization-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-green) ![JavaScript](https://img.shields.io/badge/JavaScript-ES6%2B-yellow) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 🌟 Live Demo

🚀 **Experience Model Canvas**: [Live Demo](https://model-canvas-v2.vercel.app/) 

📁 **Source Code**: [GitHub Repository](https://github.com/Shubhadip1004/Model-Canvas-v2)

## 📖 Overview

Model Canvas v2 is an educational platform designed to help students, researchers, and ML enthusiasts understand how machine learning algorithms work through interactive visualizations. Watch decision boundaries form in real-time as models train, and monitor performance metrics live.

### Key Features

- **🔬 Real-time Visualization**: Watch decision boundaries evolve during training
- **📊 Multiple Algorithms**: Logistic Regression, KNN, SVM, Decision Trees, Random Forest, Neural Networks
- **🎯 Diverse Datasets**: Iris, Wine, Breast Cancer, Diabetes, and synthetic datasets
- **⚡ Dual Training Modes**: Educational (step-by-step) vs Optimized (full speed)
- **📈 Live Metrics**: Accuracy, loss tracking, and confusion matrices
- **🔍 Feature Views**: Switch between raw features and PCA projections
- **🎨 Professional UI**: Dark/light themes with responsive design

## 🛠️ Tech Stack

### Frontend
- **HTML5** - Semantic structure
- **CSS3** - Modern styling with CSS variables
- **JavaScript (ES6+)** - Interactive functionality
- **Plotly.js** - Advanced data visualization

### Backend
- **Python** - Machine learning backend
- **Flask** - REST API server
- **Scikit-learn** - ML algorithms implementation
- **NumPy & Pandas** - Data processing

### Deployment
- **Render** - Backend hosting
- **Vercel** - Frontend hosting
- **GitHub** - Version control

## 🚀 Quick Start

### Prerequisites
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Internet connection for API calls

### Using the Platform

1. **Select Dataset**: Choose from built-in datasets or synthetic data
2. **Choose Algorithm**: Pick from 6 different ML algorithms
3. **Adjust Parameters**: Tune hyperparameters using intuitive controls
4. **Start Training**: Watch real-time visualization and metrics
5. **Analyze Results**: Compare performance across different views

## 🎯 Supported Algorithms

| Algorithm | Type | Key Parameters | Best For |
|-----------|------|----------------|----------|
| **Logistic Regression** | Linear | Regularization (C) | Linear separability |
| **K-Nearest Neighbors** | Instance-based | Number of neighbors (k) | Non-linear patterns |
| **Support Vector Machine** | Kernel-based | Kernel, C parameter | Complex boundaries |
| **Decision Tree** | Tree-based | Max depth | Interpretable rules |
| **Random Forest** | Ensemble | Number of trees | Robust performance |
| **Neural Network** | Deep Learning | Layers, Learning rate | Complex patterns |

## 📊 Dataset Information

| Dataset | Samples | Features | Classes | Description |
|---------|---------|----------|---------|-------------|
| **Iris** | 150 | 4 | 3 | Classic classification dataset |
| **Wine** | 178 | 13 | 3 | Wine chemical analysis |
| **Breast Cancer** | 569 | 30 | 2 | Medical diagnosis data |
| **Diabetes** | 442 | 10 | 2 | Disease progression |
| **Make Moons** | 100+ | 2 | 2 | Synthetic non-linear data |
| **Make Circles** | 100+ | 2 | 2 | Concentric circle data |
| **Make Blobs** | 100+ | 2 | 3 | Gaussian clusters |

## 🎨 UI/UX Features

### Interactive Controls
- **Real-time Parameter Adjustment**: Modify hyperparameters on the fly
- **Dual View Modes**: Toggle between raw features and PCA projections
- **Training Controls**: Play, pause, and reset training sessions
- **Theme Switching**: Dark/light mode for comfortable viewing

### Visualization Capabilities
- **Decision Boundaries**: Watch algorithms learn separation boundaries
- **Performance Metrics**: Live accuracy and loss graphs
- **Confusion Matrices**: Final model performance analysis
- **Data Point Tracking**: Correct vs incorrect predictions

### Educational Features
- **Step-by-Step Mode**: See each iteration of model training
- **Algorithm Comparisons**: Understand different ML approaches
- **Parameter Effects**: Observe how hyperparameters impact learning
- **Visual Feedback**: Immediate visual response to changes

## 🏗️ Project Structure

    model-canvas-v2/
    ├── images/                 # Relevant Images incl. website logo
    |   └── Model_Canvas.ico
    ├── Scrennshots/            # Live Website Screenshots
    │   ├── Screenshot 1.png          
    │   ├── Screenshot 2.png            
    │   ├── Screenshot 3.png            
    │   ├── Screenshot 4.png               
    │   └── Screenshot 5.png             
    ├── frontend/
    │   ├── index.html          # Main application structure
    │   ├── style.css           # Comprehensive styling
    │   ├── app.js              # Application logic
    │   └── plot.js             # Plotly visualization handlers
    └── backend/
        ├── app.py              # Flask/FastAPI server
        ├── models/             # ML algorithm implementations
        |   ├── init.py
        |   ├── decision_tree.py
        |   ├── knn.py
        |   ├── logistic_reg.py
        |   ├── neural_net.py
        |   ├── random_forest.py
        |   └── svm.py
        ├── utils/              # Data loading and boundary plotting
        |   ├── data_loader.py
        |   └── boundary_plot.py
        ├── runtime.txt
        └── requirements.txt    # Python dependencies


## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, fork the repository, and create pull requests.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution
- New algorithm implementations
- Additional dataset support
- UI/UX improvements
- Performance optimizations
- Documentation enhancements

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Scikit-learn** team for the robust ML library
- **Plotly** team for excellent visualization capabilities
- **Vercel** for awesome hosting services
- **Render** for reliable hosting services
- The open-source community for continuous inspiration

## 📞 Contact

**Shubhadip Mahata**  
- GitHub: [@Shubhadip1004](https://github.com/Shubhadip1004)
- Email: shubhadip.w@gmail.com
- Project Link: [https://github.com/Shubhadip1004/Model-Canvas-v2](https://github.com/Shubhadip1004/Model-Canvas-v2)

## 🚀 Future Enhancements

- [ ] Additional algorithms (XGBoost, LightGBM)
- [ ] Regression problem support
- [ ] Custom dataset upload
- [ ] Model export functionality
- [ ] Collaborative features
- [ ] Advanced visualization options

<div align="center">

**⭐ Star this repository if you find it helpful!**

*Making machine learning accessible through visualization* 🎨

</div>
