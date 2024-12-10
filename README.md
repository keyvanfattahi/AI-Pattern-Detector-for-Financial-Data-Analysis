# AI Pattern Detector for Financial Data Analysis

## 🇬🇧 English Description

An advanced AI-powered tool designed to detect and analyze patterns in financial time series data using a combination of LSTM neural networks and clustering algorithms. This tool is specifically optimized for processing Excel files containing financial indicators and sentiment data.

### Key Features
- Processes Excel files with both numerical and color-coded sentiment data
- Supports Persian (Jalali) calendar dates
- Implements a hybrid LSTM-KMeans architecture for pattern detection
- Provides comprehensive pattern analysis with statistical metrics
- Generates interactive visualizations using Plotly
- Handles missing data with intelligent imputation
- Includes sentiment analysis based on cell colors and text content

### Technical Details
The system employs a sophisticated architecture combining:
- LSTM neural networks for sequence learning
- KMeans clustering for pattern categorization
- MinMaxScaler for data normalization
- Interactive Plotly visualizations
- Pandas and NumPy for data processing
- Support for both numerical and categorical data analysis

### Requirements
```
pandas
numpy
sklearn
tensorflow
plotly
seaborn
matplotlib
jdatetime
openpyxl
```

### Usage
1. Initialize the detector:
```python
detector = AIPatternDetector(sequence_length=7)
```

2. Load and process data:
```python
detector.read_excel_with_colors("path_to_file.xlsx")
detector.preprocess_data()
```

3. Train models and analyze patterns:
```python
patterns = detector.train_models(n_clusters=5)
analysis = detector.analyze_patterns()
```

## 🇮🇷 توضیحات فارسی

یک ابزار پیشرفته مبتنی بر هوش مصنوعی برای تشخیص و تحلیل الگوها در داده‌های سری زمانی مالی با استفاده از ترکیبی از شبکه‌های عصبی LSTM و الگوریتم‌های خوشه‌بندی. این ابزار به طور خاص برای پردازش فایل‌های اکسل حاوی شاخص‌های مالی و داده‌های احساسی بهینه‌سازی شده است.

### ویژگی‌های کلیدی
- پردازش فایل‌های اکسل با داده‌های عددی و احساسی کدگذاری شده با رنگ
- پشتیبانی از تاریخ‌های تقویم فارسی (جلالی)
- پیاده‌سازی معماری ترکیبی LSTM-KMeans برای تشخیص الگو
- ارائه تحلیل جامع الگو با معیارهای آماری
- تولید نمودارهای تعاملی با استفاده از Plotly
- مدیریت داده‌های گمشده با درون‌یابی هوشمند
- شامل تحلیل احساسات بر اساس رنگ‌ها و محتوای متنی سلول‌ها

### جزئیات فنی
سیستم از یک معماری پیچیده ترکیبی استفاده می‌کند که شامل:
- شبکه‌های عصبی LSTM برای یادگیری توالی
- خوشه‌بندی KMeans برای دسته‌بندی الگوها
- MinMaxScaler برای نرمال‌سازی داده‌ها
- نمودارهای تعاملی Plotly
- Pandas و NumPy برای پردازش داده‌ها
- پشتیبانی از تحلیل داده‌های عددی و دسته‌ای

### نیازمندی‌ها
```
pandas
numpy
sklearn
tensorflow
plotly
seaborn
matplotlib
jdatetime
openpyxl
```

### نحوه استفاده
۱. راه‌اندازی تشخیص‌دهنده:
```python
detector = AIPatternDetector(sequence_length=7)
```

۲. بارگذاری و پردازش داده‌ها:
```python
detector.read_excel_with_colors("path_to_file.xlsx")
detector.preprocess_data()
```

۳. آموزش مدل‌ها و تحلیل الگوها:
```python
patterns = detector.train_models(n_clusters=5)
analysis = detector.analyze_patterns()
```
