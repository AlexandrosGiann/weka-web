# Weka Web

The core features of the **Weka 3.8.7 Explorer**, rewritten in plain **HTML, CSS and JavaScript**. It needs no Java, Node or server, and it uses no external libraries. The interface is in **English** by default, and the **ΕΛ / EN** button in the top bar switches to Greek.

*Τα βασικά features του Weka Explorer σε καθαρό HTML/CSS/JavaScript. Η προεπιλεγμένη γλώσσα είναι τα αγγλικά, και το κουμπί **ΕΛ** επάνω δεξιά αλλάζει σε ελληνικά.*

## Getting started

Double-click `index.html` to open it in Chrome, Edge or Firefox. Everything runs locally in the browser.

## Files delivered

| File / folder | What it is |
|---|---|
| `index.html`, `css/`, `js/` | The application |
| `datasets/` | 22 built-in datasets: the Weka samples plus **bank.csv** (UCI Bank Marketing) |
| `videos/` | 14 tutorial videos (7 topics × English and Greek), also shown on the **Video tutorials** page |
| `Bank_Marketing_Report_EN.pdf`, `Bank_Marketing_Report_EL.pdf` | The full Bank Marketing walkthrough, run in the app and exported with the **Report (PDF)** page |

## Features

| Area | What is included |
|---|---|
| Data | ARFF (including sparse data and weights), CSV/TSV (`,` `;` or tab), JSON; save as ARFF/CSV; data table with sort, search and editing; undo/redo |
| Preprocess | Statistics and class-coloured histograms; 18 filters: ReplaceMissingValues, Normalize, Standardize, Remove, Discretize (unsupervised & MDL), NominalToBinary (unsupervised & supervised), NumericToNominal, StringToNominal, RemoveUseless, AddExpression, Randomize, Resample (unsupervised & supervised), ClassBalancer, RemovePercentage, RemoveWithValues, RemoveDuplicates |
| Classify | ZeroR, OneR, **JRip**, PART, NaiveBayes, NaiveBayesMultinomial, J48, DecisionStump, RandomTree, RandomForest, REPTree, IBk, LinearRegression, SimpleLinearRegression, Logistic, SMO, MultilayerPerceptron, AdaBoostM1, Bagging, Vote. Test options: cross-validation, percentage split, training set, supplied test set. Results include the confusion matrix, per-class metrics, ROC/PR curves, a classifier-errors plot, tree diagrams, and **model comparison with business costs** (cost of a false negative and a false positive) |
| Cluster | SimpleKMeans, EM (with automatic k and a progress bar), FarthestFirst, HierarchicalClusterer (dendrogram). Also: classes-to-clusters evaluation, ignored attributes, a **segment × attribute cross-tab** (e.g. conversion rate per cluster), cluster plots coloured by any attribute, and AddCluster |
| Associate | Apriori (confidence, lift, leverage, conviction), FPGrowth |
| Select attributes | InfoGain, GainRatio, SymmetricalUncert, Correlation, ReliefF, OneR, CFS, Wrapper, PCA, with Ranker, BestFirst or GreedyStepwise; full training set or cross-validation |
| Visualize | Scatter-plot matrix, zoomable scatter plot with jitter and tooltips |
| **Report (PDF)** | Collects the whole session — dataset, preprocessing steps, every result with its charts and tables, and your title and conclusions. Export it as PDF (the print dialog, “Save as PDF”) or as HTML |
| **Video tutorials** | Getting started · Preprocessing · Classification · Clustering · Association rules & attribute selection · Report → PDF · Bank Marketing case study. Each video has chapters you can click |

## Fidelity to Weka

All results were compared with the original `weka.jar` 3.8.7. The comparison used the same `java.util.Random`, folds and stratification.

**Identical to Weka:**
- **J48, JRip, NaiveBayes, OneR, ZeroR, IBk, DecisionStump, SMO and SimpleLinearRegression** give the same 10-fold CV metrics on up to 11 datasets.
- **J48, NaiveBayes, Logistic and JRip** match on the bank data, both with and without `duration`.
- **LinearRegression** matches on `cpu`.
- **SimpleKMeans** gives the same SSE and the same cluster sizes.
- **Apriori** produces exactly the same text output for all four metrics.
- **InfoGain, GainRatio, SymmetricalUncert, Correlation, ReliefF and CFS + BestFirst** give the same ranking and the same selection.

**Small differences:**
- Logistic can differ at the 4th decimal on nearly separable data, and MultilayerPerceptron differs slightly.
- RandomForest, RandomTree, REPTree, EM, PART and the meta classifiers follow the same algorithms, but they do not reproduce Weka's random choices bit for bit.

Weka is developed by the University of Waikato (GPL). This port is independent of it.
