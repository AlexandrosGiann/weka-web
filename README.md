# Weka Web

The core features of the **Weka 3.8.7 Explorer**, rewritten in plain **HTML, CSS and JavaScript**. It needs no Java, Node or server, and it uses no external libraries. It also includes responsible **web data import**, a **Python / Colab** page that runs OR-Tools and Pyro notebooks in the browser and converts models to and from Google Colab, and a bilingual **AI assistant** that runs local models in the browser. The interface is in **English** by default, and the **ΕΛ / EN** button in the top bar switches to Greek.

*Τα βασικά features του Weka Explorer σε καθαρό HTML/CSS/JavaScript. Η προεπιλεγμένη γλώσσα είναι τα αγγλικά, και το κουμπί **ΕΛ** επάνω δεξιά αλλάζει σε ελληνικά.*

## Getting started

Double-click `index.html` to open it in Chrome, Edge or Firefox. Everything runs locally in the browser.

For the assistant's AI mode, voice commands and image recognition, serve the folder over http(s) — for example publish it with GitHub Pages, or run

```bash
python -m http.server 8000
```

in the folder and open <http://localhost:8000>. Browsers block Web Workers and model downloads on `file://`.

## Files delivered

| File / folder | What it is |
|---|---|
| `index.html`, `css/`, `js/` | The application |
| `datasets/` | 22 built-in datasets: the Weka samples plus **bank.csv** (UCI Bank Marketing) |
| `videos/` | 22 tutorial videos (11 topics × English and Greek), also shown on the **Video tutorials** page |
| `Bank_Marketing_Report_EN.pdf`, `Bank_Marketing_Report_EL.pdf` | The full Bank Marketing walkthrough, run in the app and exported with the **Report (PDF)** page |
| `models/` | Local AI models for the assistant (Needle 2, Whistle, MobileCLIP-S0) with their licences — see [models/README.md](models/README.md) |

## Features

| Area | What is included |
|---|---|
| Data | ARFF (including sparse data and weights), CSV/TSV (`,` `;` or tab), JSON; save as ARFF/CSV; data table with sort, search and editing; undo/redo |
| Preprocess | Statistics and class-coloured histograms; 19 filters: ReplaceMissingValues, Normalize, Standardize, Remove, Discretize (unsupervised & MDL), NominalToBinary (unsupervised & supervised), NumericToNominal, StringToNominal, RemoveUseless, AddExpression, Randomize, Resample (unsupervised & supervised), ClassBalancer, RemovePercentage, RemoveWithValues, RemoveDuplicates |
| Classify | 25 algorithms: ZeroR, OneR, **JRip**, PART, NaiveBayes, NaiveBayesMultinomial, **BayesNet**, J48, DecisionStump, RandomTree, RandomForest, REPTree, IBk, **KStar**, LinearRegression, SimpleLinearRegression, Logistic, SMO, **SGD**, **VotedPerceptron**, MultilayerPerceptron, AdaBoostM1, **LogitBoost**, Bagging, Vote. Test options: cross-validation, percentage split, training set, supplied test set. Results include the confusion matrix, per-class metrics, ROC/PR curves, a classifier-errors plot, tree diagrams, and **model comparison with business costs** (cost of a false negative and a false positive) |
| Cluster | SimpleKMeans, EM (with automatic k and a progress bar), FarthestFirst, HierarchicalClusterer (dendrogram). Also: classes-to-clusters evaluation, ignored attributes, a **segment × attribute cross-tab** (e.g. conversion rate per cluster), cluster plots coloured by any attribute, and AddCluster |
| Associate | Apriori (confidence, lift, leverage, conviction), FPGrowth |
| Select attributes | InfoGain, GainRatio, SymmetricalUncert, Correlation, ReliefF, OneR, CFS, Wrapper, PCA, with Ranker, BestFirst or GreedyStepwise; full training set or cross-validation |
| Visualize | Scatter-plot matrix, zoomable scatter plot with jitter and tooltips |
| **Web data** | Import tables from public web pages (HTML tables), CSV/TSV, JSON and ARFF files, and Wikipedia articles (official MediaWiki API). Safeguards: an explicit responsible-use statement, a robots.txt check before every page, at most 1 request per 5 s per site and 10 per minute, 10 MB / 20 s caps, no cookies or credentials, private-network addresses blocked, no CORS proxies, and the source URL and licence are added to the report |
| **Python / Colab** | **Run Colab notebooks in the browser:** import a `.ipynb` / `.py` and run cells that use OR-Tools CP-SAT (integer and interval variables, linear constraints, `AddNoOverlap`, `AddCumulative`, `AddMaxEquality`…), Pyro / torch (Monte Carlo sampling), numpy, math, random and matplotlib — with Gantt charts, histograms and the saved Colab output for comparison. **Planner:** tasks, durations, precedences, workers and uncertainty → real OR-Tools + Pyro code, solved here and downloadable as a notebook. **Weka ⇄ Python:** every classifier or clusterer run exports to a Colab notebook (closest scikit-learn model + an exact python-weka-wrapper3 cell), and scikit-learn / python-weka-wrapper3 code is recognised and run as Weka algorithms. A small Python interpreter (`js/minipy.js`) and a CP-SAT-compatible solver (`js/planning.js`) make this work offline, also from `file://` |
| **AI assistant** | A floating chat in English or Greek that performs actions: load data, describe it, set the class, remove attributes, apply filters, run and compare classifiers, cluster, find rules, rank attributes, explain results, export the PDF, open tutorials, import web data, export to Colab, import notebooks and plan tasks (“schedule design=3, build=5, design before build, one worker”). It answers ~40 data-mining questions from a curated knowledge base and (optionally) short, credited Wikipedia summaries. Optional local models: **Needle 2** for free-form commands (every AI-chosen action needs your confirmation), **Whistle** for voice commands, **MobileCLIP-S0** for image recognition and turning a folder of images into a dataset of embeddings |
| **Report (PDF)** | Collects the whole session — dataset, preprocessing steps, every result with its charts and tables, and your title and conclusions. Export it as PDF (the print dialog, “Save as PDF”) or as HTML |
| **Video tutorials** | Getting started · Preprocessing · Classification · Five new classifiers · Clustering · Association rules & attribute selection · Web data · Python / Colab · The AI assistant · Report → PDF · Bank Marketing case study. Each video has chapters you can click |

## Fidelity to Weka

All results were compared with the original `weka.jar` 3.8.7. The comparison used the same `java.util.Random`, folds and stratification.

**Identical to Weka:**
- **J48, JRip, NaiveBayes, OneR, ZeroR, IBk, DecisionStump, SMO and SimpleLinearRegression** give the same 10-fold CV metrics on up to 11 datasets.
- **BayesNet** (K2, 1 or 2 parents), **KStar** (all four missing-value modes, classification and regression), **LogitBoost**, **SGD** (hinge, log and squared loss) and **VotedPerceptron** give the same 10-fold CV metrics on 5–8 datasets each.
- **J48, NaiveBayes, Logistic and JRip** match on the bank data, both with and without `duration`.
- **LinearRegression** matches on `cpu`.
- **SimpleKMeans** gives the same SSE and the same cluster sizes.
- **Apriori** produces exactly the same text output for all four metrics.
- **InfoGain, GainRatio, SymmetricalUncert, Correlation, ReliefF and CFS + BestFirst** give the same ranking and the same selection.

**Small differences:**
- Logistic can differ at the 4th decimal on nearly separable data, and MultilayerPerceptron differs slightly.
- RandomForest, RandomTree, REPTree, EM, PART and the meta classifiers follow the same algorithms, but they do not reproduce Weka's random choices bit for bit.

Weka is developed by the University of Waikato (GPL). This port is independent of it.
