/* Weka Web — Python bridge: Weka Web ⇄ Python / Google Colab.
 *  - toNotebook(): an algorithm run in Weka Web → a Colab notebook (.ipynb) or script (.py) with the closest
 *    scikit-learn model, plus an optional python-weka-wrapper3 cell that reproduces Weka's numbers exactly;
 *  - detect(): Python code (scikit-learn or python-weka-wrapper3) → the matching Weka Web algorithms and options;
 *  - plannerCode(): a task table → OR-Tools CP-SAT (and Pyro Monte Carlo) code that runs both in Colab and here. */
(function () {
  'use strict';
  const W = window.WEKA;
  const pyStr = (s) => JSON.stringify(String(s));
  const pyNum = (v) => (typeof v === 'boolean' ? (v ? 'True' : 'False') : Number.isInteger(v) ? String(v) : String(+v));

  // ---------------------------------------------------------------- notebook files
  function notebook(cells, title) {
    const lines = (s) => { const a = s.replace(/\s+$/, '').split('\n'); return a.map((l, i) => (i < a.length - 1 ? l + '\n' : l)); };
    return JSON.stringify({
      cells: cells.map((c) => (c.md ? { cell_type: 'markdown', metadata: {}, source: lines(c.md) } : { cell_type: 'code', execution_count: null, metadata: {}, outputs: [], source: lines(c.code) })),
      metadata: { colab: { name: title || 'weka-web.ipynb', provenance: [] }, kernelspec: { display_name: 'Python 3', language: 'python', name: 'python3' }, language_info: { name: 'python' } },
      nbformat: 4, nbformat_minor: 0,
    }, null, 1);
  }
  function script(cells) {
    return cells.map((c) => (c.md ? c.md.split('\n').map((l) => '# ' + l).join('\n') : c.code.split('\n').map((l) => (/^\s*[!%]/.test(l) ? '# ' + l + '   (run this in a terminal / Colab cell)' : l)).join('\n'))).join('\n\n') + '\n';
  }
  /** Reads .ipynb JSON or .py text into [{type, src}]; "# %%" markers split .py files into cells. */
  function parseNotebook(text, fileName = '') {
    if (/\.ipynb$/i.test(fileName) || /^\s*\{/.test(text)) {
      const j = JSON.parse(text);
      if (!Array.isArray(j.cells)) throw new Error('Not a Jupyter notebook (no "cells").');
      return j.cells.filter((c) => c.cell_type === 'code' || c.cell_type === 'markdown').map((c) => ({ type: c.cell_type, src: (Array.isArray(c.source) ? c.source.join('') : String(c.source || '')).replace(/\s+$/, ''), saved: savedOutput(c) }));
    }
    const parts = text.split(/^# ?%%.*$/m).map((s) => s.replace(/^\n+|\s+$/g, '')).filter(Boolean);
    return parts.map((src) => ({ type: 'code', src }));
  }
  function savedOutput(c) {
    if (!c.outputs || !c.outputs.length) return null;
    return c.outputs.map((o) => (o.text ? (Array.isArray(o.text) ? o.text.join('') : o.text) : o.data && o.data['text/plain'] ? [].concat(o.data['text/plain']).join('') : o.ename ? o.ename + ': ' + o.evalue : '')).join('');
  }

  // ---------------------------------------------------------------- Weka → scikit-learn
  const C = (o, k, d) => (o && o[k] !== undefined ? o[k] : d);
  const SK = {
    ZeroR: (o, nom) => ({ imp: nom ? 'from sklearn.dummy import DummyClassifier' : 'from sklearn.dummy import DummyRegressor', ctor: nom ? "DummyClassifier(strategy='most_frequent')" : "DummyRegressor(strategy='mean')", exact: true }),
    OneR: () => ({ imp: 'from sklearn.tree import DecisionTreeClassifier', ctor: 'DecisionTreeClassifier(max_depth=1, random_state=1)', note: 'OneR uses one attribute with many intervals; a depth-1 tree is the closest scikit-learn model.' }),
    J48: (o) => ({ imp: 'from sklearn.tree import DecisionTreeClassifier', ctor: `DecisionTreeClassifier(criterion='entropy', min_samples_leaf=${C(o, 'M', 2)}, random_state=1)`, note: 'C4.5 pruning (confidence factor) has no exact scikit-learn equivalent.' }),
    DecisionStump: (o, nom) => ({ imp: nom ? 'from sklearn.tree import DecisionTreeClassifier' : 'from sklearn.tree import DecisionTreeRegressor', ctor: nom ? "DecisionTreeClassifier(max_depth=1, criterion='entropy')" : 'DecisionTreeRegressor(max_depth=1)' }),
    RandomTree: (o, nom) => ({ imp: nom ? 'from sklearn.tree import DecisionTreeClassifier' : 'from sklearn.tree import DecisionTreeRegressor', ctor: `${nom ? 'DecisionTreeClassifier' : 'DecisionTreeRegressor'}(max_features='sqrt', random_state=${C(o, 'S', 1)})` }),
    REPTree: (o, nom) => ({ imp: nom ? 'from sklearn.tree import DecisionTreeClassifier' : 'from sklearn.tree import DecisionTreeRegressor', ctor: `${nom ? 'DecisionTreeClassifier' : 'DecisionTreeRegressor'}(min_samples_leaf=2, ccp_alpha=0.001, random_state=1)` }),
    RandomForest: (o, nom) => ({ imp: nom ? 'from sklearn.ensemble import RandomForestClassifier' : 'from sklearn.ensemble import RandomForestRegressor', ctor: `${nom ? 'RandomForestClassifier' : 'RandomForestRegressor'}(n_estimators=${C(o, 'I', 100)}, max_features=${C(o, 'K', 0) ? C(o, 'K', 0) : "'sqrt'"}, random_state=${C(o, 'S', 1)})` }),
    NaiveBayes: () => ({ imp: 'from sklearn.naive_bayes import GaussianNB', ctor: 'GaussianNB()', note: 'Nominal attributes are one-hot encoded; Weka models them with frequency tables.' }),
    NaiveBayesMultinomial: () => ({ imp: 'from sklearn.naive_bayes import MultinomialNB', ctor: 'MultinomialNB()' }),
    BayesNet: () => ({ imp: 'from sklearn.naive_bayes import CategoricalNB', ctor: 'CategoricalNB(alpha=0.5)', discretize: true, note: 'Default BayesNet (K2, 1 parent) is naive Bayes on discretised data — CategoricalNB after KBinsDiscretizer.' }),
    IBk: (o, nom) => ({ imp: nom ? 'from sklearn.neighbors import KNeighborsClassifier' : 'from sklearn.neighbors import KNeighborsRegressor', ctor: `${nom ? 'KNeighborsClassifier' : 'KNeighborsRegressor'}(n_neighbors=${C(o, 'K', 1)})`, scale: true }),
    KStar: (o, nom) => ({ imp: nom ? 'from sklearn.neighbors import KNeighborsClassifier' : 'from sklearn.neighbors import KNeighborsRegressor', ctor: `${nom ? 'KNeighborsClassifier' : 'KNeighborsRegressor'}(n_neighbors=15, weights='distance')`, scale: true, note: 'K* (entropic distance) has no scikit-learn equivalent; distance-weighted k-NN is the closest.' }),
    LinearRegression: () => ({ imp: 'from sklearn.linear_model import LinearRegression', ctor: 'LinearRegression()' }),
    SimpleLinearRegression: () => ({ imp: 'from sklearn.linear_model import LinearRegression', ctor: 'LinearRegression()', note: 'Weka picks the single best attribute; this uses all attributes.' }),
    Logistic: (o) => ({ imp: 'from sklearn.linear_model import LogisticRegression', ctor: `LogisticRegression(C=${pyNum(1 / Math.max(C(o, 'R', 1e-8), 1e-12))}, max_iter=5000)`, scale: true }),
    SMO: (o) => ({ imp: 'from sklearn.svm import SVC', ctor: o.kernel === 'rbf' ? `SVC(kernel='rbf', C=${pyNum(C(o, 'C', 1))}, gamma=${pyNum(C(o, 'G', 0.01))})` : C(o, 'E', 1) !== 1 ? `SVC(kernel='poly', degree=${C(o, 'E', 1)}, C=${pyNum(C(o, 'C', 1))}, coef0=0, gamma=1)` : `SVC(kernel='linear', C=${pyNum(C(o, 'C', 1))})`, scale: true }),
    SGD: (o, nom) => ({ imp: nom ? 'from sklearn.linear_model import SGDClassifier' : 'from sklearn.linear_model import SGDRegressor', ctor: nom ? `SGDClassifier(loss='${+C(o, 'F', 0) === 1 ? 'log_loss' : 'hinge'}', alpha=${pyNum(C(o, 'R', 1e-4))}, learning_rate='constant', eta0=${pyNum(C(o, 'L', 0.01))}, max_iter=${C(o, 'E', 500)}, random_state=${C(o, 'S', 1)})` : `SGDRegressor(alpha=${pyNum(C(o, 'R', 1e-4))}, learning_rate='constant', eta0=${pyNum(C(o, 'L', 0.01))}, max_iter=${C(o, 'E', 500)}, random_state=${C(o, 'S', 1)})`, scale: true }),
    VotedPerceptron: (o) => ({ imp: 'from sklearn.linear_model import Perceptron', ctor: `Perceptron(max_iter=${Math.max(5, C(o, 'I', 1))}, random_state=${C(o, 'S', 1)})`, scale: true, note: 'scikit-learn has the classic perceptron, not the voted variant.' }),
    MultilayerPerceptron: (o, nom) => ({ imp: nom ? 'from sklearn.neural_network import MLPClassifier' : 'from sklearn.neural_network import MLPRegressor', ctor: `${nom ? 'MLPClassifier' : 'MLPRegressor'}(solver='sgd', learning_rate_init=${pyNum(C(o, 'L', 0.3))}, momentum=${pyNum(C(o, 'M', 0.2))}, max_iter=${C(o, 'N', 500)}, random_state=${C(o, 'S', 0)})`, scale: true }),
    AdaBoostM1: (o) => ({ imp: 'from sklearn.ensemble import AdaBoostClassifier\nfrom sklearn.tree import DecisionTreeClassifier', ctor: `AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=${C(o, 'I', 10)}, random_state=${C(o, 'S', 1)})` }),
    LogitBoost: (o) => ({ imp: 'from sklearn.ensemble import GradientBoostingClassifier', ctor: `GradientBoostingClassifier(n_estimators=${C(o, 'I', 10)}, max_depth=1, learning_rate=${pyNum(C(o, 'H', 1))}, random_state=1)`, note: 'Gradient boosting with stumps on the log-loss — the scikit-learn cousin of LogitBoost.' }),
    Bagging: (o, nom) => ({ imp: nom ? 'from sklearn.ensemble import BaggingClassifier' : 'from sklearn.ensemble import BaggingRegressor', ctor: `${nom ? 'BaggingClassifier' : 'BaggingRegressor'}(n_estimators=${C(o, 'I', 10)}, random_state=${C(o, 'S', 1)})` }),
    Vote: () => ({ imp: 'from sklearn.ensemble import VotingClassifier\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.naive_bayes import GaussianNB', ctor: "VotingClassifier([('tree', DecisionTreeClassifier()), ('nb', GaussianNB())], voting='soft')" }),
    JRip: () => ({ imp: 'from sklearn.tree import DecisionTreeClassifier', ctor: "DecisionTreeClassifier(criterion='entropy', min_samples_leaf=2, random_state=1)", note: 'RIPPER rules are not in scikit-learn — use the exact Weka cell below for JRip.' }),
    PART: () => ({ imp: 'from sklearn.tree import DecisionTreeClassifier', ctor: "DecisionTreeClassifier(criterion='entropy', min_samples_leaf=2, random_state=1)", note: 'PART rules are not in scikit-learn — use the exact Weka cell below for PART.' }),
    SimpleKMeans: (o) => ({ imp: 'from sklearn.cluster import KMeans', ctor: `KMeans(n_clusters=${C(o, 'N', 2)}, n_init=10, random_state=${C(o, 'S', 10)})`, scale: true, cluster: true }),
    EM: (o) => ({ imp: 'from sklearn.mixture import GaussianMixture', ctor: `GaussianMixture(n_components=${C(o, 'N', -1) > 0 ? C(o, 'N', -1) : 3}, random_state=${C(o, 'S', 100)})`, scale: true, cluster: true, note: C(o, 'N', -1) > 0 ? null : 'Weka chooses k by cross-validation; here k=3 — try several values.' }),
    FarthestFirst: (o) => ({ imp: 'from sklearn.cluster import KMeans', ctor: `KMeans(n_clusters=${C(o, 'N', 2)}, init='k-means++', n_init=1, max_iter=1, random_state=1)`, scale: true, cluster: true, note: 'FarthestFirst is a k-center heuristic; one k-means++ seeding step is close.' }),
    HierarchicalClusterer: (o) => ({ imp: 'from sklearn.cluster import AgglomerativeClustering', ctor: `AgglomerativeClustering(n_clusters=${C(o, 'N', 2)}, linkage='single')`, scale: true, cluster: true }),
  };

  /** source: {spec, options, data, mode?, folds?, seed?, kind: 'classifier'|'clusterer'} */
  function toNotebook(src) {
    const { spec, options = {}, data } = src;
    const kind = src.kind || (W.clusterers.includes(spec) ? 'clusterer' : 'classifier');
    const nominal = data.classIndex >= 0 && data.classIsNominal();
    const m = (SK[spec.name] || (() => ({ imp: '', ctor: null })))(options, nominal);
    const optStr = W.optionString(spec, options);
    const cls = data.classIndex >= 0 ? data.classAttr.name : null;
    const csv = W.toCSV(data);
    const big = csv.length > 3e6;
    const cells = [];
    cells.push({ md: `# ${spec.name} on “${data.relation}” — exported from Weka Web\n\nWeka command line: \`${spec.path}${optStr ? ' ' + optStr : ''}\`\n\n- **Part 1** runs the closest **scikit-learn** model (fast, no Java).${m.note ? '\n- Note: ' + m.note : ''}\n- **Part 2** (optional) runs the **real Weka** algorithm through python-weka-wrapper3, so the numbers match Weka Web exactly.\n\nOpen this file in Google Colab: *File → Upload notebook*.` });
    cells.push({ code: big ? `# The dataset is large: upload ${data.relation}.csv (Data table → Save as CSV in Weka Web) with the Colab file panel.\nimport pandas as pd\ndf = pd.read_csv(${pyStr(data.relation + '.csv')}, na_values=['?'])\nprint(df.shape)` : `import io\nimport pandas as pd\n\nCSV = """\\\n${csv.replace(/\\/g, '\\\\').replace(/"""/g, '\\"\\"\\"')}"""\ndf = pd.read_csv(io.StringIO(CSV), na_values=['?'])\nprint(df.shape)\ndf.head()` });
    if (m.ctor) {
      const prep = [
        'import warnings',
        "warnings.filterwarnings('ignore')",
        'import numpy as np',
        'from sklearn.compose import ColumnTransformer',
        'from sklearn.pipeline import make_pipeline',
        'from sklearn.impute import SimpleImputer',
        'from sklearn.preprocessing import OneHotEncoder, MinMaxScaler' + (m.discretize ? ', KBinsDiscretizer, OrdinalEncoder' : ''),
        m.imp,
        '',
      ];
      if (kind === 'classifier' && cls) prep.push(`X = df.drop(columns=[${pyStr(cls)}])`, `y = df[${pyStr(cls)}]`);
      else prep.push(cls ? `X = df.drop(columns=[${pyStr(cls)}])` : 'X = df.copy()');
      prep.push("num_cols = X.select_dtypes(include='number').columns.tolist()", "cat_cols = [c for c in X.columns if c not in num_cols]");
      if (m.discretize) prep.push("prep = ColumnTransformer([", "    ('num', make_pipeline(SimpleImputer(strategy='mean'), KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')), num_cols),", "    ('cat', make_pipeline(SimpleImputer(strategy='most_frequent'), OrdinalEncoder()), cat_cols)])");
      else prep.push("prep = ColumnTransformer([", `    ('num', make_pipeline(SimpleImputer(strategy='mean')${m.scale ? ', MinMaxScaler()' : ''}), num_cols),`, "    ('cat', make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore')), cat_cols)])");
      prep.push(`model = make_pipeline(prep, ${m.ctor})`, 'model');
      cells.push({ md: '## Part 1 — scikit-learn' }, { code: prep.join('\n') });
      if (m.cluster) {
        cells.push({ code: [
          'labels = model.fit_predict(X)' + (spec.name === 'EM' ? '' : ''),
          'counts = pd.Series(labels).value_counts().sort_index()',
          "print('Cluster sizes:')",
          'for k, n in counts.items():',
          "    print(f'  cluster {k}: {n} ({n / len(labels):.0%})')",
          ...(cls ? [`print()`, `print(pd.crosstab(labels, df[${pyStr(cls)}], rownames=['cluster']))`] : []),
        ].join('\n') });
      } else if (nominal) {
        const folds = src.folds || 10, seed = src.seed ?? 1;
        cells.push({ code: [
          'from sklearn.model_selection import StratifiedKFold, cross_val_predict',
          'from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix',
          '',
          `cv = StratifiedKFold(n_splits=${folds}, shuffle=True, random_state=${seed})`,
          'pred = cross_val_predict(model, X, y, cv=cv)',
          `print(f'Accuracy: {accuracy_score(y, pred):.2%}   Kappa: {cohen_kappa_score(y, pred):.4f}')`,
          'print(classification_report(y, pred, digits=3))',
          'labels = sorted(y.unique())',
          "print(pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=labels, columns=labels))",
        ].join('\n') });
      } else if (cls) {
        cells.push({ code: [
          'from sklearn.model_selection import KFold, cross_val_predict',
          'from sklearn.metrics import mean_absolute_error, mean_squared_error',
          '',
          `cv = KFold(n_splits=${src.folds || 10}, shuffle=True, random_state=${src.seed ?? 1})`,
          'pred = cross_val_predict(model, X, y, cv=cv)',
          "print(f'Correlation: {np.corrcoef(y, pred)[0, 1]:.4f}')",
          "print(f'MAE: {mean_absolute_error(y, pred):.4f}   RMSE: {mean_squared_error(y, pred) ** 0.5:.4f}')",
        ].join('\n') });
      }
    }
    // exact Weka through python-weka-wrapper3
    const arff = W.toARFF(data);
    const opts = optStr ? optStr.match(/"[^"]*"|\S+/g).map((x) => x.replace(/^"|"$/g, '')) : [];
    const loadData = arff.length > 3e6 ? `data = loader.load_file(${pyStr(data.relation + '.arff')})  # upload the ARFF file first` : `with open('data.arff', 'w') as f:\n    f.write(ARFF)\ndata = loader.load_file('data.arff')`;
    cells.push({ md: '## Part 2 — exact Weka (python-weka-wrapper3)\n\nInstalls Java and Weka in Colab (about 2 minutes the first time). The results are identical to Weka Web and to the Weka Explorer.' });
    cells.push({ code: '!apt-get -qq install -y openjdk-17-jdk-headless > /dev/null\n!pip -q install python-weka-wrapper3' });
    const evalLines = kind === 'clusterer'
      ? ['from weka.clusterers import Clusterer, ClusterEvaluation', `clu = Clusterer(classname=${pyStr(spec.path)}, options=${JSON.stringify(opts)})`, ...(cls ? [`data.delete_attribute(${data.classIndex})`] : []), 'clu.build_clusterer(data)', 'evl = ClusterEvaluation()', 'evl.set_model(clu)', 'evl.test_model(data)', 'print(clu)', 'print(evl.cluster_results)']
      : ['from weka.classifiers import Classifier, Evaluation', 'from weka.core.classes import Random', `data.class_index = ${data.classIndex >= 0 ? data.classIndex : 'data.num_attributes - 1'}`, `cls = Classifier(classname=${pyStr(spec.path)}, options=${JSON.stringify(opts)})`, 'evl = Evaluation(data)', `evl.crossvalidate_model(cls, data, ${src.folds || 10}, Random(${src.seed ?? 1}))`, 'print(evl.summary())', ...(nominal ? ['print(evl.class_details())', 'print(evl.matrix())'] : [])];
    cells.push({ code: [
      ...(arff.length > 3e6 ? [] : [`ARFF = """\\\n${arff.replace(/\\/g, '\\\\').replace(/"""/g, '\\"\\"\\"')}"""`, '']),
      'import weka.core.jvm as jvm',
      'from weka.core.converters import Loader',
      'jvm.start(packages=True)',
      'loader = Loader(classname="weka.core.converters.ArffLoader")',
      loadData,
      ...evalLines,
    ].join('\n') });
    const title = `${spec.name}-${data.relation}`.replace(/[^\w.-]+/g, '_');
    return { title, cells, ipynb: notebook(cells, title + '.ipynb'), py: script(cells) };
  }

  // ---------------------------------------------------------------- Python → Weka Web
  function parseArgs(s) {
    const out = { _pos: [] };
    let depth = 0, cur = '', q = null;
    const parts = [];
    for (const ch of s) {
      if (q) { cur += ch; if (ch === q) q = null; continue; }
      if (ch === '"' || ch === "'") { q = ch; cur += ch; continue; }
      if ('([{'.includes(ch)) depth++; if (')]}'.includes(ch)) depth--;
      if (ch === ',' && depth === 0) { parts.push(cur); cur = ''; } else cur += ch;
    }
    if (cur.trim()) parts.push(cur);
    for (const p of parts) {
      const m = p.match(/^\s*(\w+)\s*=\s*([\s\S]+?)\s*$/);
      const lit = (v) => { v = v.trim(); if (/^['"]/.test(v)) return v.slice(1, -1); if (v === 'True') return true; if (v === 'False') return false; if (v === 'None') return null; const n = Number(v); return Number.isNaN(n) ? v : n; };
      if (m) out[m[1]] = lit(m[2]); else out._pos.push(lit(p));
    }
    return out;
  }
  const FROM_SK = [
    ['DecisionTreeClassifier', (a) => (a.max_depth === 1 ? ['DecisionStump', {}] : ['J48', a.min_samples_leaf ? { M: a.min_samples_leaf } : {}])],
    ['DecisionTreeRegressor', (a) => (a.max_depth === 1 ? ['DecisionStump', {}] : ['REPTree', {}])],
    ['ExtraTreeClassifier', () => ['RandomTree', {}]],
    ['RandomForestClassifier', (a) => ['RandomForest', Object.assign(a.n_estimators ? { I: a.n_estimators } : {}, typeof a.random_state === 'number' ? { S: a.random_state } : {})]],
    ['RandomForestRegressor', (a) => ['RandomForest', a.n_estimators ? { I: a.n_estimators } : {}]],
    ['GaussianNB', () => ['NaiveBayes', {}]], ['MultinomialNB', () => ['NaiveBayesMultinomial', {}]], ['CategoricalNB', () => ['BayesNet', {}]], ['BernoulliNB', () => ['NaiveBayes', {}]],
    ['KNeighborsClassifier', (a) => ['IBk', { K: a.n_neighbors || a._pos[0] || 5 }]], ['KNeighborsRegressor', (a) => ['IBk', { K: a.n_neighbors || a._pos[0] || 5 }]],
    ['LogisticRegression', () => ['Logistic', {}]],
    ['LinearSVC', (a) => ['SMO', a.C ? { C: a.C } : {}]],
    ['SVC', (a) => ['SMO', Object.assign(a.C ? { C: a.C } : {}, a.kernel === 'rbf' || a.kernel === undefined ? { kernel: 'rbf', G: typeof a.gamma === 'number' ? a.gamma : 0.01 } : a.kernel === 'poly' ? { E: a.degree || 3 } : {})]],
    ['SGDClassifier', (a) => ['SGD', Object.assign({ F: /log/.test(a.loss || '') ? 1 : 0 }, a.alpha ? { R: a.alpha } : {}, a.max_iter ? { E: a.max_iter } : {})]],
    ['SGDRegressor', (a) => ['SGD', Object.assign({ F: 2 }, a.alpha ? { R: a.alpha } : {})]],
    ['Perceptron', () => ['VotedPerceptron', {}]],
    ['MLPClassifier', (a) => ['MultilayerPerceptron', Object.assign(a.max_iter ? { N: a.max_iter } : {}, a.learning_rate_init ? { L: a.learning_rate_init } : {})]], ['MLPRegressor', () => ['MultilayerPerceptron', {}]],
    ['LinearRegression', () => ['LinearRegression', {}]], ['Ridge', () => ['LinearRegression', {}]],
    ['AdaBoostClassifier', (a) => ['AdaBoostM1', a.n_estimators ? { I: a.n_estimators } : {}]],
    ['GradientBoostingClassifier', (a) => ['LogitBoost', Object.assign(a.n_estimators ? { I: a.n_estimators } : {}, a.learning_rate ? { H: a.learning_rate } : {})]],
    ['BaggingClassifier', (a) => ['Bagging', a.n_estimators ? { I: a.n_estimators } : {}]], ['BaggingRegressor', (a) => ['Bagging', a.n_estimators ? { I: a.n_estimators } : {}]],
    ['VotingClassifier', () => ['Vote', {}]],
    ['DummyClassifier', () => ['ZeroR', {}]], ['DummyRegressor', () => ['ZeroR', {}]],
    ['KMeans', (a) => ['SimpleKMeans', { N: a.n_clusters || a._pos[0] || 8 }]], ['MiniBatchKMeans', (a) => ['SimpleKMeans', { N: a.n_clusters || 8 }]],
    ['GaussianMixture', (a) => ['EM', { N: a.n_components || a._pos[0] || 1 }]],
    ['AgglomerativeClustering', (a) => ['HierarchicalClusterer', { N: a.n_clusters || 2 }]],
  ];
  /** Finds ML algorithms and datasets in Python code. */
  function detect(code) {
    const found = [];
    const strip = code.replace(/#.*$/gm, '');
    for (const [cls, fn] of FROM_SK) {
      const re = new RegExp('\\b' + cls + '\\s*\\(', 'g');
      let m;
      while ((m = re.exec(strip))) {
        let depth = 1, i = re.lastIndex;
        while (i < strip.length && depth) { if (strip[i] === '(') depth++; else if (strip[i] === ')') depth--; i++; }
        const args = parseArgs(strip.slice(re.lastIndex, i - 1));
        const [name, opts] = fn(args);
        const spec = W.findClassifier(name) || W.findClusterer(name);
        if (spec) found.push({ source: cls, name, opts, kind: W.findClassifier(name) ? 'classifier' : 'clusterer', exact: false });
      }
    }
    // python-weka-wrapper3: Classifier(classname="weka.classifiers.trees.J48", options=[...])
    const wre = /\b(Classifier|Clusterer)\s*\(\s*classname\s*=\s*["']([\w.]+)["']\s*(?:,\s*options\s*=\s*(\[[^\]]*\]|["'][^"']*["']))?/g;
    let m;
    while ((m = wre.exec(strip))) {
      const path = m[2];
      const spec = W.classifiers.find((s) => s.path === path) || W.clusterers.find((s) => s.path === path);
      if (!spec) continue;
      let toks = [];
      if (m[3]) toks = m[3].startsWith('[') ? (m[3].match(/["']([^"']*)["']/g) || []).map((x) => x.slice(1, -1)) : m[3].slice(1, -1).split(/\s+/).filter(Boolean);
      found.push({ source: 'python-weka-wrapper3', name: spec.name, opts: optionsFromFlags(spec, toks), kind: m[1] === 'Clusterer' ? 'clusterer' : 'classifier', exact: true });
    }
    // datasets
    let dataset = null;
    if (/load_iris\s*\(/.test(strip)) dataset = 'iris';
    else if (/load_diabetes\s*\(/.test(strip)) dataset = null;
    else {
      const ids = (W.DATASET_INDEX || []).map((x) => x.id).sort((a, b) => b.length - a.length);
      const strs = strip.match(/["'][^"'\n]{2,200}["']/g) || [];
      for (const s of strs) { const low = s.toLowerCase(); const id = ids.find((x) => low.includes(x.toLowerCase())); if (id) { dataset = id; break; } }
    }
    const planning = /ortools|cp_model|pyro|torch/.test(strip);
    return { algorithms: found, dataset, planning, ml: found.length > 0 || /\bsklearn\b|\bweka\./.test(strip) };
  }
  function optionsFromFlags(spec, toks) {
    const o = W.defaultOptions(spec);
    for (let i = 0; i < toks.length; i++) {
      const t = toks[i];
      if (!t.startsWith('-')) continue;
      const opt = (spec.options || []).find((x) => x.flag && '-' + x.flag === t);
      if (!opt) continue;
      if (opt.type === 'bool') o[opt.key] = true;
      else { const v = toks[++i]; o[opt.key] = opt.type === 'int' || opt.type === 'number' || opt.numeric ? Number(v) : v; }
    }
    return o;
  }

  // ---------------------------------------------------------------- planner → Python
  /** tasks: [{name, d, pred:[names], sd}], workers: 0 = unlimited, 1 = one worker (NoOverlap), k = k workers (Cumulative). */
  function plannerCode({ tasks, workers = 1, samples = 2000, deadline = null }) {
    const ident = (s) => pyStr(s);
    const durs = '{' + tasks.map((t) => `${ident(t.name)}: ${Math.round(t.d)}`).join(', ') + '}';
    const prec = '[' + tasks.flatMap((t) => (t.pred || []).map((p) => `(${ident(p)}, ${ident(t.name)})`)).join(', ') + ']';
    const lines = [
      'from ortools.sat.python import cp_model',
      '',
      '# Tasks and their durations',
      `durations = ${durs}`,
      '# Precedences: (task that must finish first, task that starts after it)',
      `precedences = ${prec}`,
      ...(workers > 1 ? [`WORKERS = ${workers}  # tasks that can run at the same time`] : []),
      '',
      'model = cp_model.CpModel()',
      'horizon = sum(durations.values())',
      "start = {t: model.NewIntVar(0, horizon, f'start_{t}') for t in durations}",
      "end = {t: model.NewIntVar(0, horizon, f'end_{t}') for t in durations}",
      "intervals = {t: model.NewIntervalVar(start[t], durations[t], end[t], f'interval_{t}') for t in durations}",
      '',
      'for before, after in precedences:',
      '    model.Add(end[before] <= start[after])',
      ...(workers === 1 ? ['', '# Only one worker: no two tasks may overlap', 'model.AddNoOverlap(list(intervals.values()))'] : workers > 1 ? ['', '# At most WORKERS tasks at the same time', 'model.AddCumulative(list(intervals.values()), [1] * len(durations), WORKERS)'] : []),
      '',
      '# Objective: finish everything as early as possible (minimise the makespan)',
      "makespan = model.NewIntVar(0, horizon, 'makespan')",
      'model.AddMaxEquality(makespan, [end[t] for t in durations])',
      'model.Minimize(makespan)',
      '',
      'solver = cp_model.CpSolver()',
      'status = solver.Solve(model)',
      '',
      'if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):',
      "    print('Status:', solver.StatusName(status))",
      '    for t in sorted(durations, key=lambda t: solver.Value(start[t])):',
      "        print(f'{t}: start = {solver.Value(start[t])}, end = {solver.Value(end[t])}')",
      "    print('Makespan:', solver.Value(makespan))",
      'else:',
      "    print('No solution found.')",
    ];
    const gantt = [
      'import matplotlib.pyplot as plt',
      '',
      'order = sorted(durations, key=lambda t: solver.Value(start[t]))',
      'fig, ax = plt.subplots(figsize=(8, 0.6 * len(order) + 1))',
      'ax.barh(order, [durations[t] for t in order], left=[solver.Value(start[t]) for t in order])',
      'ax.invert_yaxis()',
      "ax.set_xlabel('time')",
      "ax.set_title(f'Optimal schedule — makespan {solver.Value(makespan)}')",
      'plt.show()',
    ];
    const cells = [{ md: '# Automated planning with OR-Tools CP-SAT\n\nGenerated by Weka Web. It runs unchanged in Google Colab and in Weka Web (Python / Colab page).' }, { code: '!pip -q install ortools' }, { code: lines.join('\n') }, { code: gantt.join('\n') }];
    if (tasks.some((t) => t.sd > 0)) {
      const sds = '{' + tasks.map((t) => `${ident(t.name)}: ${+(t.sd || 0)}`).join(', ') + '}';
      const mc = [
        'import pyro',
        'import pyro.distributions as dist',
        'import torch',
        '',
        '# Uncertain durations: Normal(mean = planned duration, sd)',
        `sd = ${sds}`,
        'preds = {t: [a for a, b in precedences if b == t] for t in durations}',
        'order = sorted(durations, key=lambda t: solver.Value(start[t]))',
        '',
        'def simulate():',
        '    finish = {}',
        '    clock = torch.tensor(0.0)',
        '    for t in order:',
        '        if sd[t] > 0:',
        "            d = pyro.sample(f'duration_{t}', dist.Normal(float(durations[t]), sd[t]))",
        '        else:',
        '            d = torch.tensor(float(durations[t]))',
        '        ready = max([finish[p] for p in preds[t]] + [torch.tensor(0.0)])',
        ...(workers === 1 ? ['        begin = max(ready, clock)  # one worker: wait for the previous task'] : ['        begin = ready']),
        '        finish[t] = begin + d',
        '        clock = finish[t]',
        '    return max(finish.values())',
        '',
        `runs = torch.stack([simulate() for _ in range(${samples})])`,
        "print(f'Planned makespan:  {solver.Value(makespan)}')",
        "print(f'Expected makespan: {runs.mean().item():.3f}  (sd {runs.std().item():.3f})')",
        ...(deadline ? [`print(f'P(finish by ${deadline}) = {(runs <= ${deadline}).float().mean().item():.1%}')`] : []),
      ];
      cells.push({ md: '## Uncertain durations — Monte Carlo with Pyro\n\nKeeps the optimal order and samples the durations to estimate the expected completion time.' }, { code: '!pip -q install pyro-ppl' }, { code: mc.join('\n') });
    }
    return { cells, ipynb: notebook(cells, 'automated-planning.ipynb'), py: script(cells) };
  }

  W.PyBridge = { toNotebook, detect, parseNotebook, notebook, script, plannerCode, optionsFromFlags, SK };
})();
