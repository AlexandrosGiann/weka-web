/* Weka Web — AI assistant.
 * A floating chat that understands English and Greek and performs actions in the app (load data, run algorithms,
 * apply filters, compare models, export the report, import web data…).
 * Pipeline: fast bilingual intent rules → curated knowledge base / live data answers → optional local AI models:
 *   - Needle 2 (models/needle2): free-form commands → ONE tool call chosen from a closed list (finite grammar),
 *     always shown to the user for confirmation before it runs;
 *   - Whistle (models/whistle): speech-to-text for voice commands (en, de, fr, es, it, nl, pl);
 *   - MobileCLIP-S0 (models/mobileclip): image recognition and “images → dataset” embeddings.
 * Everything runs in the browser; nothing is sent to a server except optional Wikipedia look-ups. */
(function () {
  'use strict';
  const W = window.WEKA, U = W.Utils, App = window.App, h = App.h;
  const BASE = 'models/';
  const isHttp = /^https?:$/.test(location.protocol);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const frame = () => new Promise((r) => setTimeout(r, 30)); // not rAF: it pauses in background tabs

  const S = {
    open: false, view: 'chat', msgs: [], busy: false,
    set: Object.assign({ ai: false, wiki: true, voiceLang: 'en' }, App.store.get('assistant', {})),
  };
  const saveSet = () => App.store.set('assistant', S.set);
  let lang = I18N.lang; // reply language of the current turn
  const T = (en, el) => (lang === 'el' ? el : en);
  const TU = (en, el) => (I18N.lang === 'el' ? el : en);

  // ----------------------------------------------------------------- text utils
  const nz = (s) => String(s).toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/ς/g, 'σ')
    .replace(/[“”«»"'’`]/g, ' ').replace(/[^\p{L}\p{N}.\-+*%=/:_ ]+/gu, ' ').replace(/\s+/g, ' ').trim();
  const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const hasWord = (t, w) => new RegExp('(^|[\\s/(])' + esc(w) + '(?=$|[\\s.,;:!?)/])').test(t);
  const any = (t, list) => list.some((w) => (w.startsWith('=') ? hasWord(t, w.slice(1)) : t.includes(w)));

  // ----------------------------------------------------------------- vocabularies
  const PAGES = {
    home: ['=home', 'αρχικη', 'start page'], preprocess: ['preprocess', 'προεπεξεργασ', '=filters', 'φιλτρα'],
    table: ['data table', 'table view', 'πινακα δεδομενων', 'πινακασ δεδομενων', '=table', '=πινακα'],
    visualize: ['visuali', 'scatter', 'plot', 'γραφημα', 'οπτικοποι'], web: ['web data', 'scrap', 'web import', 'διαδικτυ', '=web'],
    classify: ['classif', 'ταξινομ'], cluster: ['cluster', 'ομαδοποι', 'ομαδεσ'], associate: ['associat', 'συσχετισ', 'κανονεσ συσχ'],
    select: ['attribute selection', 'feature selection', 'select attributes', 'επιλογη χαρακτηρ'], report: ['report', 'αναφορα', 'αναφορ'],
    tutorials: ['tutorial', 'video', 'βιντεο'], python: ['python', 'colab', 'jupyter', 'notebook', 'σημειωματαρι'],
  };
  const PAGE_NAMES = { home: ['Home', 'Αρχική'], preprocess: ['Preprocess', 'Προεπεξεργασία'], table: ['Data table', 'Πίνακας δεδομένων'], visualize: ['Visualize', 'Οπτικοποίηση'], web: ['Web data', 'Δεδομένα web'], classify: ['Classify', 'Ταξινόμηση'], cluster: ['Cluster', 'Ομαδοποίηση'], associate: ['Associate', 'Συσχετίσεις'], select: ['Select attributes', 'Επιλογή χαρακτηριστικών'], report: ['Report', 'Αναφορά'], tutorials: ['Video tutorials', 'Video tutorials'], python: ['Python / Colab', 'Python / Colab'] };
  const CLS_ALIASES = [
    ['random forest', 'RandomForest'], ['τυχαιο δασοσ', 'RandomForest'], ['τυχαια δαση', 'RandomForest'], ['random tree', 'RandomTree'], ['rep tree', 'REPTree'],
    ['decision stump', 'DecisionStump'], ['stump', 'DecisionStump'], ['decision tree', 'J48'], ['δεντρο αποφασ', 'J48'], ['δεντρο', 'J48'], ['c4.5', 'J48'], ['=tree', 'J48'],
    ['bayes net', 'BayesNet'], ['bayesian network', 'BayesNet'], ['μπεϋζιανο δικτυο', 'BayesNet'], ['multinomial', 'NaiveBayesMultinomial'], ['naive bayes', 'NaiveBayes'], ['=bayes', 'NaiveBayes'],
    ['nearest neighbo', 'IBk'], ['=knn', 'IBk'], ['=k-nn', 'IBk'], ['πλησιεστερ', 'IBk'], ['κοντινοτερ', 'IBk'], ['=k*', 'KStar'], ['k star', 'KStar'],
    ['support vector', 'SMO'], ['=svm', 'SMO'], ['μηχανη διανυσματ', 'SMO'], ['logistic regression', 'Logistic'], ['λογιστικη', 'Logistic'], ['=logistic', 'Logistic'],
    ['simple linear regression', 'SimpleLinearRegression'], ['linear regression', 'LinearRegression'], ['γραμμικη παλινδρομ', 'LinearRegression'],
    ['voted perceptron', 'VotedPerceptron'], ['=perceptron', 'VotedPerceptron'], ['neural', 'MultilayerPerceptron'], ['νευρωνικ', 'MultilayerPerceptron'], ['=mlp', 'MultilayerPerceptron'], ['multilayer', 'MultilayerPerceptron'],
    ['stochastic gradient', 'SGD'], ['gradient descent', 'SGD'], ['logit boost', 'LogitBoost'], ['=boosting', 'AdaBoostM1'], ['=adaboost', 'AdaBoostM1'],
    ['=ripper', 'JRip'], ['=one r', 'OneR'], ['=1r', 'OneR'], ['=zero r', 'ZeroR'], ['=baseline', 'ZeroR'],
  ];
  const CLU_ALIASES = [['k-means', 'SimpleKMeans'], ['kmeans', 'SimpleKMeans'], ['k means', 'SimpleKMeans'], ['κ-μεσ', 'SimpleKMeans'], ['k-μεσ', 'SimpleKMeans'], ['=em', 'EM'], ['expectation', 'EM'],
    ['hierarch', 'HierarchicalClusterer'], ['ιεραρχ', 'HierarchicalClusterer'], ['dendrogram', 'HierarchicalClusterer'], ['δενδρογρ', 'HierarchicalClusterer'], ['farthest', 'FarthestFirst']];
  const DS_ALIASES = [['ιριδα', 'iris'], ['ιρισ', 'iris'], ['διαβητ', 'diabetes'], ['τραπεζ', 'bank'], ['bank marketing', 'bank'], ['καιρο', 'weather.nominal'], ['weather numeric', 'weather.numeric'],
    ['=weather', 'weather.nominal'], ['σογια', 'soybean'], ['γυαλι', 'glass'], ['ψηφοφορ', 'vote'], ['καρκιν', 'breast-cancer'], ['breast cancer', 'breast-cancer'], ['φακοι', 'contact-lenses'], ['contact lenses', 'contact-lenses'],
    ['σουπερ μαρκετ', 'supermarket'], ['σουπερμαρκετ', 'supermarket'], ['credit', 'credit-g'], ['πιστ', 'credit-g'], ['ιονοσφ', 'ionosphere'], ['υποθυρεο', 'hypothyroid'], ['segment', 'segment-challenge'], ['labor', 'labor'], ['airline', 'airline'], ['αεροπορ', 'airline']];
  const FILTERS = [
    [['normaliz', 'normalis', 'κανονικοποι'], 'Normalize'], [['standardiz', 'standardis', 'τυποποι'], 'Standardize'],
    [['replace missing', 'fill missing', 'impute', 'συμπληρωσ', 'αντικαταστ'], 'ReplaceMissingValues'],
    [['discretiz', 'discretis', 'διακριτοποι'], 'Discretize'], [['nominal to binary', 'one-hot', 'one hot', 'dummy'], 'NominalToBinary'],
    [['numeric to nominal'], 'NumericToNominal'], [['randomiz', 'randomis', 'shuffle', 'ανακατε'], 'Randomize'], [['remove duplicate', 'διπλοτυπ', 'διπλεσ εγγραφ'], 'RemoveDuplicates'],
    [['class balanc', 'balance the class', 'εξισορροπ'], 'ClassBalancer'], [['remove useless', 'αχρηστ'], 'RemoveUseless'], [['resample'], 'Resample'],
  ];
  const TUTORIALS = [['getting-started', ['getting started', 'ξεκινω', 'αρχαριο', 'εισαγωγ', 'basics']], ['preprocess', ['preprocess', 'filter', 'προεπεξεργ', 'φιλτρ']], ['classify', ['classif', 'ταξινομ']],
    ['cluster', ['cluster', 'ομαδοποι']], ['associate-select', ['associat', 'rules', 'κανον', 'attribute selection', 'επιλογη']], ['report', ['report', 'pdf', 'αναφορ']], ['bank', ['bank', 'τραπεζ', 'case study']]];

  const VERB_LOAD = ['load', 'open', 'use the', 'φορτωσ', 'ανοιξ', 'ανοιγ'];
  const VERB_RUN = ['run', 'train', 'build', 'fit', 'evaluate', 'try', 'apply', 'use', 'τρεξ', 'εκπαιδευσ', 'εφαρμοσ', 'δοκιμασ', 'χρησιμοποιησ', 'φτιαξ'];
  const VERB_GO = ['go to', 'open', 'show', 'take me', 'navigate', 'switch to', 'πηγαινε', 'παμε', 'ανοιξ', 'δειξ', 'δειχνε'];
  const QUESTION = /^(what|whats|what s|why|how|explain|define|tell me|describe|meaning|which is|who|is it|does|can you explain|τι |γιατι|πωσ|εξηγησ|ορισε|πεσ μου|τι σημαινει|ποιο ειναι|ποια ειναι|ποιοσ|σημαινει)/;

  // ----------------------------------------------------------------- entity finders
  const classifierNames = () => W.classifiers.map((s) => s.name);
  function findAlias(t, list) {
    let best = null;
    for (const [a, name] of list) {
      const hit = a.startsWith('=') ? hasWord(t, a.slice(1)) : t.includes(a);
      if (hit && (!best || a.length > best[0].length)) best = [a, name];
    }
    return best;
  }
  function findClassifiers(t) {
    const out = [];
    for (const n of classifierNames()) { const k = n.toLowerCase(); if (hasWord(t, k) || (k.length > 4 && t.includes(k))) out.push({ name: n, at: t.indexOf(k) }); }
    for (const [a, n] of CLS_ALIASES) {
      const w = a.replace(/^=/, '');
      if ((a.startsWith('=') ? hasWord(t, w) : t.includes(w)) && !out.some((o) => o.name === n)) {
        if (n === 'J48' && w === 'tree' && /random tree|rep tree|tree diagram/.test(t)) continue;
        if (n === 'NaiveBayes' && out.some((o) => /Bayes/.test(o.name))) continue;
        out.push({ name: n, at: t.indexOf(w) });
      }
    }
    return out.sort((a, b) => a.at - b.at).map((o) => o.name).filter((n) => W.findClassifier(n));
  }
  function findClusterer(t) {
    for (const s of W.clusterers) if (hasWord(t, s.name.toLowerCase())) return s.name;
    const a = findAlias(t, CLU_ALIASES); return a ? a[1] : null;
  }
  function findDataset(t) {
    const ids = (W.DATASET_INDEX || []).map((x) => x.id).sort((a, b) => b.length - a.length);
    for (const id of ids) if (hasWord(t, id) || t.includes(id + '.arff')) return id;
    const a = findAlias(t, DS_ALIASES); return a ? a[1] : null;
  }
  function findPage(t) {
    let best = null;
    for (const [p, keys] of Object.entries(PAGES)) for (const k of keys) {
      const w = k.replace(/^=/, '');
      if ((k.startsWith('=') ? hasWord(t, w) : t.includes(w)) && (!best || w.length > best[1])) best = [p, w.length];
    }
    return best ? best[0] : null;
  }
  function findAttributes(t) {
    const d = App.state.data; if (!d) return [];
    const out = [];
    d.attributes.map((a, j) => ({ n: nz(a.name), j })).sort((a, b) => b.n.length - a.n.length).forEach(({ n, j }) => {
      if (n && hasWord(t, n) && !out.some((o) => d.attributes[o].name.toLowerCase().includes(n))) out.push(j);
    });
    return out;
  }
  const num = (t, re) => { const m = t.match(re); return m ? Number(m[1]) : null; };

  // ----------------------------------------------------------------- intent parsing
  /** Returns a list of steps: {type, ...args}. */
  function parse(raw) {
    const t = nz(raw);
    if (!t) return [];
    const urlM = raw.match(/https?:\/\/[^\s"'<>]+/i);
    if (urlM) return [{ type: 'web', url: urlM[0] }];
    const full = parseOne(t);
    if (['plan', 'planExample', 'pyExport', 'pyImport'].includes(full.type)) return [full];
    // split compound commands ("load iris and then run J48")
    const parts = t.split(/\s*(?:;|,?\s+(?:and then|then|after that|και μετα|και επειτα|επειτα|μετα)\s+)\s*/).filter(Boolean);
    let steps = [];
    for (const p of parts) {
      const whole = parseOne(p);
      if (whole.type === 'compare' || whole.type === 'concept' || whole.type === 'wikiTable') { steps.push(whole); continue; }
      const sub = p.split(/\s+(?:and|και)\s+/);
      if (sub.length > 1) {
        const parsed = sub.map((s) => parseOne(s));
        if (parsed.every((x) => x && x.type !== 'unknown' && x.type !== 'concept')) { steps = steps.concat(parsed); continue; }
      }
      steps.push(parseOne(p));
    }
    return steps;
  }
  function parseOne(t) {
    const question = QUESTION.test(t) || /\?$/.test(t);
    if (/^(hi|hello|hey|good (morning|evening)|γεια|γειασ|καλησπερα|καλημερα|χαιρετ)\b/.test(t) && t.split(' ').length <= 4) return { type: 'greet' };
    if (/^(thanks|thank you|thx|ευχαριστ|σ ευχαριστ)/.test(t)) return { type: 'thanks' };
    if (/^(help|βοηθεια|τι μπορεισ|what can you|commands|εντολεσ)/.test(t) || /what can you do|τι μπορεισ να κανεισ/.test(t)) return { type: 'help' };
    if (any(t, ['=undo', 'αναιρεσ', 'αναιρεσε'])) return { type: 'undo' };
    if (any(t, ['dark mode', 'dark theme', 'σκοτειν', '=dark'])) return { type: 'theme', mode: 'dark' };
    if (any(t, ['light mode', 'light theme', 'φωτειν', '=light'])) return { type: 'theme', mode: 'light' };
    if (any(t, ['in greek', 'greek language', 'στα ελληνικα', 'ελληνικα'])) return { type: 'lang', lang: 'el' };
    if (any(t, ['in english', 'english language', 'στα αγγλικα', 'αγγλικα'])) return { type: 'lang', lang: 'en' };
    if (/wikipedia/.test(t) && any(t, ['table', 'πινακ', 'import', 'εισαγ', 'data', 'δεδομεν', 'load', 'φορτωσ'])) return { type: 'wikiTable', title: t.replace(/.*wikipedia\s*(table|πινακα|πινακεσ)?\s*(of|for|about|απο|για|του|τησ)?\s*/, '').trim() };
    if (any(t, ['image', 'photo', 'picture', 'εικον', 'φωτογραφ'])) {
      if (any(t, ['dataset', 'δεδομεν', 'embedding', 'folder', 'φακελ', 'many', 'πολλεσ'])) return { type: 'imagesToData' };
      return { type: 'image' };
    }
    if (any(t, ['voice', 'microphone', 'speak', 'φωνη', 'μικροφων'])) return { type: 'voiceHelp' };

    // questions about the data or results
    if (any(t, ['how many', 'ποσεσ', 'ποσα', 'ποσοι'])) {
      if (any(t, ['missing', 'ελλιπ', 'κενεσ'])) return { type: 'missing' };
      return { type: 'describe' };
    }
    if (any(t, ['missing value', 'ελλιπεισ τιμ']) && !question) return { type: 'missing' };
    if (any(t, ['describe', 'summary', 'summarize', 'overview', 'tell me about the data', 'περιγραψ', 'συνοψ', 'περιληψ', 'about this data', 'για τα δεδομενα'])) return { type: 'describe' };
    if (any(t, ['best model', 'best classifier', 'which model', 'καλυτερο μοντελο', 'καλυτεροσ ταξινομ', 'ποιο μοντελο', 'ποιοσ ταξινομητησ ηταν', 'winner'])) return { type: 'best' };
    if (any(t, ['explain the result', 'explain result', 'interpret', 'what does the result', 'εξηγησε το αποτελεσμα', 'ερμηνευσ', 'τι σημαινει το αποτελεσμα', 'how good is', 'ποσο καλο'])) return { type: 'explain' };
    if (any(t, ['which attributes', 'most important attribute', 'important features', 'most informative', 'ποια χαρακτηριστικα', 'σημαντικοτερα', 'πιο σημαντικ', 'matter most', 'μετρανε'])) return { type: 'select' };

    if (question) {
      const c = concept(t);
      if (c) return { type: 'concept', entry: c };
      return { type: 'unknown', question: true };
    }

    // commands
    const py = parsePython(t);
    if (py) return py;
    if (any(t, ['tutorial', 'video', 'βιντεο', 'show me how', 'δειξε μου πωσ'])) {
      const tu = TUTORIALS.find(([, keys]) => any(t, keys)); return { type: 'tutorial', id: tu ? tu[0] : 'getting-started' };
    }
    const pg0 = findPage(t);
    if (pg0 && any(t, ['go to', 'take me', 'navigate', 'switch to', 'πηγαινε', 'παμε', 'μεταβα', '=page', 'σελιδα', 'open the'])) return { type: 'nav', page: pg0 };
    if (any(t, ['pdf', 'export report', 'save report', 'print report', 'εξαγωγη', 'αποθηκευσ'])) return { type: 'export', format: any(t, ['html']) ? 'html' : 'pdf' };
    const k = num(t, /(?:k\s*=\s*|with\s+|σε\s+|με\s+)(\d+)\s*(?:clusters|ομαδεσ|groups|cluster)?/) ?? num(t, /(\d+)\s*(?:clusters|ομαδεσ|groups)/);
    const ds = findDataset(t);
    if (ds && (any(t, VERB_LOAD) || any(t, ['dataset', 'data set', 'δεδομεν', 'συνολο']) || t === ds)) return { type: 'load', id: ds };
    const attrs = findAttributes(t);
    if (any(t, ['class to', 'class is', 'set class', 'target is', 'target to', 'set the target', 'predict the', 'κλαση', 'στοχοσ', 'στοχο', 'προβλεψε το', 'προβλεψε την'] ) && attrs.length && !findClassifiers(t).length) return { type: 'setClass', attr: attrs[0] };
    if (any(t, ['remove', 'delete', 'drop', 'αφαιρεσ', 'διαγραψ', 'βγαλε']) && attrs.length) return { type: 'removeAttrs', attrs };
    for (const [keys, name] of FILTERS) if (any(t, keys)) return { type: 'filter', name, supervised: any(t, ['mdl', 'supervised', 'εποπτευομ']) };
    const cls = findClassifiers(t);
    if (any(t, ['compare', 'comparison', 'σύγκρινε', 'συγκριν', 'συγκρισ', 'versus', '=vs']) || cls.length >= 2) return { type: 'compare', names: cls };
    const clu = findClusterer(t);
    if (clu || any(t, ['cluster', 'ομαδοποι', 'segment the', 'τμηματοποι'])) return { type: 'cluster', name: clu || 'SimpleKMeans', k };
    if (any(t, ['association', 'apriori', 'fpgrowth', 'κανονεσ συσχετ', 'συσχετισεισ', 'find rules', 'βρεσ κανονεσ', 'market basket', '=rules', 'κανονεσ'])) return { type: 'rules', name: /fpgrowth/.test(t) ? 'FPGrowth' : 'Apriori' };
    if (any(t, ['attribute selection', 'feature selection', 'select attributes', 'rank attributes', 'rank the attributes', 'επιλογη χαρακτηριστικ', 'καταταξ', 'infogain', 'information gain'])) return { type: 'select' };
    if (cls.length === 1) return { type: 'classify', name: cls[0], folds: num(t, /(\d+)\s*-?\s*folds?/) };
    if (any(t, ['classify', 'classification', 'train a model', 'build a model', 'predict', 'ταξινομησ', 'ταξινομηση', 'εκπαιδευσ', 'προβλεψ'])) return { type: 'classify', name: null };
    if (any(t, ['scrape', 'web data', 'from the web', 'από το διαδικτυο', 'απο το διαδικτυο', 'απο ιστοσελιδα', 'from a website'])) return { type: 'nav', page: 'web' };
    const pg = findPage(t);
    if (pg && (any(t, VERB_GO) || t.split(' ').length <= 3)) return { type: 'nav', page: pg };
    if (ds) return { type: 'load', id: ds };
    const c = concept(t);
    if (c) return { type: 'concept', entry: c };
    return { type: 'unknown', question: false };
  }
  /** Python / Colab and automated-planning requests. */
  function parsePython(t) {
    const nbWord = any(t, ['colab', 'notebook', 'ipynb', 'jupyter', '=python', '.py', 'σημειωματαρι']);
    if (nbWord && any(t, ['export', 'convert', 'save', 'download', 'generate', 'turn', 'εξαγωγ', 'εξαγαγ', 'μετατρ', 'μετετρεψ', 'κατεβασ', 'αποθηκευσ', 'φτιαξ'])) return { type: 'pyExport', fmt: /(^|\s)(\.py|script|σκριπτ)/.test(t) && !/ipynb|notebook|colab/.test(t) ? 'py' : 'ipynb' };
    if (nbWord && !any(t, ['=page', 'σελιδα']) && any(t, ['import', 'load', 'open', 'upload', 'run my', 'εισαγ', 'φορτωσ', 'ανοιξ', 'ανεβασ', 'τρεξε το'])) return { type: 'pyImport' };
    const planWord = any(t, ['schedul', 'planning', 'plan the', 'plan my', 'or-tools', 'ortools', 'cp-sat', 'cpsat', 'gantt', 'makespan', 'προγραμματισ', 'χρονοπρογραμμ', 'σχεδιασμ', 'automated planning', 'monte carlo', '=pyro']);
    if (!planWord) return null;
    const tasks = [];
    for (const m of t.matchAll(/([\p{L}][\p{L}\d_-]*)\s*(?:=|:)\s*(\d+(?:\.\d+)?)/gu)) if (!/^(k|deadline|προθεσμια|workers?)$/.test(m[1])) tasks.push({ name: m[1], d: Math.round(+m[2]), pred: [] });
    if (tasks.length >= 2) {
      const find = (n) => tasks.find((x) => x.name === n);
      for (const m of t.matchAll(/([\p{L}][\p{L}\d_-]*)\s+(?:before|πριν(?:\s+απο)?(?:\s+το|\s+τη|\s+την)?)\s+([\p{L}][\p{L}\d_-]*)/gu)) { const a = find(m[1]), b = find(m[2]); if (a && b && !b.pred.includes(a.name)) b.pred.push(a.name); }
      for (const m of t.matchAll(/([\p{L}][\p{L}\d_-]*)\s+(?:after|μετα(?:\s+απο)?(?:\s+το|\s+τη|\s+την)?)\s+([\p{L}][\p{L}\d_-]*)/gu)) { const a = find(m[1]), b = find(m[2]); if (a && b && !a.pred.includes(b.name)) a.pred.push(b.name); }
      const wm = t.match(/(\d+|one|two|three|four|ενασ|ενα|μια|δυο|τρεισ|τρια|τεσσερισ)\s+(?:workers?|developers?|machines?|people|εργατ\S*|ατομα|μηχανεσ|προγραμματιστ\S*)/);
      const words = { one: 1, two: 2, three: 3, four: 4, 'ενασ': 1, 'ενα': 1, 'μια': 1, 'δυο': 2, 'τρεισ': 3, 'τρια': 3, 'τεσσερισ': 4 };
      const workers = wm ? (words[wm[1]] || +wm[1]) : any(t, ['parallel', 'unlimited', 'παραλληλ']) ? 0 : 1;
      const dm = t.match(/(?:deadline|προθεσμια)\s*(?:of|=|:|is|ειναι)?\s*(\d+)/);
      return { type: 'plan', tasks, workers, deadline: dm ? +dm[1] : null };
    }
    const id = any(t, ['monte carlo', 'pyro', 'uncertain', 'αβεβαι', 'πιθανοτ']) ? 'uncertain' : any(t, ['two workers', 'δυο εργατ', 'cumulative']) ? 'two-workers' : any(t, ['one worker', 'single worker', 'ενασ εργατ', 'no overlap']) ? 'one-worker' : any(t, ['gantt', 'matplotlib']) ? 'gantt' : any(t, ['precedence', 'προτεραιοτ', 'assembly', 'συναρμολογ']) ? 'precedence' : 'project';
    return { type: 'planExample', id };
  }
  function concept(t) {
    let best = null, bl = 0;
    for (const e of window.ASSIST_KB || []) for (const k of e.k) {
      const kk = nz(k);
      if (kk.length > bl && (kk.length <= 3 ? hasWord(t, kk) : t.includes(kk))) { best = e; bl = kk.length; }
    }
    return best;
  }

  // ----------------------------------------------------------------- chat rendering
  let msgBox = null, inputEl = null, panel = null, fab = null, statusEl = null;
  function push(m) { S.msgs.push(m); if (S.msgs.length > 80) S.msgs.shift(); renderMsgs(); return m; }
  function bot(text, extra = {}) { return push(Object.assign({ role: 'bot', text }, extra)); }
  function update(m, patch) { Object.assign(m, patch); renderMsgs(); }
  function renderMsgs() {
    if (!msgBox) return;
    App.clear(msgBox);
    for (const m of S.msgs) {
      const el = h('div.as-msg.' + m.role);
      if (m.img) el.appendChild(h('img.as-img', { src: m.img, alt: '' }));
      if (m.text) el.appendChild(h('div', m.text));
      if (m.node) el.appendChild(m.node);
      if (m.progress != null) el.appendChild(h('div.as-prog', h('div', { style: { width: Math.round(m.progress * 100) + '%' } })));
      if (m.src) el.appendChild(h('div.as-src', m.src));
      if (m.chips && m.chips.length) el.appendChild(h('div.as-chips', m.chips.map((c) => h('button.as-chip', { on: { click: () => (c.fn ? c.fn() : send(c.cmd)) } }, c.label))));
      msgBox.appendChild(el);
    }
    if (S.busy) msgBox.appendChild(h('div.as-msg.bot.as-typing', h('span'), h('span'), h('span')));
    msgBox.scrollTop = msgBox.scrollHeight;
  }
  const chip = (label, cmd) => ({ label, cmd: cmd || label });
  const chipFn = (label, fn) => ({ label, fn });

  // ----------------------------------------------------------------- main loop
  async function send(text) {
    text = String(text || '').trim().slice(0, 300);
    if (!text || S.busy) return;
    lang = /[Ͱ-Ͽἀ-῿]/.test(text) ? 'el' : I18N.lang === 'el' && !/[a-z]{3}/i.test(text) ? 'el' : /[a-z]/i.test(text) ? 'en' : I18N.lang;
    push({ role: 'user', text });
    if (inputEl) inputEl.value = '';
    S.busy = true; renderMsgs();
    try {
      const steps = parse(text);
      let handled = false;
      for (const st of steps) {
        if (st.type === 'unknown') continue;
        handled = true;
        const ok = await act(st);
        if (ok === false) break;
      }
      if (!handled) await fallback(text, steps[0] && steps[0].question);
    } catch (e) {
      console.error(e);
      bot('⚠ ' + (e && e.message ? e.message : String(e)));
    } finally { S.busy = false; renderMsgs(); }
  }

  function needData() {
    if (App.state.data) return true;
    bot(T('Load a dataset first. For example:', 'Φόρτωσε πρώτα ένα σύνολο δεδομένων. Για παράδειγμα:'), { chips: [chip(T('Load iris', 'Φόρτωσε το iris')), chip(T('Load bank', 'Φόρτωσε το bank')), chip(T('Load weather.nominal', 'Φόρτωσε το weather.nominal'))] });
    return false;
  }
  /** Configures an algorithm on its page and presses the page's own Run button, so results appear exactly as if the user did it. */
  async function runOnPage(page, store, key) {
    for (const [k, v] of Object.entries(store)) App.store.set(k, v);
    App.state.forceTop = true;
    App.go(page);
    await frame();
    const btn = document.querySelector('#main button.btn.primary.lg');
    if (!btn) throw new Error('Run button not found');
    const before = App.state.results[key].length;
    btn.click();
    for (let i = 0; i < 18000; i++) {
      await sleep(100);
      if (App.state.results[key].length > before) return App.state.results[key][App.state.results[key].length - 1];
      if (i > 2 && btn.isConnected && !btn.disabled) return null; // finished with an error (shown as a toast)
    }
    return null;
  }
  const fmtPct = (v) => U.d2s(v, 2) + '%';

  async function act(st) {
    const d = App.state.data;
    switch (st.type) {
      case 'greet':
        bot(T('Hi! I can load data, run algorithms, explain results and much more. What would you like to do?', 'Γεια σου! Μπορώ να φορτώσω δεδομένα, να τρέξω αλγορίθμους, να εξηγήσω αποτελέσματα και πολλά ακόμα. Τι θα ήθελες να κάνουμε;'), { chips: helpChips() });
        return true;
      case 'thanks': bot(T('You’re welcome! 😊', 'Παρακαλώ! 😊')); return true;
      case 'help': return help();
      case 'undo': App.undo(); bot(T('Undone the last change.', 'Αναίρεσα την τελευταία αλλαγή.')); return true;
      case 'theme': App.setTheme(st.mode); bot(st.mode === 'dark' ? T('Dark theme on. 🌙', 'Ενεργοποίησα το σκοτεινό θέμα. 🌙') : T('Light theme on. ☀️', 'Ενεργοποίησα το φωτεινό θέμα. ☀️')); return true;
      case 'lang':
        bot(T('Switching the interface language — the page will reload and keep your data.', 'Αλλάζω τη γλώσσα του περιβάλλοντος — η σελίδα θα ανανεωθεί και θα κρατήσει τα δεδομένα σου.'));
        await sleep(900); App.switchLang(st.lang); return false;
      case 'nav':
        if (['preprocess', 'table', 'visualize', 'classify', 'cluster', 'associate', 'select', 'report'].includes(st.page) && !needData()) return false;
        App.state.forceTop = true; App.go(st.page);
        bot(T('Opened ', 'Άνοιξα: ') + PAGE_NAMES[st.page][lang === 'el' ? 1 : 0] + '.');
        return true;
      case 'load': {
        await App.openSample(st.id);
        if (['home', 'web', 'tutorials'].includes(App.state.page)) App.go('preprocess');
        const nd = App.state.data;
        bot(T(`Loaded “${nd.relation}”: ${nd.numInstances} instances, ${nd.numAttributes} attributes.`, `Φόρτωσα το «${nd.relation}»: ${nd.numInstances} εγγραφές, ${nd.numAttributes} χαρακτηριστικά.`) + (nd.classAttr ? T(` Class: ${nd.classAttr.name}.`, ` Κλάση: ${nd.classAttr.name}.`) : ''), { chips: nextChips(nd) });
        return true;
      }
      case 'describe': if (!needData()) return false; bot(describe(d), { chips: nextChips(d) }); return true;
      case 'missing': if (!needData()) return false; bot(missingText(d)); return true;
      case 'setClass': {
        if (!needData()) return false;
        App.setClass(st.attr); App.go(App.state.page);
        bot(T(`Class set to “${d.attributes[st.attr].name}”.`, `Όρισα ως κλάση το «${d.attributes[st.attr].name}».`));
        return true;
      }
      case 'removeAttrs': {
        if (!needData()) return false;
        const idx = st.attrs.filter((j) => j !== d.classIndex);
        if (!idx.length) { bot(T('I will not remove the class attribute. Name another attribute.', 'Δεν αφαιρώ το χαρακτηριστικό της κλάσης. Ανέφερε άλλο χαρακτηριστικό.')); return false; }
        const f = new W.Remove({ indices: idx }); f.setInput(d);
        const names = idx.map((j) => d.attributes[j].name);
        App.applyChange(f.apply(d), 'Remove attributes: ' + names.join(', '));
        App.go(App.state.page);
        bot(T(`Removed: ${names.join(', ')}. (Say “undo” to bring them back.)`, `Αφαίρεσα: ${names.join(', ')}. (Πες «αναίρεση» για να επανέλθουν.)`));
        return true;
      }
      case 'filter': {
        if (!needData()) return false;
        let name = st.name;
        if (name === 'Discretize' && st.supervised) name = 'Discretize (supervised)';
        const spec = W.findFilter(name);
        const opts = W.defaultOptions(spec);
        const nd = W.applyFilter(spec, opts, d);
        App.applyChange(nd, name + ' ' + W.optionString(spec, opts));
        App.go(App.state.page === 'home' ? 'preprocess' : App.state.page);
        bot(T(`Applied ${name}: ${d.numInstances} → ${nd.numInstances} instances, ${d.numAttributes} → ${nd.numAttributes} attributes.`, `Εφάρμοσα ${name}: ${d.numInstances} → ${nd.numInstances} εγγραφές, ${d.numAttributes} → ${nd.numAttributes} χαρακτηριστικά.`), { chips: [chip(T('Undo', 'Αναίρεση'))] });
        return true;
      }
      case 'classify': return classify(st.name, st.folds);
      case 'compare': return compare(st.names);
      case 'cluster': return cluster(st.name, st.k);
      case 'rules': return rules(st.name);
      case 'select': return selectAttrs();
      case 'best': return best();
      case 'explain': return explainLast();
      case 'export': {
        if (!needData()) return false;
        App.state.forceTop = true; App.go('report');
        await frame();
        if (st.format === 'html') App.report.exportHTML(); else App.report.exportPDF();
        bot(st.format === 'html' ? T('Exported the report as HTML.', 'Εξήγαγα την αναφορά σε HTML.') : T('The print dialog is open — choose “Save as PDF”.', 'Άνοιξε το παράθυρο εκτύπωσης — διάλεξε «Αποθήκευση ως PDF».'));
        return true;
      }
      case 'tutorial': App.openTutorial(st.id); bot(T('Here is the tutorial video.', 'Ορίστε το βίντεο οδηγός.')); return true;
      case 'web':
        App.web.prefill(st.url);
        bot(T('I filled in the address on the Web data page. For safety I don’t fetch it automatically: tick the responsible-use box and press “Fetch” — robots.txt and rate limits are checked first.', 'Συμπλήρωσα τη διεύθυνση στη σελίδα «Δεδομένα web». Για ασφάλεια δεν την ανακτώ αυτόματα: τσέκαρε τη δήλωση υπεύθυνης χρήσης και πάτα «Ανάκτηση» — πρώτα ελέγχονται το robots.txt και τα όρια ρυθμού.'));
        return true;
      case 'wikiTable':
        App.web.prefillWiki(st.title || '', lang === 'el' ? 'el' : 'en');
        bot(T('Opened the Wikipedia import (official API). Check the title, tick the responsible-use box and press “Fetch tables”.', 'Άνοιξα την εισαγωγή από Wikipedia (επίσημο API). Έλεγξε τον τίτλο, τσέκαρε τη δήλωση υπεύθυνης χρήσης και πάτα «Ανάκτηση πινάκων».'));
        return true;
      case 'concept': {
        const e = st.entry;
        bot(lang === 'el' ? e.el : e.en, { chips: e.try ? [chip(lang === 'el' ? e.try[1] : e.try[0])] : [] });
        return true;
      }
      case 'image': await pickImage(); return true;
      case 'imagesToData': await imagesToDataset(); return true;
      case 'pyExport': {
        if (!App.state.data && !App.state.results.classify.length) { bot(T('Load a dataset and run a classifier first — then I can turn it into a Colab notebook.', 'Φόρτωσε δεδομένα και τρέξε πρώτα έναν ταξινομητή — μετά τον μετατρέπω σε notebook για το Colab.'), { chips: [chip(T('Load iris and run J48', 'Φόρτωσε το iris και τρέξε J48'))] }); return false; }
        const nb = App.py.exportResult('last', st.fmt);
        bot(T(`Downloaded ${nb.title}.${st.fmt}. In Google Colab choose File → Upload notebook. Part 1 runs the closest scikit-learn model; part 2 runs the real Weka algorithm (python-weka-wrapper3), so its numbers match Weka Web exactly.`, `Κατέβηκε το ${nb.title}.${st.fmt}. Στο Google Colab διάλεξε File → Upload notebook. Το μέρος 1 τρέχει το πλησιέστερο μοντέλο του scikit-learn· το μέρος 2 τρέχει τον πραγματικό αλγόριθμο του Weka (python-weka-wrapper3), άρα τα νούμερα ταιριάζουν ακριβώς με το Weka Web.`), { chips: [chipFn(T('Open Python / Colab', 'Άνοιγμα Python / Colab'), () => { App.go('python'); App.py.setTab('convert'); })] });
        return true;
      }
      case 'pyImport':
        App.go('python'); App.py.setTab('notebook');
        bot(T('Choose a .ipynb (Colab / Jupyter) or .py file. OR-Tools, Pyro, torch, numpy and matplotlib cells run in the browser; scikit-learn models are mapped to Weka Web algorithms. You can also drop the file on the page.', 'Διάλεξε ένα αρχείο .ipynb (Colab / Jupyter) ή .py. Τα κελιά με OR-Tools, Pyro, torch, numpy και matplotlib τρέχουν στον browser· τα μοντέλα του scikit-learn αντιστοιχίζονται σε αλγορίθμους του Weka Web. Μπορείς και να σύρεις το αρχείο στη σελίδα.'), { chips: [chipFn(T('📂 Choose notebook', '📂 Επιλογή notebook'), async () => { await App.py.importFile(); if (App.py.state.cells.length) { App.py.runAll(); bot(T(`Ran ${App.py.state.cells.filter((c) => c.type === 'code').length} cells — results and Gantt charts are on the page.`, `Έτρεξα ${App.py.state.cells.filter((c) => c.type === 'code').length} κελιά — αποτελέσματα και διαγράμματα Gantt είναι στη σελίδα.`)); } })] });
        return true;
      case 'planExample': {
        const ex = App.py.loadExample(st.id);
        const outs = App.py.state.cells.filter((c) => c.type === 'code' && c.out).map((c) => (c.out.error ? '⚠ ' + c.out.error.type + ': ' + c.out.error.message : c.out.text.trim())).filter(Boolean).join('\n').split('\n').slice(0, 9).join('\n');
        bot(T(`Opened the planning example “${ex.title}” and ran it in the browser:\n`, `Άνοιξα το παράδειγμα σχεδιασμού «${ex.title}» και το έτρεξα στον browser:\n`) + outs, { chips: [chip(T('Schedule design=3, build=5, test=2, design before build, build before test, one worker', 'Προγραμμάτισε design=3, build=5, test=2, design πριν build, build πριν test, ένας εργάτης')), chip(T('Planning with uncertainty (Pyro)', 'Σχεδιασμός με αβεβαιότητα (Pyro)'))] });
        return true;
      }
      case 'plan': {
        const out = App.py.plan(st.tasks, st.workers, st.deadline);
        const text = (out || []).map((c) => (c.out.error ? '⚠ ' + c.out.error.message : c.out.text)).join('\n');
        const lines = text.split('\n').filter((l) => /makespan|status|P\(|:\s*start/i.test(l)).slice(0, 12).join('\n');
        bot(T(`Planned ${st.tasks.length} tasks with OR-Tools CP-SAT (${st.workers === 1 ? 'one worker' : st.workers ? st.workers + ' workers' : 'unlimited workers'}):\n`, `Προγραμμάτισα ${st.tasks.length} εργασίες με OR-Tools CP-SAT (${st.workers === 1 ? 'ένας εργάτης' : st.workers ? st.workers + ' εργάτες' : 'χωρίς όριο εργατών'}):\n`) + lines, { chips: [chipFn(T('Download as Colab notebook', 'Λήψη ως notebook Colab'), () => { const g = W.PyBridge.plannerCode({ tasks: st.tasks, workers: st.workers, deadline: st.deadline }); App.download('automated-planning.ipynb', g.ipynb, 'application/json'); })] });
        return true;
      }
      case 'voiceHelp': bot(T('Press the 🎤 button, speak your command (English, German, French, Spanish, Italian, Dutch or Polish), then press it again. The Whistle speech model runs locally in your browser.', 'Πάτα το κουμπί 🎤, πες την εντολή σου (αγγλικά, γερμανικά, γαλλικά, ισπανικά, ιταλικά, ολλανδικά ή πολωνικά — όχι ελληνικά) και πάτα το ξανά. Το μοντέλο ομιλίας Whistle τρέχει τοπικά στον browser σου.')); return true;
      default: return true;
    }
  }

  function helpChips() {
    return [chip(T('Load bank', 'Φόρτωσε το bank')), chip(T('Describe the data', 'Περίγραψε τα δεδομένα')), chip(T('Compare models', 'Σύγκρινε μοντέλα')),
      chip(T('What is cross-validation?', 'Τι είναι το cross-validation;')), chip(T('Cluster with k-means, k=3', 'Ομαδοποίηση με k-means, k=3')), chip(T('Export PDF', 'Εξαγωγή PDF'))];
  }
  function help() {
    bot(T('Here is what I can do — just ask in plain English or Greek:\n• Data: “load diabetes”, “describe the data”, “how many missing values?”, “set class to y”, “remove duration”, “normalize”, “undo”\n• Analysis: “run random forest”, “compare J48, NaiveBayes and SMO”, “cluster with k-means k=4”, “find association rules”, “which attributes matter most?”\n• Results: “which model is best?”, “explain the result”, “export PDF”\n• Learn: “what is overfitting?”, “show me the classification video”\n• Web: paste a URL or “wikipedia table List of largest cities”\n• Python / Colab: “export to Colab”, “import a notebook”, “schedule design=3, build=5, design before build, one worker”, “planning with uncertainty”\n• 🎤 voice commands, 📷 image recognition and images → dataset\n• ⚙ AI mode: a local language model (Needle 2) for free-form commands.',
      'Να τι μπορώ να κάνω — ρώτα απλά, στα ελληνικά ή στα αγγλικά:\n• Δεδομένα: «φόρτωσε το diabetes», «περίγραψε τα δεδομένα», «πόσες ελλιπείς τιμές;», «κλάση y», «αφαίρεσε το duration», «κανονικοποίησε», «αναίρεση»\n• Ανάλυση: «τρέξε random forest», «σύγκρινε J48, NaiveBayes και SMO», «ομαδοποίηση με k-means σε 4 ομάδες», «βρες κανόνες συσχέτισης», «ποια χαρακτηριστικά μετράνε;»\n• Αποτελέσματα: «ποιο μοντέλο είναι καλύτερο;», «εξήγησε το αποτέλεσμα», «εξαγωγή PDF»\n• Μάθηση: «τι είναι το overfitting;», «δείξε μου το βίντεο για ταξινόμηση»\n• Web: επικόλλησε ένα URL ή «wikipedia πίνακας List of largest cities»\n• Python / Colab: «εξαγωγή σε Colab», «εισαγωγή notebook», «προγραμμάτισε design=3, build=5, design πριν build, ένας εργάτης», «σχεδιασμός με αβεβαιότητα»\n• 🎤 φωνητικές εντολές, 📷 αναγνώριση εικόνων και εικόνες → dataset\n• ⚙ AI mode: τοπικό γλωσσικό μοντέλο (Needle 2) για ελεύθερες εντολές.'), { chips: helpChips() });
    return true;
  }
  function nextChips(d) {
    const c = [];
    if (!d) return c;
    if (d.classAttr && d.classAttr.type === 'nominal') c.push(chip(T('Compare models', 'Σύγκρινε μοντέλα')), chip(T('Run J48', 'Τρέξε J48')));
    else if (d.classAttr) c.push(chip(T('Run LinearRegression', 'Τρέξε LinearRegression')));
    c.push(chip(T('Which attributes matter most?', 'Ποια χαρακτηριστικά μετράνε;')), chip(T('Cluster with k-means, k=3', 'Ομαδοποίηση με k-means, k=3')));
    if (d.attributes.every((a) => a.type === 'nominal')) c.push(chip(T('Find association rules', 'Βρες κανόνες συσχέτισης')));
    return c;
  }
  function describe(d) {
    const nNum = d.attributes.filter((a, j) => j !== d.classIndex && W.isNumericAttr(a)).length;
    const nNom = d.attributes.filter((a, j) => j !== d.classIndex && a.type === 'nominal').length;
    let miss = 0, cells = 0;
    for (const r of d.rows) for (const v of r) { cells++; if (v !== v) miss++; }
    const lines = [T(`“${d.relation}”: ${d.numInstances} instances, ${d.numAttributes} attributes (${nNum} numeric, ${nNom} nominal${d.classIndex >= 0 ? ', plus the class' : ''}).`, `«${d.relation}»: ${d.numInstances} εγγραφές, ${d.numAttributes} χαρακτηριστικά (${nNum} αριθμητικά, ${nNom} nominal${d.classIndex >= 0 ? ', συν την κλάση' : ''}).`)];
    const c = d.classAttr;
    if (c && c.type === 'nominal') {
      const cnt = U.zeros(c.values.length); d.rows.forEach((r, i) => { const v = r[d.classIndex]; if (v === v) cnt[v] += d.weights[i]; });
      const tot = U.sum(cnt);
      lines.push(T('Class “', 'Κλάση «') + c.name + T('”: ', '»: ') + c.values.map((v, k) => `${v} ${U.d2s(100 * cnt[k] / tot, 1)}%`).join(', ') + '.');
      const mx = Math.max(...cnt) / tot;
      if (mx > 0.75) lines.push(T(`⚠ The classes are imbalanced: always predicting the majority already gives ${U.d2s(100 * mx, 1)}% accuracy — watch recall and ROC AUC.`, `⚠ Οι κλάσεις είναι άνισες: αν προβλέπεις πάντα την πλειοψηφία παίρνεις ήδη ${U.d2s(100 * mx, 1)}% ακρίβεια — πρόσεξε recall και ROC AUC.`));
    } else if (c) {
      const st = d.attributeStats(d.classIndex);
      lines.push(T(`Numeric class “${c.name}”`, `Αριθμητική κλάση «${c.name}»`) + (st && st.mean !== undefined ? `: mean ${U.d2s(st.mean, 3)}, sd ${U.d2s(st.stdDev, 3)}.` : '.'));
    } else lines.push(T('No class attribute is set.', 'Δεν έχει οριστεί χαρακτηριστικό κλάσης.'));
    lines.push(miss ? T(`Missing values: ${miss} cells (${U.d2s(100 * miss / cells, 2)}%).`, `Ελλιπείς τιμές: ${miss} κελιά (${U.d2s(100 * miss / cells, 2)}%).`) : T('No missing values.', 'Δεν υπάρχουν ελλιπείς τιμές.'));
    if (d.attributes.some((a) => a.name.toLowerCase() === 'duration') && d.relation.toLowerCase().includes('bank')) lines.push(T('Note: “duration” is only known after the call — remove it for a realistic model (data leakage).', 'Σημείωση: το «duration» είναι γνωστό μόνο μετά την κλήση — αφαίρεσέ το για ρεαλιστικό μοντέλο (διαρροή δεδομένων).'));
    return lines.join('\n');
  }
  function missingText(d) {
    const rows = d.attributes.map((a, j) => { let m = 0; for (const r of d.rows) if (r[j] !== r[j]) m++; return [a.name, m]; }).filter((x) => x[1] > 0).sort((a, b) => b[1] - a[1]);
    if (!rows.length) return T('There are no missing values in this dataset. 👍', 'Δεν υπάρχουν ελλιπείς τιμές σε αυτό το σύνολο. 👍');
    return T(`${rows.length} attribute(s) have missing values:\n`, `${rows.length} χαρακτηριστικά έχουν ελλιπείς τιμές:\n`) + rows.slice(0, 10).map(([n, m]) => `• ${n}: ${m} (${U.d2s(100 * m / d.numInstances, 1)}%)`).join('\n') + T('\nSay “replace missing values” to fill them.', '\nΠες «συμπλήρωσε τις ελλιπείς τιμές» για να τις συμπληρώσω.');
  }

  function evalLine(e) {
    return e.nominal ? T(`accuracy ${fmtPct(e.pctCorrect())}, kappa ${U.d2s(e.kappa(), 3)}`, `ακρίβεια ${fmtPct(e.pctCorrect())}, kappa ${U.d2s(e.kappa(), 3)}`) + (e.weighted ? `, ROC AUC ${U.d2s(e.weighted(e.rocArea), 3)}` : '')
      : T(`correlation ${U.d2s(e.correlationCoefficient(), 3)}, MAE ${U.d2s(e.meanAbsoluteError(), 3)}, RMSE ${U.d2s(e.rootMeanSquaredError(), 3)}`, `συσχέτιση ${U.d2s(e.correlationCoefficient(), 3)}, MAE ${U.d2s(e.meanAbsoluteError(), 3)}, RMSE ${U.d2s(e.rootMeanSquaredError(), 3)}`);
  }
  function capsError(spec, d) { try { W.checkCapabilities(spec, d); return null; } catch (e) { return e.message; } }
  async function classify(name, folds) {
    if (!needData()) return false;
    const d = App.state.data;
    if (d.classIndex < 0) { bot(T('Set a class attribute first, e.g. “set class to …”.', 'Όρισε πρώτα χαρακτηριστικό κλάσης, π.χ. «κλάση …».')); return false; }
    if (!name) name = d.classIsNominal() ? 'J48' : 'LinearRegression';
    const spec = W.findClassifier(name);
    const err = capsError(spec, d);
    if (err) { bot('⚠ ' + err, { chips: d.classIsNominal() ? [chip(T('Run J48', 'Τρέξε J48'))] : [chip(T('Run LinearRegression', 'Τρέξε LinearRegression'))] }); return false; }
    const m = bot(T(`Running ${name} (10-fold cross-validation)…`, `Τρέχω ${name} (10-fold cross-validation)…`));
    if (folds) App.store.set('cls.mode', 'cv');
    const r = await runOnPage('classify', { classifier: { name, opts: W.defaultOptions(spec) } }, 'classify');
    if (!r) { update(m, { text: T(`${name} did not finish — see the message on the page.`, `Ο ${name} δεν ολοκληρώθηκε — δες το μήνυμα στη σελίδα.`) }); return false; }
    update(m, { text: `✅ ${name} — ${r.res.label}: ${evalLine(r.res.evaluation)}.`, chips: [chip(T('Explain the result', 'Εξήγησε το αποτέλεσμα')), chip(T('Compare models', 'Σύγκρινε μοντέλα')), chip(T('Export PDF', 'Εξαγωγή PDF'))] });
    return true;
  }
  async function compare(names) {
    if (!needData()) return false;
    const d = App.state.data;
    if (d.classIndex < 0) { bot(T('Set a class attribute first.', 'Όρισε πρώτα χαρακτηριστικό κλάσης.')); return false; }
    if (!names || names.length < 2) names = d.classIsNominal() ? ['ZeroR', 'NaiveBayes', 'J48', 'Logistic', 'RandomForest'] : ['ZeroR', 'LinearRegression', 'IBk', 'REPTree'];
    names = names.filter((n) => !capsError(W.findClassifier(n), d));
    if (d.numInstances > 3000) names = names.filter((n) => !['MultilayerPerceptron', 'SMO', 'KStar', 'IBk'].includes(n) || names.length <= 3);
    const m = bot(T(`Comparing ${names.join(', ')} with 10-fold cross-validation…`, `Συγκρίνω ${names.join(', ')} με 10-fold cross-validation…`), { progress: 0 });
    const res = [];
    for (let i = 0; i < names.length; i++) {
      update(m, { progress: i / names.length, text: T(`Comparing… ${names[i]} (${i + 1}/${names.length})`, `Σύγκριση… ${names[i]} (${i + 1}/${names.length})`) });
      const r = await runOnPage('classify', { classifier: { name: names[i], opts: W.defaultOptions(W.findClassifier(names[i])) } }, 'classify');
      if (r) res.push(r);
    }
    if (!res.length) { update(m, { progress: null, text: T('None of the models finished.', 'Κανένα μοντέλο δεν ολοκληρώθηκε.') }); return false; }
    const nominal = res[0].res.evaluation.nominal;
    const score = (r) => (nominal ? r.res.evaluation.pctCorrect() + r.res.evaluation.weighted(r.res.evaluation.rocArea) * 1e-3 : r.res.evaluation.correlationCoefficient());
    res.sort((a, b) => score(b) - score(a));
    const tbl = h('table.as-tbl', h('tr', h('th', T('Model', 'Μοντέλο')), h('th', nominal ? T('Accuracy', 'Ακρίβεια') : 'r'), h('th', nominal ? 'AUC' : 'RMSE')),
      res.map((r) => { const e = r.res.evaluation; return h('tr', h('td', r.name), h('td', nominal ? fmtPct(e.pctCorrect()) : U.d2s(e.correlationCoefficient(), 3)), h('td', nominal ? U.d2s(e.weighted(e.rocArea), 3) : U.d2s(e.rootMeanSquaredError(), 3))); }));
    update(m, { progress: null, node: tbl, text: T(`Done. Best: ${res[0].name}. The full comparison (with business costs) is at the bottom of the Classify page.`, `Έτοιμο. Καλύτερο: ${res[0].name}. Η πλήρης σύγκριση (με επιχειρηματικό κόστος) είναι στο κάτω μέρος της σελίδας Ταξινόμηση.`), chips: [chip(T('Explain the result', 'Εξήγησε το αποτέλεσμα')), chip(T('Export PDF', 'Εξαγωγή PDF'))] });
    setTimeout(() => { const c = [...document.querySelectorAll('#main .card')].pop(); if (c) c.scrollIntoView({ behavior: 'smooth', block: 'start' }); }, 300);
    return true;
  }
  async function cluster(name, k) {
    if (!needData()) return false;
    const spec = W.findClusterer(name);
    const opts = W.defaultOptions(spec);
    if (k && 'N' in opts) opts.N = k;
    const m = bot(T(`Clustering with ${name}${k ? ` (k=${k})` : ''}…`, `Ομαδοποίηση με ${name}${k ? ` (k=${k})` : ''}…`));
    const r = await runOnPage('cluster', { clusterer: { name, opts } }, 'cluster');
    if (!r) { update(m, { text: T('Clustering did not finish — see the message on the page.', 'Η ομαδοποίηση δεν ολοκληρώθηκε — δες το μήνυμα στη σελίδα.') }); return false; }
    const counts = r.res.counts || [];
    const tot = U.sum(counts);
    update(m, { text: `✅ ${name}: ${r.res.k} ${T('clusters', 'ομάδες')} — ` + counts.map((c, i) => `#${i}: ${c} (${U.d2s(100 * c / tot, 0)}%)`).join(', ') + '.', chips: [chip(T('Cluster with k-means, k=4', 'Ομαδοποίηση με k-means, k=4')), chip(T('Cluster with EM', 'Ομαδοποίηση με EM'))] });
    return true;
  }
  async function rules(name) {
    if (!needData()) return false;
    const d = App.state.data;
    if (d.attributes.some((a) => a.type !== 'nominal')) {
      bot(T('Association rules need nominal attributes only. Discretize the numeric ones first?', 'Οι κανόνες συσχέτισης θέλουν μόνο nominal χαρακτηριστικά. Να διακριτοποιήσω πρώτα τα αριθμητικά;'), { chips: [chipFn(T('Yes, discretize', 'Ναι, διακριτοποίησε'), async () => {
        const f = W.findFilter('Discretize'); const opts = Object.assign(W.defaultOptions(f), { B: 10, ignoreClass: true });
        let nd = W.applyFilter(f, opts, App.state.data); const s2n = W.findFilter('StringToNominal'); nd = W.applyFilter(s2n, W.defaultOptions(s2n), nd);
        App.applyChange(nd, 'Discretize -B 10'); await send(T('find association rules', 'βρες κανόνες συσχέτισης'));
      })] });
      return false;
    }
    const spec = W.associators.find((s) => s.name === name);
    const m = bot(T(`Mining rules with ${name}…`, `Εξόρυξη κανόνων με ${name}…`));
    const r = await runOnPage('associate', { associator: { name, opts: W.defaultOptions(spec) } }, 'associate');
    if (!r) { update(m, { text: T('No rules — see the message on the page.', 'Δεν βρέθηκαν κανόνες — δες το μήνυμα στη σελίδα.') }); return false; }
    const top = r.res.rules.slice(0, 3).map((x, i) => `${i + 1}. ${x.premise} ⇒ ${x.consequence} (conf ${U.d2s(x.conf, 2)}, lift ${U.d2s(x.lift, 2)})`).join('\n');
    update(m, { text: T(`✅ ${r.res.rules.length} rules. Top 3:\n`, `✅ ${r.res.rules.length} κανόνες. Οι 3 κορυφαίοι:\n`) + top });
    return true;
  }
  async function selectAttrs() {
    if (!needData()) return false;
    const d = App.state.data;
    if (d.classIndex < 0) { bot(T('Set a class attribute first — the ranking measures how much each attribute tells about the class.', 'Όρισε πρώτα χαρακτηριστικό κλάσης — η κατάταξη μετρά πόσα λέει κάθε χαρακτηριστικό για την κλάση.')); return false; }
    const evName = d.classIsNominal() ? 'InfoGainAttributeEval' : 'CorrelationAttributeEval';
    const ev = W.evaluators.find((s) => s.name === evName), se = W.searches.find((s) => s.name === 'Ranker');
    const m = bot(T(`Ranking attributes with ${evName}…`, `Κατάταξη χαρακτηριστικών με ${evName}…`));
    const r = await runOnPage('select', { evaluator: { name: evName, opts: W.defaultOptions(ev) }, search: { name: 'Ranker', opts: W.defaultOptions(se) } }, 'select');
    if (!r || !r.res.ranking) { update(m, { text: T('Attribute selection did not finish.', 'Η επιλογή χαρακτηριστικών δεν ολοκληρώθηκε.') }); return false; }
    const top = r.res.ranking.slice(0, 6).map((x, i) => `${i + 1}. ${x.name} (${U.d2s(x.merit, 4)})`).join('\n');
    const worst = r.res.ranking.filter((x) => x.merit <= 0).map((x) => x.name);
    update(m, { text: T('✅ Most informative attributes:\n', '✅ Τα πιο κατατοπιστικά χαρακτηριστικά:\n') + top + (worst.length ? T(`\nNo information about the class: ${worst.slice(0, 5).join(', ')}.`, `\nΧωρίς πληροφορία για την κλάση: ${worst.slice(0, 5).join(', ')}.`) : '') });
    return true;
  }
  function best() {
    const rs = App.state.results.classify;
    if (!rs.length) { bot(T('No models yet. Want me to compare a few?', 'Δεν υπάρχουν ακόμα μοντέλα. Να συγκρίνω μερικά;'), { chips: [chip(T('Compare models', 'Σύγκρινε μοντέλα'))] }); return true; }
    const nominal = rs.filter((r) => r.res.evaluation.nominal);
    const pool = nominal.length ? nominal : rs;
    const sc = (r) => (r.res.evaluation.nominal ? r.res.evaluation.pctCorrect() + r.res.evaluation.weighted(r.res.evaluation.rocArea) * 1e-3 : r.res.evaluation.correlationCoefficient());
    const sorted = pool.slice().sort((a, b) => sc(b) - sc(a));
    bot(T('Ranking of the models so far:\n', 'Κατάταξη των μοντέλων μέχρι τώρα:\n') + sorted.slice(0, 5).map((r, i) => `${i + 1}. ${r.name} (${r.relation}) — ${evalLine(r.res.evaluation)}`).join('\n') +
      (nominal.length ? T('\nTip: with imbalanced classes also compare business costs on the Classify page.', '\nΣυμβουλή: με άνισες κλάσεις σύγκρινε και το επιχειρηματικό κόστος στη σελίδα Ταξινόμηση.') : ''));
    return true;
  }
  function explainLast() {
    const r = App.state.results.classify[App.state.results.classify.length - 1];
    if (!r) { bot(T('Run a classifier first, then ask me to explain it.', 'Τρέξε πρώτα έναν ταξινομητή και μετά ζήτα μου να τον εξηγήσω.')); return true; }
    const e = r.res.evaluation, lines = [];
    if (e.nominal) {
      const labels = e.header.classAttr.values;
      const rowSums = e.confusion.map((row) => U.sum(row)), prior = (100 * Math.max(...rowSums)) / (U.sum(rowSums) || 1);
      lines.push(T(`${r.name} classified ${fmtPct(e.pctCorrect())} correctly; always guessing the majority class would give ${U.d2s(prior, 2)}%.`, `Ο ${r.name} ταξινόμησε σωστά το ${fmtPct(e.pctCorrect())}· αν μάντευε πάντα την πλειοψηφική κλάση θα έπαιρνε ${U.d2s(prior, 2)}%.`));
      const kp = e.kappa();
      lines.push(T(`Kappa ${U.d2s(kp, 3)} = ${kp > 0.8 ? 'almost perfect' : kp > 0.6 ? 'substantial' : kp > 0.4 ? 'moderate' : kp > 0.2 ? 'fair' : 'weak'} agreement beyond chance.`, `Kappa ${U.d2s(kp, 3)} = ${kp > 0.8 ? 'σχεδόν τέλεια' : kp > 0.6 ? 'ισχυρή' : kp > 0.4 ? 'μέτρια' : kp > 0.2 ? 'ασθενής' : 'πολύ ασθενής'} συμφωνία πέρα από την τύχη.`));
      labels.forEach((l, k) => lines.push(`• ${l}: recall ${U.d2s(e.recall(k), 3)}, precision ${U.d2s(e.precision(k), 3)}, AUC ${U.d2s(e.rocArea(k), 3)}`));
      const weak = labels.map((l, k) => [l, e.recall(k)]).sort((a, b) => a[1] - b[1])[0];
      if (weak[1] < 0.6) lines.push(T(`The model misses many “${weak[0]}” instances (recall ${U.d2s(weak[1], 2)}). If they matter most, try ClassBalancer, cost-sensitive comparison, or a model with better AUC.`, `Το μοντέλο χάνει πολλές εγγραφές «${weak[0]}» (recall ${U.d2s(weak[1], 2)}). Αν αυτές μετράνε περισσότερο, δοκίμασε ClassBalancer, σύγκριση κόστους ή μοντέλο με καλύτερο AUC.`));
    } else {
      lines.push(T(`${r.name}: correlation ${U.d2s(e.correlationCoefficient(), 3)} between predictions and true values (1 = perfect).`, `${r.name}: συσχέτιση ${U.d2s(e.correlationCoefficient(), 3)} μεταξύ προβλέψεων και πραγματικών τιμών (1 = τέλεια).`));
      lines.push(T(`On average the prediction is off by ${U.d2s(e.meanAbsoluteError(), 3)} (MAE); relative absolute error ${U.d2s(e.relativeAbsoluteError(), 1)}% of always predicting the mean.`, `Κατά μέσο όρο η πρόβλεψη απέχει ${U.d2s(e.meanAbsoluteError(), 3)} (MAE)· σχετικό απόλυτο σφάλμα ${U.d2s(e.relativeAbsoluteError(), 1)}% σε σχέση με την πρόβλεψη του μέσου όρου.`));
    }
    bot(lines.join('\n'), { chips: [chip(T('Compare models', 'Σύγκρινε μοντέλα')), chip(T('What is kappa?', 'Τι είναι το kappa;'))] });
    return true;
  }

  // ----------------------------------------------------------------- fallback: AI model, Wikipedia
  async function fallback(text, question) {
    if (S.set.ai && !question) {
      const ok = await aiInterpret(text);
      if (ok) return;
    }
    if (question && S.set.wiki) { if (await wikiAnswer(text)) return; }
    bot(T('Sorry, I did not understand that.', 'Συγγνώμη, δεν το κατάλαβα.') + (S.set.ai ? '' : T(' Tip: turn on AI mode (⚙) so a local language model can interpret free-form commands.', ' Συμβουλή: ενεργοποίησε το AI mode (⚙) ώστε ένα τοπικό γλωσσικό μοντέλο να ερμηνεύει ελεύθερες εντολές.')), { chips: helpChips().slice(0, 4).concat([chip(T('Help', 'Βοήθεια'))]) });
  }
  let lastWiki = 0;
  async function wikiAnswer(text) {
    let term = text.replace(/[?;!.]+$/, '').replace(/^(what is|what are|what's|whats|who is|who was|define|explain|tell me about|τι είναι|τι ειναι|τι σημαίνει|τι σημαινει|εξήγησε|εξηγησε|πες μου για|ποιος είναι|ποια είναι|ποιο είναι)\s+/i, '')
      .replace(/^(the|a|an|το|η|ο|τα|οι|την|τον)\s+/i, '').trim();
    if (!term || term.length > 80) return false;
    const wl = lang === 'el' ? 'el' : 'en';
    const wait = lastWiki + 1500 - Date.now(); if (wait > 0) await sleep(wait);
    lastWiki = Date.now();
    try {
      const opt = { credentials: 'omit', referrerPolicy: 'no-referrer' };
      const os = await (await fetch(`https://${wl}.wikipedia.org/w/api.php?action=opensearch&format=json&origin=*&limit=1&namespace=0&search=${encodeURIComponent(term)}`, opt)).json();
      const title = os && os[1] && os[1][0];
      if (!title) return false;
      const sm = await (await fetch(`https://${wl}.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title.replace(/ /g, '_'))}`, opt)).json();
      if (!sm || !sm.extract) return false;
      const ex = sm.extract.length > 700 ? sm.extract.slice(0, 700).replace(/\s\S*$/, '') + '…' : sm.extract;
      bot(ex, { src: T('Source: Wikipedia — ', 'Πηγή: Wikipedia — ') + sm.title + ' (CC BY-SA 4.0)', chips: sm.content_urls ? [chipFn(T('Open article', 'Άνοιγμα άρθρου'), () => window.open(sm.content_urls.desktop.page, '_blank', 'noopener'))] : [] });
      return true;
    } catch (e) { return false; }
  }

  // ----------------------------------------------------------------- Needle 2 (AI mode)
  const AI = { state: 'off', worker: null, runner: null, pending: new Map(), id: 0, progress: 0 };
  function loadScript(src) { return new Promise((res, rej) => { const s = document.createElement('script'); s.src = src; s.onload = res; s.onerror = () => rej(new Error('Cannot load ' + src)); document.head.appendChild(s); }); }
  async function aiLoad(fileBuf) {
    if (AI.state === 'ready') return true;
    AI.state = 'loading'; renderStatus();
    try {
      if (isHttp && window.Worker) {
        AI.worker = new Worker(BASE + 'needle2/needle-worker.js');
        await new Promise((resolve, reject) => {
          AI.worker.onmessage = (ev) => {
            const m = ev.data;
            if (m.type === 'ready') resolve();
            else if (m.type === 'progress' && m.id == null) { AI.progress = m.fraction; renderStatus(); }
            else if (m.type === 'error' && m.id == null) reject(new Error(m.message));
            else if (m.id != null && AI.pending.has(m.id)) {
              const p = AI.pending.get(m.id);
              if (m.type === 'progress') p.onProgress(m.fraction);
              else if (m.type === 'result') { AI.pending.delete(m.id); p.resolve(m.result); }
              else if (m.type === 'error') { AI.pending.delete(m.id); p.reject(new Error(m.message)); }
            }
          };
          AI.worker.onerror = (e) => reject(new Error(e.message || 'worker error'));
          AI.worker.postMessage({ type: 'load', buffer: fileBuf || null }, fileBuf ? [fileBuf] : []);
        });
      } else {
        if (!fileBuf) throw new Error(TU('Opened from file:// — choose models/needle2/needle2.cact to load the model.', 'Άνοιγμα από file:// — διάλεξε το models/needle2/needle2.cact για να φορτωθεί το μοντέλο.'));
        if (!window.NeedleRunner) for (const f of ['cact-full.js', 'tokenizer-full.js', 'engine-full.js', 'needle-runner.js']) await loadScript(BASE + 'needle2/' + f);
        AI.runner = new window.NeedleRunner(fileBuf);
      }
      AI.state = 'ready'; renderStatus();
      return true;
    } catch (e) {
      AI.state = 'error'; AI.error = e.message; renderStatus();
      if (AI.worker) { AI.worker.terminate(); AI.worker = null; }
      throw e;
    }
  }
  function aiTools() {
    const P = (props) => ({ type: 'object', properties: props, required: Object.keys(props) });
    return [
      { name: 'navigate', description: 'Open a page of the app', parameters: P({ page: { type: 'string' } }) },
      { name: 'load_dataset', description: 'Load a built-in dataset', parameters: P({ name: { type: 'string' } }) },
      { name: 'run_classifier', description: 'Train and evaluate a classifier', parameters: P({ name: { type: 'string' } }) },
      { name: 'run_clusterer', description: 'Cluster the data', parameters: P({ name: { type: 'string' } }) },
      { name: 'find_rules', description: 'Find association rules', parameters: P({}) },
      { name: 'select_attributes', description: 'Rank the most useful attributes', parameters: P({}) },
      { name: 'set_theme', description: 'Switch light or dark theme', parameters: P({ mode: { type: 'string' } }) },
      { name: 'export_pdf', description: 'Save the report as PDF', parameters: P({}) },
    ];
  }
  function aiCandidates() {
    const J = (n, a) => JSON.stringify({ name: n, arguments: a });
    const c = [];
    Object.keys(PAGES).forEach((p) => c.push(J('navigate', { page: p })));
    (W.DATASET_INDEX || []).forEach((x) => c.push(J('load_dataset', { name: x.id })));
    W.classifiers.forEach((s) => c.push(J('run_classifier', { name: s.name })));
    W.clusterers.forEach((s) => c.push(J('run_clusterer', { name: s.name })));
    c.push(J('find_rules', {}), J('select_attributes', {}), J('set_theme', { mode: 'dark' }), J('set_theme', { mode: 'light' }), J('export_pdf', {}));
    return c;
  }
  function aiRun(req, onProgress) {
    if (AI.worker) {
      const id = ++AI.id;
      return new Promise((resolve, reject) => { AI.pending.set(id, { resolve, reject, onProgress }); AI.worker.postMessage({ type: 'run', id, req }); });
    }
    return AI.runner.run(req, onProgress, () => new Promise((r) => setTimeout(r, 0)));
  }
  function callToStep(call) {
    const a = call.arguments || {};
    switch (call.name) {
      case 'navigate': return { type: 'nav', page: a.page };
      case 'load_dataset': return { type: 'load', id: a.name };
      case 'run_classifier': return { type: 'classify', name: a.name };
      case 'run_clusterer': return { type: 'cluster', name: a.name };
      case 'find_rules': return { type: 'rules', name: 'Apriori' };
      case 'select_attributes': return { type: 'select' };
      case 'set_theme': return { type: 'theme', mode: a.mode };
      case 'export_pdf': return { type: 'export', format: 'pdf' };
      default: return null;
    }
  }
  function describeCall(call) {
    const a = call.arguments || {};
    const m = { navigate: T('open the page “', 'άνοιγμα της σελίδας «') + (PAGE_NAMES[a.page] ? PAGE_NAMES[a.page][lang === 'el' ? 1 : 0] : a.page) + T('”', '»'),
      load_dataset: T('load the dataset ', 'φόρτωση του συνόλου ') + a.name, run_classifier: T('run the classifier ', 'εκτέλεση του ταξινομητή ') + a.name,
      run_clusterer: T('cluster with ', 'ομαδοποίηση με ') + a.name, find_rules: T('find association rules', 'εύρεση κανόνων συσχέτισης'), select_attributes: T('rank the attributes', 'κατάταξη χαρακτηριστικών'),
      set_theme: T('switch to the ', 'αλλαγή σε ') + (a.mode === 'dark' ? T('dark', 'σκοτεινό') : T('light', 'φωτεινό')) + T(' theme', ' θέμα'), export_pdf: T('export the report as PDF', 'εξαγωγή της αναφοράς σε PDF') };
    return m[call.name] || call.name;
  }
  async function aiInterpret(text) {
    if (AI.state !== 'ready') {
      try { await aiLoad(); } catch (e) { bot('⚠ ' + e.message); return false; }
    }
    const m = bot(T('🤖 Thinking with the local AI model (Needle 2)…', '🤖 Σκέφτομαι με το τοπικό μοντέλο AI (Needle 2)…'), { progress: 0 });
    const r = await aiRun({ system: 'You operate a data mining web app. Use the tool that matches the request.', tools: aiTools(), user: text, candidates: aiCandidates() }, (f) => update(m, { progress: f }));
    if (!r.call) { update(m, { progress: null, text: T('🤖 The AI model did not find a matching action.', '🤖 Το μοντέλο AI δεν βρήκε κατάλληλη ενέργεια.') + (r.think ? '\n' + T('(It thought: ', '(Σκέφτηκε: ') + r.think + ')' : '') }); return false; }
    const step = callToStep(r.call);
    if (!step) { update(m, { progress: null, text: T('🤖 No matching action.', '🤖 Καμία κατάλληλη ενέργεια.') }); return false; }
    update(m, { progress: null, text: T('🤖 I understood: ', '🤖 Κατάλαβα: ') + describeCall(r.call) + T('. Shall I do it?', '. Να το κάνω;'), src: `Needle 2 · ${(r.ms / 1000).toFixed(1)} s · ${T('runs locally', 'τρέχει τοπικά')}`,
      chips: [chipFn(T('✔ Yes, do it', '✔ Ναι, κάν’ το'), async () => { update(m, { chips: [] }); S.busy = true; renderMsgs(); try { await act(step); } catch (e) { bot('⚠ ' + e.message); } finally { S.busy = false; renderMsgs(); } }),
        chipFn(T('✖ No', '✖ Όχι'), () => update(m, { chips: [], text: m.text + T(' — cancelled.', ' — ακυρώθηκε.') }))] });
    return true;
  }

  // ----------------------------------------------------------------- Whistle (voice)
  const VOICE = { state: 'off', worker: null, model: null, rec: null, chunks: [], stream: null, timer: null, id: 0, pending: null };
  async function voiceLoad() {
    if (VOICE.state === 'ready') return;
    VOICE.state = 'loading'; renderStatus();
    try {
      if (isHttp && window.Worker) {
        const res = await fetch(BASE + 'whistle/whistle.cact');
        if (!res.ok) throw new Error('whistle.cact: HTTP ' + res.status);
        const buf = await res.arrayBuffer();
        VOICE.worker = new Worker(BASE + 'whistle/whistle_worker.js');
        await new Promise((resolve, reject) => {
          VOICE.worker.onmessage = (ev) => {
            const m = ev.data;
            if (m.type === 'loaded') resolve();
            else if (m.type === 'error' && !VOICE.pending) reject(new Error(m.message));
            else if (VOICE.pending) {
              if (m.type === 'progress') VOICE.pending.onProgress(m.overall != null ? m.overall : m.fraction);
              else if (m.type === 'result') { const p = VOICE.pending; VOICE.pending = null; p.resolve(m); }
              else if (m.type === 'error') { const p = VOICE.pending; VOICE.pending = null; p.reject(new Error(m.message)); }
            }
          };
          VOICE.worker.onerror = (e) => reject(new Error(e.message || 'worker error'));
          VOICE.worker.postMessage({ type: 'load', buffer: buf }, [buf]);
        });
      } else {
        const f = await App.openFileDialog('.cact');
        if (!f) throw new Error(TU('Choose models/whistle/whistle.cact to enable voice input.', 'Διάλεξε το models/whistle/whistle.cact για να ενεργοποιηθεί η φωνητική είσοδος.'));
        if (!window.Whistle) await loadScript(BASE + 'whistle/whistle.js');
        VOICE.model = new window.Whistle.Model(await f.arrayBuffer());
      }
      VOICE.state = 'ready'; renderStatus();
    } catch (e) { VOICE.state = 'error'; renderStatus(); throw e; }
  }
  function transcribe(pcm, onProgress) {
    const language = S.set.voiceLang || null;
    if (VOICE.worker) return new Promise((resolve, reject) => { VOICE.pending = { resolve, reject, onProgress }; VOICE.worker.postMessage({ type: 'transcribe', id: ++VOICE.id, pcm, language, beams: 3 }, [pcm.buffer]); });
    return sleep(30).then(() => VOICE.model.transcribe(pcm, { language, beams: 3 }));
  }
  async function voiceToggle(btn) {
    if (VOICE.rec && VOICE.rec.state === 'recording') { VOICE.rec.stop(); return; }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { bot(TU('This browser does not allow microphone access here (it needs https or localhost).', 'Ο browser δεν επιτρέπει πρόσβαση στο μικρόφωνο εδώ (χρειάζεται https ή localhost).')); return; }
    const m = VOICE.state !== 'ready' ? bot(TU('Loading the Whistle speech model (17 MB, runs locally)…', 'Φορτώνω το μοντέλο ομιλίας Whistle (17 MB, τρέχει τοπικά)…')) : null;
    try { await voiceLoad(); if (m) update(m, { text: TU('Speech model ready. 🎤', 'Το μοντέλο ομιλίας είναι έτοιμο. 🎤') }); } catch (e) { if (m) update(m, { text: '⚠ ' + e.message }); return; }
    try { VOICE.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } }); }
    catch (e) { bot(TU('Microphone permission was denied.', 'Δεν δόθηκε άδεια για το μικρόφωνο.')); return; }
    VOICE.chunks = [];
    VOICE.rec = new MediaRecorder(VOICE.stream);
    VOICE.rec.ondataavailable = (e) => { if (e.data.size) VOICE.chunks.push(e.data); };
    VOICE.rec.onstop = async () => {
      clearTimeout(VOICE.timer); btn.classList.remove('rec');
      VOICE.stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(VOICE.chunks, { type: VOICE.rec.mimeType });
      const mm = bot(TU('Transcribing…', 'Μεταγραφή…'), { progress: 0 });
      try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        const audio = await ctx.decodeAudioData(await blob.arrayBuffer());
        ctx.close();
        const pcm = new Float32Array(audio.getChannelData(0));
        if (pcm.length < 16000 * 0.4) { update(mm, { progress: null, text: TU('That was too short — hold the button a little longer.', 'Ήταν πολύ σύντομο — μίλησε λίγο περισσότερο.') }); return; }
        const r = await transcribe(pcm, (f) => update(mm, { progress: f }));
        const text = (r.text || '').trim();
        if (!text || r.silent) { update(mm, { progress: null, text: TU('I did not hear anything. Try again closer to the microphone.', 'Δεν άκουσα τίποτα. Δοκίμασε ξανά πιο κοντά στο μικρόφωνο.') }); return; }
        S.msgs.splice(S.msgs.indexOf(mm), 1);
        await send(text);
      } catch (e) { update(mm, { progress: null, text: '⚠ ' + e.message }); }
    };
    VOICE.rec.start();
    btn.classList.add('rec');
    VOICE.timer = setTimeout(() => { if (VOICE.rec.state === 'recording') VOICE.rec.stop(); }, 12000);
  }

  // ----------------------------------------------------------------- MobileCLIP (vision)
  const VIS = { state: 'off', worker: null, helpers: [], queue: [], cur: null, ready: null, progress: 0 };
  function visionStart() {
    if (VIS.ready) return VIS.ready;
    if (!isHttp) return Promise.reject(new Error(TU('Image recognition needs the app to be served over http(s) — e.g. run “python -m http.server” in the app folder, or use the GitHub Pages site.', 'Η αναγνώριση εικόνων χρειάζεται η εφαρμογή να σερβίρεται μέσω http(s) — π.χ. «python -m http.server» στον φάκελο της εφαρμογής ή η σελίδα GitHub Pages.')));
    VIS.state = 'loading'; renderStatus();
    const url = BASE + 'mobileclip/browser-fp32-opt-v3-worker-bundle.js?v=fp32-opt-v3';
    VIS.ready = new Promise((resolve, reject) => {
      const w = (VIS.worker = new Worker(url));
      const threads = Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1));
      for (let i = 0; i + 1 < threads; i++) {
        let hw; try { hw = new Worker(url + '&role=helper'); } catch (e) { continue; }
        const lost = () => w.postMessage({ type: 'pool-helper', helper: i, alive: false });
        hw.onmessage = (ev) => { const m = ev.data || {};
          if (m.type === 'helper-ready') w.postMessage({ type: 'pool-helper', helper: i, alive: true });
          else if (m.type === 'slice-result') w.postMessage({ type: 'pool-result', id: m.id, out: m.out }, [m.out]);
          else if (m.type === 'slice-error') lost(); };
        hw.onerror = lost;
        VIS.helpers[i] = hw;
      }
      w.onmessage = (ev) => {
        const m = ev.data || {};
        if (m.type === 'pool-task') { const hw = VIS.helpers[m.to]; if (hw) hw.postMessage(m.task, [m.task.x, m.task.w, m.task.b].filter(Boolean)); return; }
        if (m.type === 'load-progress') { VIS.progress = (m.percent || 0) / 100; renderStatus(); return; }
        if (m.type === 'ready') { VIS.state = 'ready'; renderStatus(); resolve(); pump(); return; }
        if (m.type === 'infer-progress') { if (VIS.cur && VIS.cur.onProgress) VIS.cur.onProgress((m.percent || 0) / 100); return; }
        if (m.type === 'result') { const c = VIS.cur; VIS.cur = null; if (c) c.resolve(m); pump(); return; }
        if (m.type === 'error') { const c = VIS.cur; VIS.cur = null; if (c) c.reject(new Error(m.text)); else { VIS.state = 'error'; renderStatus(); reject(new Error(m.text)); } pump(); }
      };
      w.onerror = (e) => { VIS.state = 'error'; renderStatus(); reject(new Error(e.message || 'worker error')); };
    });
    return VIS.ready;
  }
  function pump() {
    if (VIS.cur || VIS.state !== 'ready' || !VIS.queue.length) return;
    VIS.cur = VIS.queue.shift();
    VIS.worker.postMessage({ type: 'run', inputBuffer: VIS.cur.input.buffer }, [VIS.cur.input.buffer]);
  }
  async function imageTensor(file) {
    const bmp = await createImageBitmap(file);
    const c = document.createElement('canvas'); c.width = c.height = 256;
    const ctx = c.getContext('2d');
    const s = Math.min(bmp.width, bmp.height);
    ctx.drawImage(bmp, (bmp.width - s) / 2, (bmp.height - s) / 2, s, s, 0, 0, 256, 256);
    const px = ctx.getImageData(0, 0, 256, 256).data, plane = 256 * 256, out = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) { out[i] = px[i * 4] / 255; out[plane + i] = px[i * 4 + 1] / 255; out[2 * plane + i] = px[i * 4 + 2] / 255; }
    return { input: out, thumb: c.toDataURL('image/jpeg', 0.7) };
  }
  async function embed(file, onProgress) {
    await visionStart();
    const { input, thumb } = await imageTensor(file);
    const r = await new Promise((resolve, reject) => { VIS.queue.push({ input, resolve, reject, onProgress }); pump(); });
    return Object.assign(r, { thumb });
  }
  async function pickImage(file) {
    if (!file) { file = await App.openFileDialog('image/*'); if (!file) return; }
    const m = bot(VIS.state === 'ready' ? TU('Looking at the image…', 'Εξετάζω την εικόνα…') : TU('Loading MobileCLIP-S0 (45 MB, runs locally)… the first time takes a little while.', 'Φορτώνω το MobileCLIP-S0 (45 MB, τρέχει τοπικά)… την πρώτη φορά θέλει λίγο χρόνο.'), { progress: 0 });
    try {
      const r = await embed(file, (f) => update(m, { progress: f }));
      const top = (r.top || []).slice(0, 5);
      update(m, { progress: null, img: r.thumb, text: TU('I think this is: ', 'Νομίζω ότι είναι: ') + (top[0] ? top[0].label : '?') + '\n' + top.map((x) => `• ${x.label} — ${U.d2s(100 * x.score, 1)}%`).join('\n'),
        src: `MobileCLIP-S0 · ${(r.inferenceMs / 1000).toFixed(1)} s · ${TU('zero-shot over ~1,250 labels', 'zero-shot σε ~1.250 ετικέτες')}`,
        chips: [chip(TU('Images → dataset', 'Εικόνες → dataset'))] });
    } catch (e) { update(m, { progress: null, text: '⚠ ' + e.message }); }
  }
  function pickFiles(dir) {
    return new Promise((resolve) => {
      const inp = h('input', { type: 'file', accept: 'image/*', multiple: true, style: { display: 'none' } });
      if (dir) inp.webkitdirectory = true;
      inp.addEventListener('change', () => { resolve([...inp.files].filter((f) => /^image\//.test(f.type)).slice(0, 300)); inp.remove(); });
      document.body.appendChild(inp); inp.click();
    });
  }
  async function imagesToDataset() {
    bot(TU('Choose a folder with one sub-folder per class (e.g. cats/, dogs/), or several images whose names start with the class (cat_01.jpg). Each image becomes a row with 512 MobileCLIP embedding features — then try SMO, Logistic or k-means.', 'Διάλεξε έναν φάκελο με έναν υποφάκελο ανά κλάση (π.χ. cats/, dogs/) ή πολλές εικόνες με όνομα που ξεκινά από την κλάση (cat_01.jpg). Κάθε εικόνα γίνεται μια γραμμή με 512 χαρακτηριστικά embedding του MobileCLIP — μετά δοκίμασε SMO, Logistic ή k-means.'),
      { chips: [chipFn(TU('📁 Choose folder', '📁 Επιλογή φακέλου'), () => buildImageData(true)), chipFn(TU('🖼 Choose images', '🖼 Επιλογή εικόνων'), () => buildImageData(false))] });
  }
  async function buildImageData(dir) {
    const files = await pickFiles(dir);
    if (!files.length) return;
    const labelOf = (f) => {
      const parts = (f.webkitRelativePath || '').split('/');
      if (parts.length >= 3) return parts[parts.length - 2];
      const m = f.name.match(/^([A-Za-zͰ-Ͽ]+)[_\-\s.]?\d*/); return m ? m[1].toLowerCase() : 'unknown';
    };
    const m = bot(TU(`Embedding ${files.length} images…`, `Υπολογίζω embeddings για ${files.length} εικόνες…`), { progress: 0 });
    S.busy = true; renderMsgs();
    try {
      const rows = [], labels = [], zs = [];
      for (let i = 0; i < files.length; i++) {
        const r = await embed(files[i]);
        rows.push(Array.from(r.embedding)); labels.push(labelOf(files[i])); zs.push(r.top && r.top[0] ? r.top[0].label : '?');
        update(m, { progress: (i + 1) / files.length, text: TU(`Embedding images… ${i + 1}/${files.length}`, `Embeddings εικόνων… ${i + 1}/${files.length}`) });
      }
      const dim = rows[0].length;
      const classVals = [...new Set(labels)], zsVals = [...new Set(zs)];
      const attrs = [];
      for (let k = 0; k < dim; k++) attrs.push({ name: 'emb' + (k + 1), type: 'numeric' });
      attrs.push({ name: 'clip_label', type: 'nominal', values: zsVals }, { name: 'class', type: 'nominal', values: classVals });
      const data = rows.map((e, i) => [...e, zsVals.indexOf(zs[i]), classVals.indexOf(labels[i])]);
      App.loadNew(new W.Dataset('images-mobileclip', attrs, data, null, attrs.length - 1), 'images-mobileclip.arff');
      App.go('preprocess');
      update(m, { progress: null, text: TU(`Created a dataset: ${files.length} images × ${dim} embedding features, ${classVals.length} classes (${classVals.slice(0, 6).join(', ')}).`, `Δημιούργησα dataset: ${files.length} εικόνες × ${dim} χαρακτηριστικά embedding, ${classVals.length} κλάσεις (${classVals.slice(0, 6).join(', ')}).`),
        chips: [chip(TU('Run SMO', 'Τρέξε SMO')), chip(TU('Run Logistic', 'Τρέξε Logistic')), chip(TU('Cluster with k-means', 'Ομαδοποίηση με k-means'))] });
    } catch (e) { update(m, { progress: null, text: '⚠ ' + e.message }); } finally { S.busy = false; renderMsgs(); }
  }

  // ----------------------------------------------------------------- panel UI
  function renderStatus() {
    if (!statusEl) return;
    const pill = (name, st, p) => h('span.as-pill.' + st, name + (st === 'loading' && p ? ' ' + Math.round(p * 100) + '%' : ''));
    App.clear(statusEl).append(
      pill('Needle 2', S.set.ai ? AI.state === 'off' ? 'idle' : AI.state : 'off', AI.progress),
      pill('Whistle', VOICE.state === 'off' ? 'idle' : VOICE.state), pill('MobileCLIP', VIS.state === 'off' ? 'idle' : VIS.state, VIS.progress));
    if (S.view === 'settings') renderBody();
  }
  let bodyEl = null;
  function renderBody() {
    if (!bodyEl) return;
    App.clear(bodyEl);
    if (S.view === 'chat') { bodyEl.appendChild(msgBox); renderMsgs(); return; }
    const row = (title, desc, control) => h('div.as-set', h('div', h('div.as-set-t', title), h('div.as-set-d', desc)), control);
    const sw = (on, fn) => h('label.switch', h('input', { type: 'checkbox', checked: on, on: { change: (e) => fn(e.target.checked) } }), h('span'));
    const aiCtl = sw(S.set.ai, async (v) => {
      S.set.ai = v; saveSet();
      if (v && AI.state !== 'ready') {
        let buf = null;
        if (!isHttp) { const f = await App.openFileDialog('.cact'); if (!f) { S.set.ai = false; saveSet(); renderBody(); return; } buf = await f.arrayBuffer(); }
        aiLoad(buf).catch((e) => App.toast(e.message, 'error'));
      }
      renderStatus(); renderBody();
    });
    const langSel = h('select.select', { style: { width: '110px' }, on: { change: (e) => { S.set.voiceLang = e.target.value; saveSet(); } } },
      [['en', 'English'], ['de', 'Deutsch'], ['fr', 'Français'], ['es', 'Español'], ['it', 'Italiano'], ['nl', 'Nederlands'], ['pl', 'Polski']].map(([v, l]) => h('option', { value: v, selected: S.set.voiceLang === v }, l)));
    bodyEl.appendChild(h('div.as-settings',
      row(TU('AI mode (Needle 2)', 'AI mode (Needle 2)'), TU('A 13.7 MB language model that runs locally and turns free-form commands into one app action — always shown for your confirmation. About 10–40 s per command, depending on your device.', 'Γλωσσικό μοντέλο 13,7 MB που τρέχει τοπικά και μετατρέπει ελεύθερες εντολές σε μία ενέργεια της εφαρμογής — πάντα με επιβεβαίωσή σου. Περίπου 10–40 s ανά εντολή, ανάλογα με τη συσκευή.') + (AI.state === 'error' ? ' ⚠ ' + AI.error : ''), aiCtl),
      row(TU('Wikipedia look-ups', 'Αναζητήσεις στη Wikipedia'), TU('For “what is …?” questions I do not know, send only the term to Wikipedia and show a short, credited summary.', 'Για ερωτήσεις «τι είναι …;» που δεν ξέρω, στέλνεται μόνο ο όρος στη Wikipedia και εμφανίζεται σύντομη περίληψη με αναφορά.'), sw(S.set.wiki, (v) => { S.set.wiki = v; saveSet(); })),
      row(TU('Voice language (Whistle)', 'Γλώσσα φωνής (Whistle)'), TU('Whistle (17 MB) supports these 7 languages — not Greek.', 'Το Whistle (17 MB) υποστηρίζει αυτές τις 7 γλώσσες — όχι ελληνικά.'), langSel),
      h('div.as-set-d', { style: { marginTop: '10px' } }, TU('Image recognition uses MobileCLIP-S0 (Apple ML Research licence, research/academic use). Models load only when you first use them, and everything runs on your device. ', 'Η αναγνώριση εικόνων χρησιμοποιεί το MobileCLIP-S0 (άδεια Apple ML Research, για έρευνα/εκπαίδευση). Τα μοντέλα φορτώνονται μόνο όταν τα χρησιμοποιήσεις πρώτη φορά και όλα τρέχουν στη συσκευή σου. '),
        isHttp ? '' : TU('You opened the app from file://, so the AI and voice models are loaded from a file you choose, and image recognition is unavailable.', 'Άνοιξες την εφαρμογή από file://, οπότε τα μοντέλα AI και φωνής φορτώνονται από αρχείο που επιλέγεις και η αναγνώριση εικόνων δεν είναι διαθέσιμη.')),
      h('button.btn.sm', { style: { marginTop: '12px' }, on: { click: () => { S.view = 'chat'; renderBody(); } } }, TU('← Back to chat', '← Πίσω στη συζήτηση'))));
  }
  function build() {
    fab = h('button.as-fab', { title: TU('Assistant', 'Βοηθός'), 'aria-label': TU('Open the assistant', 'Άνοιγμα βοηθού'), on: { click: toggle } }, App.icon('chat'));
    msgBox = h('div.as-msgs');
    statusEl = h('div.as-status');
    inputEl = h('input.input', { maxlength: 300, placeholder: TU('Ask or give a command… e.g. “compare models”', 'Ρώτα ή δώσε εντολή… π.χ. «σύγκρινε μοντέλα»'), on: { keydown: (e) => { if (e.key === 'Enter') send(inputEl.value); } } });
    const mic = h('button.btn.ghost.icon.as-mic', { title: TU('Voice command (Whistle)', 'Φωνητική εντολή (Whistle)'), on: { click: () => voiceToggle(mic) } }, App.icon('mic'));
    const img = h('button.btn.ghost.icon', { title: TU('Recognise an image (MobileCLIP)', 'Αναγνώριση εικόνας (MobileCLIP)'), on: { click: () => send(TU('recognise an image', 'αναγνώριση εικόνας')) } }, App.icon('image'));
    const sendBtn = h('button.btn.primary.icon', { title: TU('Send', 'Αποστολή'), on: { click: () => send(inputEl.value) } }, App.icon('send'));
    bodyEl = h('div.as-body');
    panel = h('div.as-panel', { role: 'dialog', 'aria-label': TU('Weka assistant', 'Βοηθός Weka') },
      h('div.as-head', h('div.as-avatar', App.icon('sparkles')), h('div', { style: { flex: 1, minWidth: 0 } }, h('div.as-title', TU('Weka assistant', 'Βοηθός Weka')), statusEl),
        h('button.btn.ghost.icon.sm', { title: TU('Settings & models', 'Ρυθμίσεις & μοντέλα'), on: { click: () => { S.view = S.view === 'chat' ? 'settings' : 'chat'; renderBody(); } } }, App.icon('gear')),
        h('button.btn.ghost.icon.sm', { title: TU('Clear the conversation', 'Καθαρισμός συζήτησης'), on: { click: () => { S.msgs = []; greet(); } } }, App.icon('undo')),
        h('button.btn.ghost.icon.sm', { title: TU('Close', 'Κλείσιμο'), on: { click: toggle } }, App.icon('x'))),
      bodyEl,
      h('div.as-input', img, mic, inputEl, sendBtn));
    panel.style.display = 'none';
    document.body.append(fab, panel);
    // drop an image on the panel to recognise it
    panel.addEventListener('dragover', (e) => e.preventDefault());
    panel.addEventListener('drop', (e) => { const f = e.dataTransfer.files[0]; if (f && /^image\//.test(f.type)) { e.preventDefault(); lang = I18N.lang; pickImage(f); } });
    renderBody(); renderStatus();
  }
  function greet() {
    lang = I18N.lang;
    bot(T('Hi! I am the Weka assistant. I can load data, run and compare algorithms, explain results, import web tables and more — in English or Greek. Try:', 'Γεια! Είμαι ο βοηθός του Weka Web. Φορτώνω δεδομένα, τρέχω και συγκρίνω αλγορίθμους, εξηγώ αποτελέσματα, εισάγω πίνακες από το web και άλλα — στα ελληνικά ή στα αγγλικά. Δοκίμασε:'), { chips: helpChips() });
  }
  function toggle() {
    S.open = !S.open;
    panel.style.display = S.open ? 'flex' : 'none';
    fab.classList.toggle('open', S.open);
    if (S.open) { if (!S.msgs.length) greet(); renderBody(); setTimeout(() => inputEl.focus(), 50); }
  }

  App.assistant = { send, toggle, parse, open: () => { if (!S.open) toggle(); }, image: (f) => { lang = I18N.lang; if (!S.open) toggle(); return pickImage(f); }, _state: S, _models: { voiceLoad, transcribe, embed, aiLoad } };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build); else build();
})();
