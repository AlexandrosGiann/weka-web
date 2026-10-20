/* Weka Web — Evaluation: μετρικές, cross-validation, αναφορές σε μορφή Weka. */
(function () {
  'use strict';
  const W = window.WEKA, U = W.Utils;

  /** Utils.doubleToString(value, width, afterDecimal) με τη λογική στοίχισης του Weka. */
  function dts(value, width, after) {
    const temp = U.d2s(value, after);
    if (after >= width) return temp;
    const result = new Array(width).fill(' ');
    let dot;
    if (after > 0) {
      dot = temp.indexOf('.');
      if (dot === -1) dot = temp.length;
      else result[width - after - 1] = '.';
    } else dot = temp.length;
    let offset = width - after - dot;
    if (after > 0) offset--;
    if (offset < 0) return temp;
    for (let i = 0; i < dot; i++) result[offset + i] = temp[i];
    for (let i = dot + 1; i < temp.length; i++) result[offset + i] = temp[i];
    return result.join('');
  }
  W.dts = dts;

  class Evaluation {
    constructor(train) {
      this.header = train;
      this.nominal = train.classIsNominal();
      this.numClasses = this.nominal ? train.numClasses : 1;
      const k = this.numClasses;
      this.confusion = U.matrix(k, k);
      this.withClass = 0; this.unclassified = 0; this.correct = 0; this.incorrect = 0;
      this.sumErr = 0; this.sumAbsErr = 0; this.sumSqrErr = 0;
      this.sumPriorAbsErr = 0; this.sumPriorSqrErr = 0;
      this.sumClass = 0; this.sumSqrClass = 0; this.sumPredicted = 0; this.sumSqrPredicted = 0; this.sumClassPredicted = 0;
      this.predictions = []; // {actual, predicted, dist, weight, index}
      this.setPriors(train);
    }
    setPriors(train) {
      const ci = train.classIndex;
      if (this.nominal) {
        this.priors = new Array(this.numClasses).fill(1);
        this.priorsSum = this.numClasses;
        for (let i = 0; i < train.numInstances; i++) {
          const c = train.rows[i][ci];
          if (c === c) { this.priors[c] += train.weights[i]; this.priorsSum += train.weights[i]; }
        }
      } else {
        this.priors = [0]; this.priorsSum = 0;
        for (let i = 0; i < train.numInstances; i++) {
          const c = train.rows[i][ci];
          if (c === c) { this.priors[0] += c * train.weights[i]; this.priorsSum += train.weights[i]; }
        }
      }
    }
    /** Αξιολόγηση μοντέλου σε σύνολο ελέγχου· επιστρέφει τις προβλέψεις. */
    evaluateModel(model, test, rowIndexBase = null) {
      const ci = test.classIndex;
      for (let i = 0; i < test.numInstances; i++) {
        const row = test.rows[i];
        const dist = model.distributionForRow(row);
        this.record(dist, row[ci], test.weights[i], rowIndexBase ? rowIndexBase[i] : i, row);
      }
    }
    record(dist, actual, weight, index, row) {
      if (this.nominal) {
        if (actual !== actual) return;
        this.withClass += weight;
        let pred = -1, best = 0;
        for (let i = 0; i < dist.length; i++) if (dist[i] > best) { pred = i; best = dist[i]; }
        // numeric scores
        let se = 0, sse = 0, spe = 0, spse = 0;
        for (let i = 0; i < this.numClasses; i++) {
          const a = i === actual ? 1 : 0;
          const d = (dist[i] || 0) - a;
          se += Math.abs(d); sse += d * d;
          const pd = this.priors[i] / this.priorsSum - a;
          spe += Math.abs(pd); spse += pd * pd;
        }
        this.sumAbsErr += weight * se / this.numClasses;
        this.sumSqrErr += weight * sse / this.numClasses;
        this.sumPriorAbsErr += weight * spe / this.numClasses;
        this.sumPriorSqrErr += weight * spse / this.numClasses;
        this.predictions.push({ actual, predicted: pred, dist: Array.from(dist), weight, index, row });
        if (pred < 0) { this.unclassified += weight; return; }
        this.confusion[actual][pred] += weight;
        if (pred === actual) this.correct += weight; else this.incorrect += weight;
      } else {
        if (actual !== actual) return;
        const p = dist[0];
        this.withClass += weight;
        this.predictions.push({ actual, predicted: p, weight, index, row });
        if (p !== p) { this.unclassified += weight; return; }
        this.sumClass += weight * actual; this.sumSqrClass += weight * actual * actual;
        this.sumClassPredicted += weight * actual * p;
        this.sumPredicted += weight * p; this.sumSqrPredicted += weight * p * p;
        const d = p - actual;
        this.sumErr += weight * d;
        this.sumAbsErr += weight * Math.abs(d); this.sumSqrErr += weight * d * d;
        const prior = this.priorsSum > 0 ? this.priors[0] / this.priorsSum : 0;
        const pd = prior - actual;
        this.sumPriorAbsErr += weight * Math.abs(pd); this.sumPriorSqrErr += weight * pd * pd;
      }
    }
    get n() { return this.withClass - this.unclassified; }
    pctCorrect() { return 100 * this.correct / this.withClass; }
    pctIncorrect() { return 100 * this.incorrect / this.withClass; }
    pctUnclassified() { return 100 * this.unclassified / this.withClass; }
    meanAbsoluteError() { return this.sumAbsErr / this.n; }
    rootMeanSquaredError() { return Math.sqrt(this.sumSqrErr / this.n); }
    meanPriorAbsoluteError() { return this.sumPriorAbsErr / this.withClass; }
    rootMeanPriorSquaredError() { return Math.sqrt(this.sumPriorSqrErr / this.withClass); }
    relativeAbsoluteError() { return 100 * this.meanAbsoluteError() / this.meanPriorAbsoluteError(); }
    rootRelativeSquaredError() { return 100 * this.rootMeanSquaredError() / this.rootMeanPriorSquaredError(); }
    kappa() {
      const k = this.numClasses, cm = this.confusion;
      const rows = U.zeros(k), cols = U.zeros(k);
      let tot = 0;
      for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) { rows[i] += cm[i][j]; cols[j] += cm[i][j]; tot += cm[i][j]; }
      let correct = 0, chance = 0;
      for (let i = 0; i < k; i++) { chance += rows[i] * cols[i]; correct += cm[i][i]; }
      chance /= tot * tot; correct /= tot;
      return chance < 1 ? (correct - chance) / (1 - chance) : 1;
    }
    correlationCoefficient() {
      const n = this.n;
      const va = this.sumSqrClass - this.sumClass * this.sumClass / n;
      const vp = this.sumSqrPredicted - this.sumPredicted * this.sumPredicted / n;
      const vprod = this.sumClassPredicted - this.sumClass * this.sumPredicted / n;
      return va * vp <= 0 ? 0 : vprod / Math.sqrt(va * vp);
    }
    // ---- per-class
    tp(c) { return this.confusion[c][c]; }
    numTruePositives(c) { return this.confusion[c][c]; }
    numFalsePositives(c) { let s = 0; for (let i = 0; i < this.numClasses; i++) if (i !== c) s += this.confusion[i][c]; return s; }
    numFalseNegatives(c) { let s = 0; for (let j = 0; j < this.numClasses; j++) if (j !== c) s += this.confusion[c][j]; return s; }
    numTrueNegatives(c) { let s = 0; for (let i = 0; i < this.numClasses; i++) for (let j = 0; j < this.numClasses; j++) if (i !== c && j !== c) s += this.confusion[i][j]; return s; }
    truePositiveRate(c) { const tp = this.tp(c), fn = this.numFalseNegatives(c); return tp + fn === 0 ? 0 : tp / (tp + fn); }
    falsePositiveRate(c) { const fp = this.numFalsePositives(c), tn = this.numTrueNegatives(c); return fp + tn === 0 ? 0 : fp / (fp + tn); }
    precision(c) { const tp = this.tp(c), fp = this.numFalsePositives(c); return tp + fp === 0 ? NaN : tp / (tp + fp); }
    recall(c) { return this.truePositiveRate(c); }
    fMeasure(c) { const p = this.precision(c), r = this.recall(c); if (p !== p) return NaN; return p + r === 0 ? 0 : 2 * p * r / (p + r); }
    mcc(c) {
      const tp = this.tp(c), fp = this.numFalsePositives(c), fn = this.numFalseNegatives(c), tn = this.numTrueNegatives(c);
      const den = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
      return den === 0 ? NaN : (tp * tn - fp * fn) / den;
    }
    classCount(c) { let s = 0; for (let j = 0; j < this.numClasses; j++) s += this.confusion[c][j]; return s; }
    /** Σημεία καμπύλης ROC/PR για μια κλάση (λογική ThresholdCurve: ομαδοποίηση ισοβαθμιών). */
    thresholdCurve(c) {
      const preds = this.predictions.filter((p) => p.predicted >= 0 || p.dist);
      const items = preds.map((p) => ({ s: p.dist[c], pos: p.actual === c, w: p.weight }));
      items.sort((a, b) => b.s - a.s);
      let P = 0, N = 0;
      for (const it of items) if (it.pos) P += it.w; else N += it.w;
      const pts = [{ fpr: 0, tpr: 0, precision: 1, recall: 0, threshold: Infinity }];
      let tp = 0, fp = 0;
      for (let i = 0; i < items.length;) {
        const s = items[i].s;
        while (i < items.length && items[i].s === s) { if (items[i].pos) tp += items[i].w; else fp += items[i].w; i++; }
        pts.push({ fpr: N ? fp / N : 0, tpr: P ? tp / P : 0, precision: tp + fp ? tp / (tp + fp) : 1, recall: P ? tp / P : 0, threshold: s });
      }
      return { points: pts, P, N };
    }
    rocArea(c) {
      const { points, P, N } = this.thresholdCurve(c);
      if (!P || !N) return NaN;
      let a = 0;
      for (let i = 1; i < points.length; i++) a += (points[i].fpr - points[i - 1].fpr) * (points[i].tpr + points[i - 1].tpr) / 2;
      return a;
    }
    prcArea(c) {
      const { points, P } = this.thresholdCurve(c);
      if (!P) return NaN;
      let a = 0;
      for (let i = 1; i < points.length; i++) a += (points[i].recall - points[i - 1].recall) * (points[i].precision + points[i - 1].precision) / 2;
      return a;
    }
    weighted(fn) {
      let s = 0, tot = 0;
      for (let c = 0; c < this.numClasses; c++) {
        const w = this.classCount(c);
        const v = fn.call(this, c);
        if (v === v) s += w * v;
        tot += w;
      }
      return tot > 0 ? s / tot : NaN;
    }

    // ---------------------------------------------------------- αναφορές
    toSummaryString(title = '=== Summary ===') {
      const t = [title, ''];
      if (this.withClass > 0) {
        if (this.nominal) {
          t.push('Correctly Classified Instances     ' + dts(this.correct, 12, 4) + '     ' + dts(this.pctCorrect(), 12, 4) + ' %');
          t.push('Incorrectly Classified Instances   ' + dts(this.incorrect, 12, 4) + '     ' + dts(this.pctIncorrect(), 12, 4) + ' %');
          t.push('Kappa statistic                    ' + dts(this.kappa(), 12, 4));
        } else {
          t.push('Correlation coefficient            ' + dts(this.correlationCoefficient(), 12, 4));
        }
        t.push('Mean absolute error                ' + dts(this.meanAbsoluteError(), 12, 4));
        t.push('Root mean squared error            ' + dts(this.rootMeanSquaredError(), 12, 4));
        t.push('Relative absolute error            ' + dts(this.relativeAbsoluteError(), 12, 4) + ' %');
        t.push('Root relative squared error        ' + dts(this.rootRelativeSquaredError(), 12, 4) + ' %');
        if (this.unclassified > 0) {
          t.push('UnClassified Instances             ' + dts(this.unclassified, 12, 4) + '     ' + dts(this.pctUnclassified(), 12, 4) + ' %');
        }
      }
      t.push('Total Number of Instances          ' + dts(this.withClass, 12, 4));
      return t.join('\n') + '\n';
    }
    toClassDetailsString() {
      const labels = this.header.classAttr.values;
      const f = (v) => (v !== v ? '?        ' : U.padR(v.toFixed(3), 9));
      const lines = ['=== Detailed Accuracy By Class ===', '',
        '                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class'];
      for (let c = 0; c < this.numClasses; c++) {
        lines.push('                 ' + f(this.truePositiveRate(c)) + f(this.falsePositiveRate(c)) + U.padR(f(this.precision(c)), 11) + f(this.recall(c)) + U.padR(f(this.fMeasure(c)), 11) + f(this.mcc(c)) + U.padR(f(this.rocArea(c)), 10) + U.padR(f(this.prcArea(c)), 10) + labels[c]);
      }
      lines.push('Weighted Avg.    ' + f(this.weighted(this.truePositiveRate)) + f(this.weighted(this.falsePositiveRate)) + U.padR(f(this.weighted(this.precision)), 11) + f(this.weighted(this.recall)) + U.padR(f(this.weighted(this.fMeasure)), 11) + f(this.weighted(this.mcc)) + U.padR(f(this.weighted(this.rocArea)), 10) + U.padR(f(this.weighted(this.prcArea)), 10));
      return lines.join('\n') + '\n';
    }
    toMatrixString() {
      const k = this.numClasses, labels = this.header.classAttr.values;
      const ids = [];
      for (let i = 0; i < k; i++) ids.push(letterId(i, k));
      let fractional = false, maxv = 0;
      for (const r of this.confusion) for (const v of r) { if (v % 1 !== 0) fractional = true; if (v > maxv) maxv = v; }
      const IDWidth = Math.max(1 + Math.max(Math.floor(Math.log10(maxv || 1)) + (fractional ? 4 : 1), ids[k - 1].length), 2) + (fractional ? 1 : 0);
      const lines = ['=== Confusion Matrix ===', ''];
      let head = '';
      for (let i = 0; i < k; i++) head += ' ' + U.padL(ids[i], IDWidth - 1);
      lines.push(head + '   <-- classified as');
      for (let i = 0; i < k; i++) {
        let l = '';
        for (let j = 0; j < k; j++) l += ' ' + (fractional ? dts(this.confusion[i][j], IDWidth - 1, 2) : U.padL(this.confusion[i][j], IDWidth - 1));
        lines.push(l + ' | ' + ids[i] + ' = ' + labels[i]);
      }
      return lines.join('\n') + '\n';
    }
  }
  function letterId(index, num) {
    const IDChars = 'abcdefghijklmnopqrstuvwxyz';
    let IDWidth = 1, n = num;
    while (n >= IDChars.length) { n = Math.floor(n / IDChars.length); IDWidth++; }
    let s = '';
    let idx = index;
    for (let j = IDWidth - 1; j >= 0; j--) { s = IDChars[idx % IDChars.length] + s; idx = Math.floor(idx / IDChars.length) - 1; if (idx < 0) break; }
    return s;
  }
  W.Evaluation = Evaluation;

  /**
   * Εκτέλεση ταξινομητή με επιλεγμένο τρόπο αξιολόγησης.
   * opts: { data, spec, options, mode: 'cv'|'split'|'train'|'test', folds, seed, percentage, preserveOrder, testData, onProgress }
   */
  W.runClassifier = async function (opts) {
    const { spec, options } = opts;
    const data = opts.data.withoutMissingClass();
    if (data.classIndex < 0) throw new Error(L`Επίλεξε χαρακτηριστικό στόχο (class).`);
    W.checkCapabilities(spec, data);
    const progress = opts.onProgress || (() => {});
    const t0 = performance.now();
    const make = () => spec.create(Object.assign({}, options));
    progress(0, L`Εκπαίδευση μοντέλου σε όλο το σύνολο…`);
    await W.nextFrame();
    const fullModel = make();
    fullModel.build(data);
    const buildTime = (performance.now() - t0) / 1000;
    let evaluation, label;
    const mode = opts.mode || 'cv';
    const seed = opts.seed ?? 1;
    if (mode === 'cv') {
      const folds = Math.max(2, Math.min(opts.folds || 10, data.numInstances));
      label = `${folds}-fold cross-validation`;
      const rand = new W.JavaRandom(seed);
      const d = data.copy();
      d.randomize(rand);
      if (d.classIsNominal()) d.stratify(folds);
      evaluation = new Evaluation(d);
      for (let f = 0; f < folds; f++) {
        progress((f + 1) / (folds + 1), `Fold ${f + 1} / ${folds}`);
        await W.nextFrame();
        const train = d.trainCV(folds, f, rand);
        evaluation.setPriors(train);
        const m = make();
        m.build(train);
        const test = d.testCV(folds, f);
        evaluation.evaluateModel(m, test);
      }
    } else if (mode === 'split') {
      const pct = opts.percentage ?? 66;
      const d = data.copy();
      if (!opts.preserveOrder) d.randomize(new W.JavaRandom(seed));
      const trainSize = Math.round(d.numInstances * pct / 100);
      const testSize = d.numInstances - trainSize;
      if (trainSize < 1 || testSize < 1) throw new Error(L`Ο διαχωρισμός δίνει κενό σύνολο εκπαίδευσης ή ελέγχου.`);
      label = `split ${pct}% train, remainder test`;
      const train = d.withRows(d.rows.slice(0, trainSize), d.weights.slice(0, trainSize));
      const test = d.withRows(d.rows.slice(trainSize), d.weights.slice(trainSize));
      progress(0.5, L`Εκπαίδευση στο ` + pct + '%…');
      await W.nextFrame();
      const m = make();
      m.build(train);
      evaluation = new Evaluation(train);
      evaluation.evaluateModel(m, test);
    } else if (mode === 'train') {
      label = 'evaluate on training data';
      evaluation = new Evaluation(data);
      evaluation.evaluateModel(fullModel, data);
    } else if (mode === 'test') {
      const test = opts.testData;
      if (!test) throw new Error(L`Φόρτωσε πρώτα σύνολο ελέγχου (test set).`);
      const msg = W.equalHeadersMsg(data, test);
      if (msg) throw new Error(L`Ασύμβατο σύνολο ελέγχου: ` + msg);
      test.classIndex = data.classIndex;
      label = 'user supplied test set: ' + test.numInstances + ' instances';
      evaluation = new Evaluation(data);
      evaluation.evaluateModel(fullModel, test);
    } else throw new Error(L`Άγνωστος τρόπος αξιολόγησης`);
    progress(1, L`Ολοκληρώθηκε`);
    const totalTime = (performance.now() - t0) / 1000;

    const optStr = W.optionString(spec, options);
    const runInfo = [
      '=== Run information ===', '',
      'Scheme:       ' + spec.path + (optStr ? ' ' + optStr : ''),
      'Relation:     ' + data.relation,
      'Instances:    ' + data.numInstances,
      'Attributes:   ' + data.numAttributes,
      ...data.attributes.map((a) => '              ' + a.name),
      'Test mode:    ' + label, '',
      '=== Classifier model (full training set) ===', '',
      fullModel.toString(), '',
      'Time taken to build model: ' + U.d2s(buildTime, 2) + ' seconds', '',
    ];
    const evalTitle = mode === 'cv' ? '=== Stratified cross-validation ===' : mode === 'train' ? '=== Evaluation on training set ===' : mode === 'split' ? '=== Evaluation on test split ===' : '=== Evaluation on test set ===';
    let report = runInfo.join('\n') + '\n' + evalTitle + '\n' + evaluation.toSummaryString();
    if (evaluation.nominal) report += '\n' + evaluation.toClassDetailsString() + '\n' + evaluation.toMatrixString();
    return { model: fullModel, evaluation, report, label, mode, time: totalTime, spec, options, optStr, data };
  };

  /** Έλεγχος δυνατοτήτων (π.χ. ο αλγόριθμος δέχεται μόνο nominal κλάση). */
  W.checkCapabilities = function (spec, data) {
    const cap = spec.caps || {};
    const ca = data.classAttr;
    if (ca) {
      if (ca.type === 'nominal' && cap.nominalClass === false) throw new Error(L`Ο ${spec.name} δεν υποστηρίζει nominal κλάση. Επίλεξε αριθμητικό στόχο ή άλλον αλγόριθμο.`);
      if (ca.type === 'nominal' && cap.binaryClass && ca.values.length !== 2) throw new Error(L`${spec.name} supports only two-valued (binary) classes. Pick a binary target or use another algorithm.`);
      if (ca.type !== 'nominal' && cap.numericClass === false) throw new Error(L`Ο ${spec.name} δεν υποστηρίζει αριθμητική κλάση. Επίλεξε nominal στόχο ή εφάρμοσε Discretize.`);
    }
    for (let j = 0; j < data.numAttributes; j++) {
      if (j === data.classIndex) continue;
      const a = data.attributes[j];
      if (a.type === 'string') throw new Error(L`Το χαρακτηριστικό "${a.name}" είναι τύπου string. Αφαίρεσέ το ή μετέτρεψέ το (π.χ. StringToNominal) πρώτα.`);
      if (a.type === 'nominal' && cap.nominalAttrs === false) throw new Error(L`Ο ${spec.name} δέχεται μόνο αριθμητικά χαρακτηριστικά ("${a.name}" είναι nominal). Δοκίμασε το φίλτρο NominalToBinary.`);
      if (a.type !== 'nominal' && cap.numericAttrs === false) throw new Error(L`Ο ${spec.name} δέχεται μόνο nominal χαρακτηριστικά ("${a.name}" είναι αριθμητικό). Δοκίμασε το φίλτρο Discretize.`);
    }
  };

  /** Αναπαράσταση επιλογών σε μορφή γραμμής εντολών Weka. */
  W.optionString = function (spec, options) {
    const parts = [];
    for (const o of spec.options || []) {
      if (o.hidden) continue;
      const v = options[o.key];
      if (o.type === 'bool') { if (v && o.flag) parts.push('-' + o.flag); if (!v && o.flagFalse) parts.push('-' + o.flagFalse); continue; }
      if (o.cliFlags) { if (o.cliFlags[v]) parts.push('-' + o.cliFlags[v]); continue; }
      if (o.flag && v !== undefined && v !== '') parts.push('-' + o.flag + ' ' + (v === -1.7976931348623157e308 ? '-1.7976931348623157E308' : v));
    }
    return parts.join(' ');
  };
  W.defaultOptions = function (spec) {
    const o = {};
    for (const op of spec.options || []) o[op.key] = op.default;
    return o;
  };
})();
