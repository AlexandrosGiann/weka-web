/* Weka Web — γλώσσες. Τα κείμενα του κώδικα περνούν από το tag L`...`.
 * Κλειδί = το κείμενο της πηγής (ελληνικά για τον αρχικό κώδικα, αγγλικά για νεότερο κώδικα) με {0},{1}… για τιμές.
 * Η προεπιλεγμένη γλώσσα είναι τα αγγλικά. */
(function () {
  'use strict';
  let lang = 'en';
  try { lang = JSON.parse(localStorage.getItem('wekaweb.lang')) || 'en'; } catch (e) { /* ignore */ }
  if (lang !== 'el' && lang !== 'en') lang = 'en';
  const I18N = (window.I18N = { lang, dict: { en: {}, el: {} } });
  I18N.add = (code, map) => Object.assign(I18N.dict[code], map);
  I18N.locale = () => (I18N.lang === 'el' ? 'el-GR' : 'en-GB');
  I18N.setLang = (l) => { try { localStorage.setItem('wekaweb.lang', JSON.stringify(l)); } catch (e) { /* ignore */ } };
  /** Tagged template: L`Κείμενο ${x}` → μετάφραση με αντικατάσταση {0}. */
  window.L = function (strings, ...vals) {
    let key = '';
    for (let i = 0; i < strings.length; i++) key += strings[i] + (i < vals.length ? '{' + i + '}' : '');
    const d = I18N.dict[I18N.lang];
    let t = d && Object.prototype.hasOwnProperty.call(d, key) ? d[key] : key;
    if (!vals.length) return t;
    return t.replace(/\{(\d+)\}/g, (m, k) => (+k < vals.length ? String(vals[+k]) : m));
  };
  /** Μετάφραση κειμένου χρόνου εκτέλεσης (π.χ. ονόματα που αποθηκεύτηκαν). */
  window.Lk = (s) => { const d = I18N.dict[I18N.lang]; return d && Object.prototype.hasOwnProperty.call(d, s) ? d[s] : s; };
  if (typeof document !== 'undefined') document.documentElement.setAttribute('lang', lang);
})();
