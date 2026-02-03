'use strict';

Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });

const FastText = require('../../FastText.cjs');
const languages = require('./assets/languages.json.cjs');

const LANGUAGES_LIMIT = Object.keys(languages.default).length;
class BaseLanguageIdentificationModel {
  getFastTextModule;
  wasmPath;
  modelPath;
  model = null;
  static formatLang(raw) {
    const identifyLang = raw.replace("__label__", "");
    return BaseLanguageIdentificationModel.normalizeIdentifyLang(identifyLang);
  }
  /**
   * fastText [language identification model](https://fasttext.cc/docs/en/language-identification.html)
   * result is based on [List of Wikipedias](https://en.wikipedia.org/wiki/List_of_Wikipedias) `WP code`
   *
   * This lib provide a normalize method to transform `WP code` to ISO 639-3 as much as possible.
   *
   * More detail refer to [languages scripts](https://github.com/yunsii/fasttext.wasm.js/tree/master/scripts/languages).
   *
   * Notice: ISO 639 provides two and three-character codes for representing names of languages. ISO 3166 provides two and three-character codes for representing names of countries.
   */
  static normalizeIdentifyLang(lang) {
    return languages.default[lang];
  }
  constructor(options) {
    const { getFastTextModule, wasmPath, modelPath } = options;
    if (!modelPath) {
      throw new Error("No model path provided.");
    }
    this.getFastTextModule = getFastTextModule;
    this.wasmPath = wasmPath;
    this.modelPath = modelPath;
  }
  /**
   * Use `lid.176.ftz` as default model
   */
  async load() {
    if (!this.model) {
      const FastText$1 = await FastText.getFastTextClass({
        getFastTextModule: () => {
          return this.getFastTextModule({
            wasmPath: this.wasmPath
          });
        }
      });
      const fastText = new FastText$1();
      const modelHref = this.modelPath;
      this.model = await fastText.loadModel(modelHref);
    }
    return this.model;
  }
  async identify(text, top) {
    const minTop = Math.max(top || 1, 1);
    const limitTop = Math.min(LANGUAGES_LIMIT, minTop);
    const vector = (await this.load()).predict(text, limitTop, 0);
    if (typeof top === "undefined") {
      return {
        ...BaseLanguageIdentificationModel.formatLang(vector.get(0)[1]),
        possibility: vector.get(0)[0]
      };
    }
    return Array.from({ length: vector.size() }).map((_, index) => {
      return {
        ...BaseLanguageIdentificationModel.formatLang(vector.get(index)[1]),
        possibility: vector.get(index)[0]
      };
    });
  }
}

exports.BaseLanguageIdentificationModel = BaseLanguageIdentificationModel;
