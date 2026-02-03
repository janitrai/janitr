'use strict';

Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });

const node = require('../../helpers/modules/node.cjs');
const base = require('./base.cjs');

class LanguageIdentificationModel extends base.BaseLanguageIdentificationModel {
  constructor(options = {}) {
    const modeRelativePath = "./assets/lid.176.ftz";
    const defaultModelHref = new URL(modeRelativePath, (typeof document === 'undefined' ? require('u' + 'rl').pathToFileURL(__filename).href : (document.currentScript && document.currentScript.src || new URL('models/language-identification/node.cjs', document.baseURI).href))).href;
    const { modelPath = defaultModelHref } = options;
    super({ modelPath, getFastTextModule: node.getFastTextModule });
  }
}
async function getLanguageIdentificationModel(options = {}) {
  return new LanguageIdentificationModel(options);
}
const getLIDModel = getLanguageIdentificationModel;

exports.LanguageIdentificationModel = LanguageIdentificationModel;
exports.getLIDModel = getLIDModel;
exports.getLanguageIdentificationModel = getLanguageIdentificationModel;
