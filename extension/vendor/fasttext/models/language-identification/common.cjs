'use strict';

Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });

const common = require('../../helpers/modules/common.cjs');
const base = require('./base.cjs');
const constants = require('../../constants.cjs');

class LanguageIdentificationModel extends base.BaseLanguageIdentificationModel {
  constructor(options = {}) {
    const origin = typeof globalThis.location !== "undefined" ? globalThis.location.origin : "";
    super({
      wasmPath: `${origin}${constants.commonAssetsDir}/fastText.common.wasm`,
      modelPath: `${origin}${constants.commonAssetsModelsDir}/lid.176.ftz`,
      ...options,
      getFastTextModule: common.getFastTextModule
    });
  }
}
async function getLanguageIdentificationModel(options = {}) {
  return new LanguageIdentificationModel(options);
}
const getLIDModel = getLanguageIdentificationModel;

exports.LanguageIdentificationModel = LanguageIdentificationModel;
exports.getLIDModel = getLIDModel;
exports.getLanguageIdentificationModel = getLanguageIdentificationModel;
