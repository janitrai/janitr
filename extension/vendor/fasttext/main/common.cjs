'use strict';

Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });

const common = require('../helpers/modules/common.cjs');
const FastText = require('../FastText.cjs');
const FastTextModel = require('../FastTextModel.cjs');
const common$1 = require('../models/language-identification/common.cjs');



exports.getFastTextModule = common.getFastTextModule;
exports.getFastTextClass = FastText.getFastTextClass;
exports.FastTextModel = FastTextModel.FastTextModel;
exports.getLIDModel = common$1.getLIDModel;
exports.getLanguageIdentificationModel = common$1.getLanguageIdentificationModel;
