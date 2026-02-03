'use strict';

Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });

const node = require('../helpers/modules/node.cjs');
const FastText = require('../FastText.cjs');
const FastTextModel = require('../FastTextModel.cjs');
const node$1 = require('../models/language-identification/node.cjs');



exports.getFastTextModule = node.getFastTextModule;
exports.getFastTextClass = FastText.getFastTextClass;
exports.FastTextModel = FastTextModel.FastTextModel;
exports.getLIDModel = node$1.getLIDModel;
exports.getLanguageIdentificationModel = node$1.getLanguageIdentificationModel;
