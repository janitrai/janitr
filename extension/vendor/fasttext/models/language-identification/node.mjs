import { getFastTextModule } from '../../helpers/modules/node.mjs';
import { BaseLanguageIdentificationModel } from './base.mjs';

class LanguageIdentificationModel extends BaseLanguageIdentificationModel {
  constructor(options = {}) {
    const modeRelativePath = "./assets/lid.176.ftz";
    const defaultModelHref = new URL(modeRelativePath, import.meta.url).href;
    const { modelPath = defaultModelHref } = options;
    super({ modelPath, getFastTextModule });
  }
}
async function getLanguageIdentificationModel(options = {}) {
  return new LanguageIdentificationModel(options);
}
const getLIDModel = getLanguageIdentificationModel;

export { LanguageIdentificationModel, getLIDModel, getLanguageIdentificationModel };
