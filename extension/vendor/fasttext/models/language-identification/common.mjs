import { getFastTextModule } from '../../helpers/modules/common.mjs';
import { BaseLanguageIdentificationModel } from './base.mjs';
import { commonAssetsDir, commonAssetsModelsDir } from '../../constants.mjs';

class LanguageIdentificationModel extends BaseLanguageIdentificationModel {
  constructor(options = {}) {
    const origin = typeof globalThis.location !== "undefined" ? globalThis.location.origin : "";
    super({
      wasmPath: `${origin}${commonAssetsDir}/fastText.common.wasm`,
      modelPath: `${origin}${commonAssetsModelsDir}/lid.176.ftz`,
      ...options,
      getFastTextModule
    });
  }
}
async function getLanguageIdentificationModel(options = {}) {
  return new LanguageIdentificationModel(options);
}
const getLIDModel = getLanguageIdentificationModel;

export { LanguageIdentificationModel, getLIDModel, getLanguageIdentificationModel };
