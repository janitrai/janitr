import { BaseLanguageIdentificationModel } from './base';
import type { BaseLanguageIdentificationModelOptions } from './base';
export interface LanguageIdentificationModelOptions extends Omit<BaseLanguageIdentificationModelOptions, 'getFastTextModule'> {
}
export declare class LanguageIdentificationModel extends BaseLanguageIdentificationModel {
    constructor(options?: LanguageIdentificationModelOptions);
}
export declare function getLanguageIdentificationModel(options?: LanguageIdentificationModelOptions): Promise<LanguageIdentificationModel>;
/** Alias of `getLanguageIdentificationModel` */
export declare const getLIDModel: typeof getLanguageIdentificationModel;
