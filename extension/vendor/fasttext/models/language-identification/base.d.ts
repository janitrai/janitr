import type { IdentifyLangResult, RawIdentifyLang } from './types';
import type { FastTextModel } from '../../FastTextModel';
import type { InitializeFastTextModuleOptions } from '../../helpers/modules/types';
import type { GetFastTextModule } from '../../helpers/modules';
export interface BaseLanguageIdentificationModelOptions extends InitializeFastTextModuleOptions {
    getFastTextModule: GetFastTextModule;
    modelPath?: string;
}
export declare class BaseLanguageIdentificationModel {
    getFastTextModule: GetFastTextModule;
    wasmPath: InitializeFastTextModuleOptions['wasmPath'];
    modelPath: string;
    model: FastTextModel | null;
    static formatLang(raw: string): {
        /**
         * The three-letter 639-3 identifier.
         *
         * Attentions:
         *
         * - `eml` is retired in ISO 639-3
         * - `bih` and `nah` are ISO 639-2 codes, but not standard ISO 639-3 codes
         */
        alpha3: string;
        alpha2: string | null;
        /** refName: manually normalize rawLanguage fit https://iso639-3.sil.org/code_tables/download_tables */
        refName: string;
    };
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
    static normalizeIdentifyLang(lang: RawIdentifyLang): {
        /**
         * The three-letter 639-3 identifier.
         *
         * Attentions:
         *
         * - `eml` is retired in ISO 639-3
         * - `bih` and `nah` are ISO 639-2 codes, but not standard ISO 639-3 codes
         */
        alpha3: string;
        alpha2: string | null;
        /** refName: manually normalize rawLanguage fit https://iso639-3.sil.org/code_tables/download_tables */
        refName: string;
    };
    constructor(options: BaseLanguageIdentificationModelOptions);
    /**
     * Use `lid.176.ftz` as default model
     */
    load(): Promise<FastTextModel>;
    /** Return the top possibility language */
    identify(text: string, top: number): Promise<IdentifyLangResult[]>;
    /** Return the max possibility language */
    identify(text: string): Promise<IdentifyLangResult>;
}
