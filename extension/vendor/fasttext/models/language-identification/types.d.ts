/** https://fasttext.cc/blog/2017/10/02/blog-post.html#iso-codes-of-languages-supported */
export type RawIdentifyLang = 'af' | 'als' | 'am' | 'an' | 'ar' | 'arz' | 'as' | 'ast' | 'av' | 'az' | 'azb' | 'ba' | 'bar' | 'bcl' | 'be' | 'bg' | 'bh' | 'bn' | 'bo' | 'bpy' | 'br' | 'bs' | 'bxr' | 'ca' | 'cbk' | 'ce' | 'ceb' | 'ckb' | 'co' | 'cs' | 'cv' | 'cy' | 'da' | 'de' | 'diq' | 'dsb' | 'dty' | 'dv' | 'el' | 'eml' | 'en' | 'eo' | 'es' | 'et' | 'eu' | 'fa' | 'fi' | 'fr' | 'frr' | 'fy' | 'ga' | 'gd' | 'gl' | 'gn' | 'gom' | 'gu' | 'gv' | 'he' | 'hi' | 'hif' | 'hr' | 'hsb' | 'ht' | 'hu' | 'hy' | 'ia' | 'id' | 'ie' | 'ilo' | 'io' | 'is' | 'it' | 'ja' | 'jbo' | 'jv' | 'ka' | 'kk' | 'km' | 'kn' | 'ko' | 'krc' | 'ku' | 'kv' | 'kw' | 'ky' | 'la' | 'lb' | 'lez' | 'li' | 'lmo' | 'lo' | 'lrc' | 'lt' | 'lv' | 'mai' | 'mg' | 'mhr' | 'min' | 'mk' | 'ml' | 'mn' | 'mr' | 'mrj' | 'ms' | 'mt' | 'mwl' | 'my' | 'myv' | 'mzn' | 'nah' | 'nap' | 'nds' | 'ne' | 'new' | 'nl' | 'nn' | 'no' | 'oc' | 'or' | 'os' | 'pa' | 'pam' | 'pfl' | 'pl' | 'pms' | 'pnb' | 'ps' | 'pt' | 'qu' | 'rm' | 'ro' | 'ru' | 'rue' | 'sa' | 'sah' | 'sc' | 'scn' | 'sco' | 'sd' | 'sh' | 'si' | 'sk' | 'sl' | 'so' | 'sq' | 'sr' | 'su' | 'sv' | 'sw' | 'ta' | 'te' | 'tg' | 'th' | 'tk' | 'tl' | 'tr' | 'tt' | 'tyv' | 'ug' | 'uk' | 'ur' | 'uz' | 'vec' | 'vep' | 'vi' | 'vls' | 'vo' | 'wa' | 'war' | 'wuu' | 'xal' | 'xmf' | 'yi' | 'yo' | 'yue' | 'zh';
export interface IdentifyLangVector {
    lang: RawIdentifyLang;
    possibility: number;
}
export interface IdentifyLangResult {
    /**
     * The three-letter 639-3 identifier.
     *
     * Attentions:
     *
     * - `eml` is retired in ISO 639-3
     * - `bih` and `nah` are ISO 639-2 codes, but not standard ISO 639-3 codes
     */
    alpha3: string;
    /** It will be `null` if `alpha3` equal `eml` or `nah` at least */
    alpha2: string | null;
    refName: string;
    possibility: number;
}
