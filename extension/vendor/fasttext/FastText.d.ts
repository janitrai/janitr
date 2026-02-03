import { FastTextModel } from './FastTextModel';
import type { InternalGetFastTextModule } from './helpers/modules';
import type { FastTextCoreConstructor, FastTextModule } from './core/fastText';
export interface GetFastTextClassOptions {
    getFastTextModule: InternalGetFastTextModule;
}
export declare const getFastTextClass: (options: GetFastTextClassOptions) => Promise<{
    new (): {
        core: FastTextModule;
        ft: FastTextCoreConstructor;
        fs: FastTextModule['FS'];
        /**
         * loadModel
         *
         * Loads the model file from the specified url, and returns the
         * corresponding `FastTextModel` object.
         */
        loadModel(url: string): Promise<FastTextModel>;
        _train(url: RequestInfo | URL, modelName: string, kwargs?: Record<string, any>, callback?: (() => void) | null): Promise<unknown>;
        /**
         * trainSupervised
         *
         * Downloads the input file from the specified url, trains a supervised
         * model and returns a `FastTextModel` object.
         */
        trainSupervised(url: string, kwargs: Record<string, any> | undefined, callback: (() => void) | null | undefined): Promise<unknown>;
        /**
         * trainUnsupervised
         *
         * Downloads the input file from the specified url, trains an unsupervised
         * model and returns a `FastTextModel` object.
         *
         * @param {function}   callback
         *     train callback function
         *     `callback` function is called regularly from the train loop:
         *     `callback(progress, loss, wordsPerSec, learningRate, eta)`
         *
         * @return {Promise}   promise object that resolves to a `FastTextModel`
         *
         */
        trainUnsupervised(url: RequestInfo | URL, modelName: string, kwargs: {} | undefined, callback: (() => void) | null | undefined): Promise<unknown>;
    };
}>;
