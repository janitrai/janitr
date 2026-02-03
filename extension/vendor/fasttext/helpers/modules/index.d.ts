import type { InitializeFastTextModuleOptions } from './types';
import type { FastTextModule } from '../../core/fastText';
export declare const buffer2Uin8Array: (buf: Buffer) => Uint8Array;
export declare const fetchFile: (url: string) => Promise<Uint8Array>;
export type GetFastTextModule = (options?: InitializeFastTextModuleOptions) => Promise<FastTextModule>;
export type InternalGetFastTextModule = () => Promise<FastTextModule>;
