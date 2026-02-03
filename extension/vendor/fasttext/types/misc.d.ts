export interface Vector<T> {
    clone(): Vector<T>;
    delete(): void;
    get(index: number): T;
    push_back(value: T): void;
    resize(size: number, value?: T): void;
    set(index: number, value: T): void;
    size(): number;
}
export type Pair<T1, T2> = [T1, T2];
