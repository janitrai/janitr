export type Engine = "fasttext" | "transformer" | "auto";

export type ScoreMap = Record<string, number>;

export interface ThresholdMap extends Record<string, number> {
  clean?: number;
  topic_crypto?: number;
  scam?: number;
  promo?: number;
}

export interface ClassifierResult {
  isFlagged: boolean;
  probability: number;
  threshold: number;
  thresholds: ThresholdMap;
  label: string;
  labels: string[];
  scores: ScoreMap;
  mode?: string;
  classes?: readonly string[];
}

export interface InferenceResponse {
  ok: boolean;
  results?: ClassifierResult[];
  engine?: string;
  fallbackFrom?: string | null;
  fallbackReason?: string;
  error?: string;
}
